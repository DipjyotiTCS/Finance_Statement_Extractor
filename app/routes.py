import os
import io
import csv
import threading
import datetime
import json
import re
import random
from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash, send_file, jsonify, abort

from .db import get_conn, q, e
from .services.pdf_processor import process_job
from .services.llm_client import LLMClient

bp = Blueprint("main", __name__)

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds")

@bp.get("/")
def index():
    conn = get_conn(current_app.config["DB_PATH"])
    try:
        jobs = q(conn, "SELECT * FROM jobs ORDER BY id DESC LIMIT 50")
    finally:
        conn.close()
    return render_template("index.html", jobs=jobs)

@bp.post("/upload")
def upload():
    if "pdf" not in request.files:
        flash("No file part", "danger")
        return redirect(url_for("main.index"))

    f = request.files["pdf"]
    if not f.filename.lower().endswith(".pdf"):
        flash("Please upload a PDF file.", "danger")
        return redirect(url_for("main.index"))

    # Create a job folder under data/jobs/<timestamp>_<rand>
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_dir = os.path.join(current_app.config["JOBS_DIR"], f"job_{ts}")
    os.makedirs(job_dir, exist_ok=True)

    pdf_path = os.path.join(job_dir, f.filename)
    f.save(pdf_path)

    conn = get_conn(current_app.config["DB_PATH"])
    try:
        job_id = e(conn,
                   "INSERT INTO jobs(created_at, status, pdf_filename, pdf_path) VALUES (?,?,?,?)",
                   (_now_iso(), "Work in progress", f.filename, pdf_path))
    finally:
        conn.close()

    # Start background processing
    t = threading.Thread(target=process_job, args=(current_app.config["DB_PATH"], job_id), daemon=True)
    t.start()

    flash(f"Job {job_id} started.", "success")
    # Stay on the home page; the new job row will appear in the uploads list.
    return redirect(url_for("main.index"))

@bp.get("/job/<int:job_id>")
def job_detail(job_id: int):
    conn = get_conn(current_app.config["DB_PATH"])
    try:
        job_rows = q(conn, "SELECT * FROM jobs WHERE id = ?", (job_id,))
        if not job_rows:
            abort(404)
        job = job_rows[0]
        meta = q(conn, "SELECT * FROM job_meta WHERE job_id = ?", (job_id,))
        meta = meta[0] if meta else None
        pages = q(conn, "SELECT * FROM job_pages WHERE job_id = ? ORDER BY page_number ASC", (job_id,))
    finally:
        conn.close()


    # --- Header table values (persisted in job_meta; fallback if missing) ---
    kob_no = "KOB-0001"
    account_type = "NOR"
    account_class = "B"
    currency_code = "DKK"

    pdf_filename = (job["pdf_filename"] if (hasattr(job, "keys") and "pdf_filename" in job.keys()) else "") or ""
    pdf_idnumber = None
    if meta and (hasattr(meta, "keys") and "pdf_idnumber" in meta.keys()) and meta["pdf_idnumber"]:
        pdf_idnumber = meta["pdf_idnumber"]
    elif pdf_filename:
        base = os.path.basename(pdf_filename)
        m = re.match(r"(\d+)[_\-].*\.pdf$", base, flags=re.IGNORECASE)
        if m:
            pdf_idnumber = m.group(1)

    publication_date = None
    if meta and (hasattr(meta, "keys") and "publication_date" in meta.keys()) and meta["publication_date"]:
        publication_date = re.sub(r"\D", "", str(meta["publication_date"]))
        if len(publication_date) != 8:
            publication_date = None

    period_start = None
    period_end = None
    if meta and (hasattr(meta, "keys") and "period_start" in meta.keys()) and meta["period_start"]:
        period_start = str(meta["period_start"])
    if meta and (hasattr(meta, "keys") and "period_end" in meta.keys()) and meta["period_end"]:
        period_end = str(meta["period_end"])

    # Deterministic fallback if period isn't persisted yet
    if publication_date and (not period_start or not period_end):
        try:
            pub_dt = datetime.datetime.strptime(publication_date, "%Y%m%d").date()
            end_dt = pub_dt - datetime.timedelta(days=1)
            start_dt = end_dt - datetime.timedelta(days=365)
            period_start = start_dt.strftime("%Y%m%d")
            period_end = end_dt.strftime("%Y%m%d")
        except Exception:
            pass
# Use a fluid container for the modern viewer layout.
    return render_template("job.html", job=job, meta=meta, pages=pages, container_class="container-fluid", kob_no="KOB-0001", pdf_idnumber=pdf_idnumber, publication_date=publication_date, period_start=period_start, period_end=period_end)

@bp.get("/job/<int:job_id>/page/<int:page_number>/png")
def job_page_png(job_id: int, page_number: int):
    conn = get_conn(current_app.config["DB_PATH"])
    try:
        rows = q(conn, "SELECT png_path FROM job_pages WHERE job_id = ? AND page_number = ? LIMIT 1", (job_id, page_number))
        if not rows:
            abort(404)
        png_path = rows[0]["png_path"]
    finally:
        conn.close()
    return send_file(png_path, mimetype="image/png")

@bp.get("/job/<int:job_id>/page/<int:page_number>/json")
def job_page_json(job_id: int, page_number: int):
    conn = get_conn(current_app.config["DB_PATH"])
    try:
        rows = q(conn, "SELECT extracted_json FROM job_pages WHERE job_id = ? AND page_number = ? LIMIT 1", (job_id, page_number))
        if not rows:
            abort(404)
        return jsonify({"job_id": job_id, "page_number": page_number, "extracted": rows[0]["extracted_json"]})
    finally:
        conn.close()

@bp.get("/job/<int:job_id>/download/csv")
def job_download_csv(job_id: int):
    """
    Download a populated Experian taxonomy CSV.

    This uses the provided taxonomy template (semicolon-delimited) and fills the
    matching fields using consolidated extracted_json from all pages.

    Matching strategy:
    1) Direct match: normalized extracted "Experian Value" == normalized taxonomy header
    2) Heuristic shortlist (difflib)
    3) LLM selection over the shortlist (cached in sqlite table taxonomy_match_cache)
    """
    conn = get_conn(current_app.config["DB_PATH"])
    try:
        job_rows = q(conn, "SELECT * FROM jobs WHERE id = ?", (job_id,))
        if not job_rows:
            abort(404)

        meta_rows = q(conn, "SELECT * FROM job_meta WHERE job_id = ? LIMIT 1", (job_id,))
        meta = meta_rows[0] if meta_rows else None

        page_rows = q(
            conn,
            "SELECT page_number, extracted_json FROM job_pages WHERE job_id = ? ORDER BY page_number ASC",
            (job_id,),
        )
    finally:
        conn.close()

    # --- Load taxonomy template header order ---
    tmpl_path = current_app.config.get("TAXONOMY_TEMPLATE_CSV", "").strip()
    if not tmpl_path:
        tmpl_path = os.path.join(os.path.dirname(__file__), "..", "taxonomy_template.csv")
        tmpl_path = os.path.abspath(tmpl_path)

    if not os.path.exists(tmpl_path):
        return jsonify({"error": f"Taxonomy template CSV not found at {tmpl_path}"}), 400

    with open(tmpl_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = next(reader, None) or []

    headers = [h.strip() for h in headers if str(h).strip()]
    if not headers:
        return jsonify({"error": "Taxonomy template CSV has no headers."}), 400

    # --- Consolidate extracted values across pages ---
    values_by_label: dict[str, dict[str, object]] = {}
    years_seen: set[str] = set()
    years_list: list[str] = []

    def _is_year_key(k: object) -> bool:
        return isinstance(k, str) and k.isdigit() and len(k) == 4

    def _norm(s: str) -> str:
        s = (s or "").lower().strip()
        s = s.replace("&", "and")
        return re.sub(r"[^a-z0-9]+", "", s)

    for pr in page_rows:
        raw = ""
        try:
            raw = (pr["extracted_json"] or "").strip()
        except Exception:
            raw = ""
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        yrs = obj.get("years")
        if isinstance(yrs, list):
            for y in yrs:
                if _is_year_key(y) and y not in years_seen:
                    years_seen.add(y)
                    years_list.append(y)

        rows_obj = obj.get("rows")
        if not isinstance(rows_obj, dict):
            continue

        for item_name, item_obj in rows_obj.items():
            if not isinstance(item_obj, dict):
                continue

            label = item_obj.get("Experian Value")
            if not isinstance(label, str) or not label.strip():
                label = item_name if isinstance(item_name, str) else ""
            label = str(label).strip()
            if not label:
                continue

            bucket = values_by_label.setdefault(label, {})
            for k, v in item_obj.items():
                if not _is_year_key(k):
                    continue
                years_seen.add(k)
                # Keep the first non-empty value per (label, year)
                if (bucket.get(k) in ("", None)) and (v not in ("", None)):
                    bucket[k] = v
                elif k not in bucket:
                    bucket[k] = v

    if not values_by_label:
        return jsonify({"error": "No extracted_json found for this job."}), 400

    # Decide which year to populate when template has no year-specific columns:
    # prefer most recent year present in the job.
    years_all = sorted([y for y in years_seen if _is_year_key(y)], reverse=True)
    primary_year = years_all[0] if years_all else None

    # --- Build direct lookup map for headers ---
    header_norm_map: dict[str, str] = {}
    for h in headers:
        header_norm_map[_norm(h)] = h

    llm = LLMClient()
    row_out: dict[str, str] = {h: "" for h in headers}

    # --- Populate header fields (shown on job.html) into the output CSV ---
    # Prefer persisted values from job_meta; fallback to the same deterministic derivation.
    try:
        kob_no = (meta["kob_no"] if meta and "kob_no" in meta.keys() else None) or "KOB-0001"
        pdf_idnumber = (meta["pdf_idnumber"] if meta and "pdf_idnumber" in meta.keys() else None)
        account_type = (meta["account_type"] if meta and "account_type" in meta.keys() else None) or "NOR"
        publication_date = (meta["publication_date"] if meta and "publication_date" in meta.keys() else None)
        period_start = (meta["period_start"] if meta and "period_start" in meta.keys() else None)
        period_end = (meta["period_end"] if meta and "period_end" in meta.keys() else None)
        account_class = (meta["account_class"] if meta and "account_class" in meta.keys() else None) or "B"
        currency_code = (meta["currency_code"] if meta and "currency_code" in meta.keys() else None) or "DKK"
    except Exception:
        kob_no, pdf_idnumber, account_type, publication_date, period_start, period_end, account_class, currency_code = "KOB-0001", None, "NOR", None, None, None, "B", "DKK"

    # Fallback derive pdf_idnumber from filename if not persisted
    if not pdf_idnumber:
        try:
            pdf_filename = (job_rows[0]["pdf_filename"] or "")
            base = os.path.basename(pdf_filename)
            m = re.match(r"(\d+)[_\-].*\.pdf$", base, flags=re.IGNORECASE)
            if m:
                pdf_idnumber = m.group(1)
        except Exception:
            pass

    # Normalize publication_date to YYYYMMDD if possible
    pub_yyyymmdd = None
    if publication_date:
        try:
            pub_yyyymmdd = re.sub(r"\D", "", str(publication_date))
            if len(pub_yyyymmdd) != 8:
                pub_yyyymmdd = None
        except Exception:
            pub_yyyymmdd = None

    # Deterministic period if not present
    if pub_yyyymmdd and (not period_start or not period_end):
        try:
            pub_dt = datetime.datetime.strptime(pub_yyyymmdd, "%Y%m%d").date()
            end_dt = pub_dt - datetime.timedelta(days=1)
            start_dt = end_dt - datetime.timedelta(days=365)
            period_start = start_dt.strftime("%Y%m%d")
            period_end = end_dt.strftime("%Y%m%d")
        except Exception:
            pass

    # Set into CSV if these columns exist in the template
    if "KOB_no" in row_out:
        row_out["KOB_no"] = str(kob_no or "")
    if "PDF_IDnumber" in row_out:
        row_out["PDF_IDnumber"] = str(pdf_idnumber or "")
    if "Account_type" in row_out:
        row_out["Account_type"] = str(account_type or "")
    if "Publication_date" in row_out:
        row_out["Publication_date"] = str(pub_yyyymmdd or publication_date or "")
    if "Period_start" in row_out:
        row_out["Period_start"] = str(period_start or "")
    if "Period_end" in row_out:
        row_out["Period_end"] = str(period_end or "")
    if "Account_class" in row_out:
        row_out["Account_class"] = str(account_class or "")
    if "Currency_code" in row_out:
        row_out["Currency_code"] = str(currency_code or "")

    # Preserve IDs if the job has anything relevant (optional; keep blanks otherwise)
    # If you later extract these from the PDF, set them here as well.

    # --- Populate fields ---
    import difflib

    header_norms = list(header_norm_map.keys())

    for extracted_label, year_map in values_by_label.items():
        if not isinstance(year_map, dict):
            continue

        # choose value from primary year, otherwise take any year (most recent first)
        val = ""
        if primary_year and primary_year in year_map:
            val = year_map.get(primary_year, "")
        else:
            # fallback: pick first available year value
            for y in years_all:
                if y in year_map:
                    val = year_map.get(y, "")
                    break
            if val in ("", None) and year_map:
                # any value
                val = next(iter(year_map.values()))

        if val in ("", None):
            continue

        # direct match
        n = _norm(extracted_label)
        direct = header_norm_map.get(n)

        matched_field = None
        if direct:
            matched_field = direct
        else:
            # shortlist using difflib ratio over normalized strings
            scored = []
            for hn in header_norms:
                scored.append((difflib.SequenceMatcher(None, n, hn).ratio(), header_norm_map[hn]))
            scored.sort(reverse=True, key=lambda t: t[0])
            shortlist = [s[1] for s in scored[:20]]

            # Ask LLM to select from shortlist (cached)
            ctx = f"primary_year={primary_year}" if primary_year else ""
            out = llm.match_taxonomy_field(
                db_path=current_app.config["DB_PATH"],
                source_label=extracted_label,
                candidate_fields=shortlist,
                context_hint=ctx,
            )
            if isinstance(out, dict) and out.get("match") in shortlist:
                matched_field = out.get("match")

        if matched_field and matched_field in row_out:
            # stringify complex values
            if isinstance(val, (dict, list)):
                val = json.dumps(val, ensure_ascii=False)
            row_out[matched_field] = str(val)

    # --- Write CSV in the exact template format (semicolon, UTF-8 BOM for Excel) ---
    out_io = io.StringIO()
    writer = csv.writer(out_io, delimiter=";")
    writer.writerow(headers)
    writer.writerow([row_out.get(h, "") for h in headers])

    data = out_io.getvalue().encode("utf-8-sig")
    bio = io.BytesIO(data)
    bio.seek(0)
    filename = f"job_{job_id}_experian_taxonomy.csv"
    return send_file(
        bio,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
        max_age=0,
    )


@bp.get("/job/<int:job_id>/page/<int:page_number>/ocr")
def job_page_ocr(job_id: int, page_number: int):
    """Return word-level OCR boxes for the rendered PNG page."""
    conn = get_conn(current_app.config["DB_PATH"])
    try:
        rows = q(conn, "SELECT ocr_json_path FROM job_page_ocr WHERE job_id = ? AND page_number = ? LIMIT 1", (job_id, page_number))
        if not rows:
            # If OCR hasn't been generated for this page, return empty payload.
            return jsonify({"job_id": job_id, "page_number": page_number, "words": [], "image": None})
        ocr_path = rows[0]["ocr_json_path"]
    finally:
        conn.close()

    try:
        with open(ocr_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        payload = {"page_number": page_number, "image": None, "words": []}

    # Normalize shape in response
    return jsonify({
        "job_id": job_id,
        "page_number": page_number,
        "image": payload.get("image"),
        "words": payload.get("words") or [],
    })

@bp.get("/api/jobs")
def api_jobs():
    conn = get_conn(current_app.config["DB_PATH"])
    try:
        jobs = [dict(r) for r in q(conn, "SELECT * FROM jobs ORDER BY id DESC LIMIT 200")]
    finally:
        conn.close()
    return jsonify(jobs)

@bp.get("/api/job/<int:job_id>")
def api_job(job_id: int):
    conn = get_conn(current_app.config["DB_PATH"])
    try:
        job_rows = q(conn, "SELECT * FROM jobs WHERE id = ?", (job_id,))
        if not job_rows:
            abort(404)
        job = dict(job_rows[0])
        meta_rows = q(conn, "SELECT * FROM job_meta WHERE job_id = ?", (job_id,))
        meta = dict(meta_rows[0]) if meta_rows else None
        pages = [dict(r) for r in q(conn, "SELECT page_number, png_path, extracted_json, created_at FROM job_pages WHERE job_id = ? ORDER BY page_number", (job_id,))]
    finally:
        conn.close()
    return jsonify({"job": job, "meta": meta, "pages": pages})


@bp.post("/api/job/<int:job_id>/page/<int:page_number>/item")
def api_update_item(job_id: int, page_number: int):
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode")
    item = payload.get("item")
    if not item:
        return jsonify({"error": "Missing item"}), 400

    new_value = payload.get("value", "")
    new_type = payload.get("item_type", "")
    new_conf = payload.get("confidence_score", "")
    updates = payload.get("updates") or {}

    conn = get_conn(current_app.config["DB_PATH"])
    try:
        rows = q(conn, "SELECT extracted_json FROM job_pages WHERE job_id = ? AND page_number = ? LIMIT 1", (job_id, page_number))
        if not rows:
            abort(404)
        raw = rows[0]["extracted_json"] or ""

        # Parse existing JSON if possible; otherwise keep as dict.
        extracted_obj = {}
        parsed_ok = False
        if raw.strip():
            try:
                extracted_obj = json.loads(raw)
                parsed_ok = isinstance(extracted_obj, (dict, list))
            except Exception:
                extracted_obj = {}

        # --- Financial-table mode: extracted_obj has {page_title, rows:{ item:{YYYY:..., nota:'', confidence_score:..., bbox? }}}
        if mode == "financial" and isinstance(extracted_obj, dict) and isinstance(extracted_obj.get("rows"), dict):
            rows_obj = extracted_obj.get("rows") or {}
            row_obj = rows_obj.get(item)
            if not isinstance(row_obj, dict):
                row_obj = {}

            # Track originals per-field once (for inline diff)
            original_map = row_obj.get("_original")
            if not isinstance(original_map, dict):
                original_map = {}

            years_updates = (updates.get("years") or {}) if isinstance(updates, dict) else {}
            for y, v in years_updates.items():
                if isinstance(y, str) and y.isdigit():
                    if y not in original_map:
                        original_map[y] = row_obj.get(y, "")
                    row_obj[y] = v

            if "nota" in updates:
                if "nota" not in original_map:
                    original_map["nota"] = row_obj.get("nota", "")
                row_obj["nota"] = updates.get("nota", "")

            if "confidence_score" in updates:
                if "confidence_score" not in original_map:
                    original_map["confidence_score"] = row_obj.get("confidence_score", "")
                row_obj["confidence_score"] = updates.get("confidence_score", "")

            row_obj["_original"] = original_map
            rows_obj[item] = row_obj
            extracted_obj["rows"] = rows_obj

            new_raw = json.dumps(extracted_obj, ensure_ascii=False)
            e(conn, "UPDATE job_pages SET extracted_json = ? WHERE job_id = ? AND page_number = ?", (new_raw, job_id, page_number))

            return jsonify({"ok": True, "updated": {"item": item}})

        # --- Default KV mode: top-level dict items
        if not isinstance(extracted_obj, dict):
            extracted_obj = {"_raw": raw}

        existing = extracted_obj.get(item)

        if isinstance(existing, dict):
            existing_obj = existing
        else:
            existing_obj = {"value": existing if existing is not None else ""}

        if "_original" not in existing_obj:
            existing_obj["_original"] = existing_obj.get("value", "")

        existing_obj["value"] = new_value
        existing_obj["item_type"] = new_type
        existing_obj["confidence_score"] = new_conf

        extracted_obj[item] = existing_obj

        new_raw = json.dumps(extracted_obj, ensure_ascii=False)
        e(conn, "UPDATE job_pages SET extracted_json = ? WHERE job_id = ? AND page_number = ?", (new_raw, job_id, page_number))

        updated = {
            "item": item,
            "value": existing_obj.get("value", ""),
            "item_type": existing_obj.get("item_type", ""),
            "confidence_score": existing_obj.get("confidence_score", ""),
            "original_value": existing_obj.get("_original", ""),
            "bbox": existing_obj.get("bbox"),
        }
        return jsonify({"ok": True, "updated": updated})
    finally:
        conn.close()