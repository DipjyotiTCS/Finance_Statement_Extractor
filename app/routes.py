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


    # --- Header table values ---
    # sqlite3.Row behaves like a mapping but does not implement .get()
    pdf_filename = (job["pdf_filename"] if (hasattr(job, "keys") and "pdf_filename" in job.keys()) else "") or ""
    pdf_idnumber = None
    if pdf_filename:
        base = os.path.basename(pdf_filename)
        m = re.match(r"(\d+)[_\-].*\.pdf$", base, flags=re.IGNORECASE)
        if m:
            pdf_idnumber = m.group(1)

    publication_date = None
    if meta and (hasattr(meta, "keys") and "publication_date" in meta.keys()) and meta["publication_date"]:
        # expecting YYYYMMDD
        publication_date = re.sub(r"\D", "", str(meta["publication_date"]))
        if len(publication_date) != 8:
            publication_date = None

    # Random 1-year period ending before publication_date (if available)
    period_start = None
    period_end = None
    if publication_date:
        try:
            pub_dt = datetime.datetime.strptime(publication_date, "%Y%m%d").date()
            # pick a start date between pub_dt-3y and pub_dt-1y-5d
            latest_start = pub_dt - datetime.timedelta(days=370)  # ensure end < pub
            earliest_start = pub_dt - datetime.timedelta(days=365*3)
            if earliest_start < latest_start:
                span = (latest_start - earliest_start).days
                start_dt = earliest_start + datetime.timedelta(days=random.randint(0, max(span, 0)))
            else:
                start_dt = pub_dt - datetime.timedelta(days=730)
            end_dt = start_dt + datetime.timedelta(days=365)
            # Ensure strictly less than publication date
            if end_dt >= pub_dt:
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
    Consolidate extracted_json across all pages for a job and download as CSV.

    CSV format:
    - Header is: year,<all unique translated item labels across pages>
      (uses row['Experian Value'] when present, otherwise falls back to the original row key)
    - Each row is a year with corresponding values per item.
    """
    conn = get_conn(current_app.config["DB_PATH"])
    try:
        job_rows = q(conn, "SELECT * FROM jobs WHERE id = ?", (job_id,))
        if not job_rows:
            abort(404)

        page_rows = q(
            conn,
            "SELECT page_number, extracted_json FROM job_pages WHERE job_id = ? ORDER BY page_number ASC",
            (job_id,),
        )
    finally:
        conn.close()

    # Column labels (in first-seen order)
    cols_in_order: list[str] = []
    col_seen: set[str] = set()

    # Year ordering: prefer page-provided 'years' arrays, otherwise infer from year keys
    years_in_order: list[str] = []
    year_seen: set[str] = set()

    # Values: year -> {col_label: value}
    values_by_year: dict[str, dict[str, object]] = {}

    def _is_year_key(k: object) -> bool:
        return isinstance(k, str) and k.isdigit() and len(k) == 4

    for pr in page_rows:
        # sqlite3.Row does not implement .get(); use key access
        raw_val = ""
        try:
            raw_val = pr["extracted_json"] if "extracted_json" in pr.keys() else ""
        except Exception:
            raw_val = ""
        raw = (raw_val or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue

        # Capture explicit years ordering if present
        yrs = obj.get("years")
        if isinstance(yrs, list):
            for y in yrs:
                if _is_year_key(y) and y not in year_seen:
                    year_seen.add(y)
                    years_in_order.append(y)

        rows_obj = obj.get("rows")
        if not isinstance(rows_obj, dict):
            continue

        for item_name, item_obj in rows_obj.items():
            if not isinstance(item_obj, dict):
                continue

            # Column header label: prefer translated "Experian Value"
            label = item_obj.get("Experian Value")
            if not isinstance(label, str) or not label.strip():
                label = item_name if isinstance(item_name, str) else ""
            label = str(label).strip()

            if not label:
                continue

            if label not in col_seen:
                col_seen.add(label)
                cols_in_order.append(label)

            # Add year values
            for k, v in item_obj.items():
                if not _is_year_key(k):
                    continue
                year = k
                if year not in year_seen:
                    year_seen.add(year)
                    years_in_order.append(year)

                values_by_year.setdefault(year, {})
                existing = values_by_year[year].get(label, "")

                # Keep first non-empty value if duplicates occur across pages
                if (existing in ("", None)) and (v not in ("", None)):
                    values_by_year[year][label] = v
                elif label not in values_by_year[year]:
                    values_by_year[year][label] = v

    if not cols_in_order:
        return jsonify({"error": "No extracted_json found for this job."}), 400

    # If we never got an explicit 'years' list, order years descending for nicer financial statements
    if years_in_order:
        years_final = [y for y in years_in_order if y in values_by_year or y in year_seen]
    else:
        years_final = sorted(values_by_year.keys(), reverse=True)

    out = io.StringIO()
    # Produce a "long" CSV that Excel opens cleanly:
    # Experian Taxonomy, Value <year1>, Value <year2>
    # (year1/year2 are fixed to 2024/2023 as requested; if missing, cells remain blank)
    y1, y2 = "2024", "2023"

    writer = csv.writer(out)  # default delimiter=',' (Excel-friendly with UTF-8 BOM below)
    writer.writerow(["Experian Taxonomy", f"Value {y1}", f"Value {y2}"])

    bucket_y1 = values_by_year.get(y1, {})
    bucket_y2 = values_by_year.get(y2, {})

    for taxonomy in cols_in_order:
        v1 = bucket_y1.get(taxonomy, "")
        v2 = bucket_y2.get(taxonomy, "")

        # Preserve separators as-is; stringify complex values
        if isinstance(v1, (dict, list)):
            v1 = json.dumps(v1, ensure_ascii=False)
        if isinstance(v2, (dict, list)):
            v2 = json.dumps(v2, ensure_ascii=False)

        writer.writerow([taxonomy, v1, v2])

    data = out.getvalue().encode("utf-8-sig")  # Excel-friendly BOM
    bio = io.BytesIO(data)
    bio.seek(0)
    filename = f"job_{job_id}_extracted.csv"
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