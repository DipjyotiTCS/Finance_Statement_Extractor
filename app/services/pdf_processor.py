import os
import re
import json
import datetime
import fitz  # PyMuPDF
from typing import Optional, Tuple, Dict, Any, List
import sqlite3

from PIL import Image
import pytesseract
from pytesseract import Output

from ..db import get_conn, q, e
from .llm_client import LLMClient

HEADER_TOKEN = "Rakstrarroknskapur"
DB_PATH = "data/app.db"

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds")

def find_start_page(doc: fitz.Document) -> Optional[int]:
    """Return 0-based page index where the header line contains HEADER_TOKEN."""
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        head = lines[:12]
        if any(ln.startswith(HEADER_TOKEN) or ln == HEADER_TOKEN for ln in head):
            return i
    return None

def render_pages_to_png(doc: fitz.Document, start_idx: int, out_dir: str) -> List[Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    rendered = []
    for i in range(start_idx, doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # decent quality
        png_path = os.path.join(out_dir, f"page_{i+1:03d}.png")  # 1-based filename
        pix.save(png_path)

        # Generate word-level OCR coordinates alongside each PNG.
        # Output is stored as JSON with bounding boxes in PNG pixel coordinates.
        ocr_json_path = os.path.join(out_dir, f"page_{i+1:03d}.words.json")
        try:
            _generate_word_coords_json(png_path, ocr_json_path, page_number=i+1)
        except Exception as ex:
            # OCR is best-effort; do not fail the entire job if OCR isn't available.
            print("OCR coordinate generation failed for", png_path, "->", ex)
            ocr_json_path = ""

        rendered.append({"page_index": i, "page_number": i + 1, "png_path": png_path, "ocr_json_path": ocr_json_path})
    return rendered


def _normalize_token(s: str) -> str:
    s = (s or "").strip()
    return s


def _generate_word_coords_json(png_path: str, out_json_path: str, page_number: int) -> None:
    """Generate OCR JSON for UI highlighting using the same OCR logic as extraction."""
    from .ocr_utils import write_ocr_json
    write_ocr_json(png_path, out_json_path, page_number)

def get_png_path_for_page(db_path: str, job_id: str, page_index: int = 0) -> Optional[str]:
    """
    Fetch PNG path for a given job_id + page_index from SQLite.
    Adjust table/column names to match your schema.
    """
    sql = """
    SELECT png_path
    FROM job_pages
    WHERE job_id = ? AND page_number = ?
    LIMIT 1
    """
    print(db_path, job_id, page_index)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(sql, (job_id, page_index)).fetchone()
        return row[0] if row else None

def _safe_year(y: Optional[str]) -> Optional[str]:
    if not y:
        return None
    m = re.search(r"\b(19\d{2}|20\d{2}|2100)\b", str(y))
    return m.group(0) if m else None

def process_job(db_path: str, job_id: int) -> None:
    conn = get_conn(db_path)
    llm = LLMClient()
    try:
        job = q(conn, "SELECT * FROM jobs WHERE id = ?", (job_id,))
        if not job:
            return
        job = job[0]
        pdf_path = job["pdf_path"]

        with fitz.open(pdf_path) as doc:
            start_idx = 0
            temp_folder = os.path.join(os.path.dirname(pdf_path), f"pages_from_{start_idx+1:03d}")
            rendered = render_pages_to_png(doc, start_idx, temp_folder)

            for ren1 in rendered:
                e(conn,
                  "INSERT INTO job_pages(job_id, page_number, png_path, extracted_json, created_at) VALUES (?,?,?,?,?)",
                  (job_id, int(ren1.get("page_number")), str(ren1.get("png_path")), "", _now_iso())
                )

                # Link OCR coordinate file (word-level bounding boxes) to the page.
                ocr_path = ren1.get("ocr_json_path") or ""
                if ocr_path:
                    e(conn,
                      "INSERT OR REPLACE INTO job_page_ocr(job_id, page_number, ocr_json_path, created_at) VALUES (?,?,?,?)",
                      (job_id, int(ren1.get("page_number")), str(ocr_path), _now_iso())
                    )

            initial_page = next((r.get("png_path") for r in rendered if r.get("page_number") == 1), None)

            print("Rendered initial page", initial_page)
            # Update job with temp folder + range
            e(conn, "UPDATE jobs SET temp_folder = ?, start_page = ?, end_page = ? WHERE id = ?",
              (temp_folder, start_idx + 1, doc.page_count, job_id))

            # Phase 1: per-page extraction (PNG -> JSON)
            meta = llm._openai_extract_company_year_from_png(
                png_path=initial_page
            )
            e(conn, "INSERT OR REPLACE INTO job_meta(job_id, company_name, publication_year, publication_date) VALUES (?,?,?,?)",
              (job_id, meta.get("company_name"), meta.get("publication_year"), meta.get("publication_date")))
            
            # Phase 2: per-page extraction (PNG -> JSON)
            #From the render check first 3 pages if they have TOC
            #Once TOC is found identify the page number of the keyword.
            #Establish a list if pages which needs to be parsed.

            toc_payload = []
            for ren in rendered:
                if ren.get("page_number") < 3 :
                    toc_payload.append({"page_number": ren.get("page_number"), "png_path": ren.get("png_path")})

            toc_details = llm.fetch_toc_json(
                toc_payload = toc_payload
            )

            processing_page = toc_details.get("page_number")

            extraction_payload = []
            for ren2 in rendered:
                if ren2.get("page_number") > int(processing_page):
                    extraction_payload.append({"page_number": ren2.get("page_number"), "png_path": ren2.get("png_path")})

            llm.process_pages(job_id, extraction_payload)
            

            #Phase 3: Parse that list and formulate the json for front end

        e(conn, "UPDATE jobs SET status = ? WHERE id = ?", ("Complete", job_id))

    except Exception as ex:
        print("Exception occured", ex)
        e(conn, "UPDATE jobs SET status = ?, error = ? WHERE id = ?", ("Failed", str(ex), job_id))
    finally:
        conn.close()
