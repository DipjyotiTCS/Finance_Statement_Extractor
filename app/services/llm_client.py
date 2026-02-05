import os
import re
import json
import base64
from typing import Dict, Any, Optional, List, Set
from openai import OpenAI
import sqlite3
import cv2
import pytesseract

from ..db import get_conn, q, e
from copy import deepcopy

class LLMClient:

    def __init__(self):
        self.client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").strip())
        self.openai_model = os.environ.get("OPENAI_MODEL_VISION", "gpt-4o-mini").strip()
        self.mode = os.environ.get("LLM_MODE", "openai").strip().lower()
        self._openai_client = self.client


    pytesseract.pytesseract.tesseract_cmd = (
        r"C:/Program Files/Tesseract-OCR/tesseract.exe"
    )
    # -------------------------
    # Public API (used by app)
    # -------------------------
    def extract_page_json(self, png_path: str, page_number: int, page_text: str) -> Dict[str, Any]:
        """
        Extract per-page JSON.

        - In mock mode: regex-based, deterministic extraction from OCR/text.
        - In openai mode: you can enable image-based extraction by setting LLM_MODE=openai.
          If OpenAI is unavailable at runtime, it automatically falls back to mock.
        """
        if self.mode == "openai" and self._openai_client is not None:
            try:
                return self._openai_extract_page_json(png_path, page_number)
            except Exception:
                # Safety fallback; never hard-fail the pipeline.
                return self._mock_extract_page(png_path, page_number, page_text)

        return self._mock_extract_page(png_path, page_number, page_text)
    
    # --------------------
    # OpenAI (Vision) impl
    # --------------------
    def _png_to_data_url(self, png_path: str) -> str:
        with open(png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{b64}"

    def _openai_extract_company_year_from_png(self, png_path: str) -> Dict[str, Any]:
        """
        Uses OpenAI Responses API (vision) in JSON mode.
        Returns dict with keys: company_name, publication_year, publication_date
        """
        prompt = (
            "You are given the FIRST PAGE of a PDF as an image.\n\n"
            "Extract:\n"
            "1) company_name: the primary company/publisher/organization name on the page.\n"
            "2) publication_year: the 4-digit publication/report/Ársfrásøgn year (e.g., Ársfrásøgn 2024. Return only the 4 digit year).\n"
            "3) publication_date: the publication date printed on the page (if present). Return as YYYYMMDD (digits only).\n\n"

            "Return ONLY valid JSON with EXACTLY these keys:\n"
            '{ "company_name": string|null, "publication_year": string|null, "publication_date": string|null }\n\n'
            "Rules:\n"
            "- Use the exact text as printed.\n"
            "- If multiple years exist, choose the most relevant publication/report year.\n"
            "- If uncertain, use null.\n"
            "- Do not add extra keys. Do not include explanations."
        )

        content=[
            {"type":"text","text":(prompt)},
            {"type":"image_url","image_url":{"url": self._png_to_data_url(png_path)}}
        ]

        resp=self.client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role":"user","content":content}],
            response_format={"type":"json_object"},
        )

        return json.loads(resp.choices[0].message.content or "{}")
    
    def fetch_toc_json(self, toc_payload):

        prompt = (
            """
            You are given first few pages of a PDF as image.
            Identify the 'Innihaldsyvirlit' page and from that page Extract:
            1) Rakstrarroknskapur page number: the page number of the income statement.
            
            Return ONLY valid JSON with EXACTLY these keys:
            { "toc_term": should always be Rakstrarroknskapur, "page_number": string|null }
            Rules:
            - Use the exact text as printed.
            - If uncertain, use null.
            - Do not add extra keys. Do not include explanations.
            """
        )

        content=[
            {"type":"text","text":(prompt)},
        ]

        for tp in toc_payload:
            if(tp.get("page_number") > 1):
                content.append({"type":"image_url","image_url":{"url": self._png_to_data_url(tp.get("png_path"))}})

        resp=self.client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role":"user","content":content}],
            response_format={"type":"json_object"},
        )

        return json.loads(resp.choices[0].message.content or "{}")
         
    def process_pages(self, job_id, extraction_payload):
        """
        New extraction flow:
        - Run Tesseract OCR on each PNG page to produce an OCR JSON payload (no image sent to LLM).
        - Send the OCR JSON to LLM to construct the final extracted JSON in the required schema.
        """
        for ep in extraction_payload:
            png_path = ep.get("png_path")
            page_number = int(ep.get("page_number") or 0)
            try:
                ocr_payload = self.ocr_page_to_json(str(png_path), page_number=page_number)

                print("Calling LLM to get OCR data extraction")
                ocr_extracted = self._openai_extract_from_ocr_json(ocr_payload)

                print("Calling LLM to get visual data extraction")
                llm_extarcted = self.process_other_pages(job_id, page_number, png_path)

                print("Performing QC for json check")
                extracted = self.verify_ocr_against_llm(ocr_extracted, llm_extarcted)

                # Persist extracted JSON back to DB
                db_path = "data/app.db"
                conn = get_conn(db_path)
                try:
                    e(
                        conn,
                        "UPDATE job_pages SET extracted_json = ? WHERE job_id = ? AND page_number = ?",
                        (json.dumps(extracted, ensure_ascii=False), int(job_id), int(page_number)),
                    )
                    conn.commit()
                finally:
                    conn.close()

            except Exception as ex:
                print("exception occurred while extracting records from page", page_number, ex)

    
    def ocr_page_to_json(self, png_path: str, page_number: int) -> Dict[str, Any]:
        """Run Tesseract OCR and return a compact JSON payload.

        NOTE: This delegates to app.services.ocr_utils so the *same* OCR logic/schema
        is used for both extraction and UI highlighting.
        """
        from .ocr_utils import ocr_page_to_json as _ocr_page_to_json
        return _ocr_page_to_json(png_path, page_number)

    def _openai_extract_from_ocr_json(self, ocr_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use the LLM (text-only) to transform OCR JSON into the required extracted schema.

        Required output schema:
        {
          "page_title": "<string>",
          "years": ["2024","2023"],
          "rows": {
            "<row label>": {
              "Experian Value": "<english translation>",
              "2024": "<string>",
              "nota": "<string>",
              "confidence_score": <0..1>
            }
          }
        }
        """
        prompt = (
            "You are a financial statement table extraction engine.\n"
            "You are given OCR output from a single PDF page as JSON (from Tesseract).\n\n"
            "TASK:\n"
            "1) Identify the best page title/heading (page_title).\n"
            "2) Extract a table-like set of rows with ONLY FOUR columns considered:\n"
            "   - Nota (note reference)\n"
            "   - Line Item (row label)\n"
            "   - 2024 (value)\n"
            "   - 2023 (value)\n"
            "Ignore any other columns/years even if present.\n\n"
            "OUTPUT JSON MUST EXACTLY MATCH THIS SHAPE (no extra keys):\n"
            '{\n'
            '  "page_title": "<string>",\n'
            '  "years": ["2024","2023"],\n'
            '  "rows": {\n'
            '    "<row label>": {\n'
            '      "Experian Value": "<english translation of row label>",\n'
            '      "2024": "<string>",\n'
            '      "2023": "<string>",\n'
            '      "nota": "<string>",\n'
            '      "confidence_score": <number between 0 and 1>\n'
            "    }\n"
            "  }\n"
            "}\n\n"

            "Nota Column RULES:\n"
                "-Pay special attention to the Nota column if its present. Fetch the correct Nota values against each row item. \n"
                "-Look for Nota column values for each row. If no nota is found for that row keep the json value as blank\n"
                "-Do not provide arbitary nota values. \n"
                "-Never use numeric 0 for nota\n"

            "Important General Rules:\n"
            "- Keep numbers EXACTLY as printed (keep thousand separators, decimal commas/dots).\n"
            "- If a field is missing, use an empty string for that field (not null).\n"
            "- confidence_score: estimate 0..1 using OCR confidence and extraction certainty.\n"
            "- Do not add wrapper keys. Do not include explanations. Return JSON only.\n"

            "COMPLETENESS CHECK (MANDATORY):\n"
                "Before producing the final JSON:\n"
                    "-Ensure that all the rows having fiancial numbers under 2024 column has been fetched.\n"
                    "-Check if Nota values are populated properly for each row items.\n"
                    "-Cross check if the line items were translated to english and populated under 'Experian Value' field.\n"
                    "-Ensure the JSON contains AT LEAST the same number of row entries.\n"
                    "-If not, retry internally and add the missing rows.\n"
        )

        # Keep OCR payload compact to avoid token blowups (words can be huge).
        compact = {
            "page_number": ocr_payload.get("page_number"),
            "image": ocr_payload.get("image"),
            "text": ocr_payload.get("text", ""),
            "lines": ocr_payload.get("lines", [])[:400],  # cap
        }

        resp = self.client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL_TEXT", os.environ.get("OPENAI_MODEL", "gpt-4o-mini")).strip(),
            messages=[
                {"role": "user", "content": prompt + "\n\nOCR_JSON:\n" + json.dumps(compact, ensure_ascii=False)}
            ],
            response_format={"type": "json_object"},
        )

        out = json.loads(resp.choices[0].message.content or "{}")

        # Post-normalize to enforce required keys and formats
        if "years" not in out or not isinstance(out.get("years"), list):
            out["years"] = ["2024", "2023"]
        if out.get("years") != ["2024", "2023"]:
            out["years"] = ["2024", "2023"]

        out.setdefault("page_title", "")
        out.setdefault("rows", {})
        if not isinstance(out.get("rows"), dict):
            out["rows"] = {}

        # Ensure each row has the required keys
        for k, v in list(out["rows"].items()):
            if not isinstance(v, dict):
                out["rows"][k] = {"Experian Value": "", "2024": "", "nota": "", "confidence_score": 0.0}
                continue
            v.setdefault("Experian Value", "")
            v.setdefault("2024", "")
            v.setdefault("nota", "")
            cs = v.get("confidence_score", 0.0)
            try:
                csf = float(cs)
            except Exception:
                csf = 0.0
            if csf < 0: csf = 0.0
            if csf > 1: csf = 1.0
            v["confidence_score"] = csf
            out["rows"][k] = v

        return out

    def process_page_header(self, png_path):
        
        try:
            prompt = (
                f"""
                You are an expert at extracting information from PNG files. 
                TASK:
                    Analyze the provided png file and return the file header in json format.

                JSON SHAPE (MUST BE PRESERVED):
                    "header": extracted header    

                RULES:
                    - Do NOT add any wrapper keys
                    - Return JSON only
                """
            )

            content=[
                {"type":"text","text":(prompt)},
                {"type":"image_url","image_url":{"url": self._png_to_data_url(str(png_path))}}
            ]
                    
            resp=self.client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role":"user","content":content}],
                response_format={"type":"json_object"},
            )

            json_response = json.loads(resp.choices[0].message.content or "{}")

            return json_response

        except Exception as ex:
            print("exception occurred while extracting records from page", ex)
    
    def process_notur_page(self, job_id, page_number, png_path):
        
        db_path = "data/app.db"
        try:
            prompt = (
                f"""
                    You are a financial statement extraction engine. 

                    TASK:
                        Extract ALL meaningful content from this single financial NOTES page.
                        Return ONLY valid JSON (no extra text).

                        GOAL:
                        Represent the page "as-is" in a structured JSON so the consumer can
                        render or post-process it later.

                        OUTPUT JSON SHAPE:

                        'page_title': "<best effort title/heading for the page>",
                        "sections": 
                                "title": "<sub-heading if any>",
                                    "paragraphs": ["..."],
                                    "tables": 
                                        "title": "<table title if any>",
                                        "columns": ["<col1>", "<col2>", "..."],
                                        "rows":
                                            "name": "<row label>", "values": "<col>": <number|"">, "<col>": <number|"">
                    
                                
                    RULES:
                    - Do NOT wrap the output under a top-level "Notur" key.
                    - Capture headings/subheadings if present.
                    - Capture tables if present (keep columns as they appear, e.g. "2023", "2024", "31-12-24").
                    - Keep row labels exactly as in the page.
                    - Numbers must be JSON numbers when possible; otherwise use "".
                    - Preserve minus signs.

                    NUMBER NORMALIZATION (MANDATORY):

                    The source text uses Danish-style formatting:
                    - Thousands separator is "." (dot)
                    - Decimal separator is "," (comma)

                    Convert all such values into standard JSON numbers using:
                    - Remove all "." used as thousands separators
                    - Replace the final "," with "." as the decimal separator
                    - Preserve the sign (leading "-" or parentheses)

                    Examples:
                    - "120.000.000"          -> 120000000
                    - "120.980,89"           -> 120980.89
                    - "-126.378"             -> -126378          (dot here is thousands separator, no decimals)
                    - "-1.234.567,00"        -> -1234567
                    - "(1.234,50)"           -> -1234.5
                    - "0"                    -> 0

                    Disambiguation rule:
                    - If a number contains BOTH "." and "," => "." is thousands, "," is decimals.
                    - If a number contains ONLY ".":
                    - Treat "." as thousands separator (NOT decimals) unless there are exactly 2 digits after the dot AND the context explicitly indicates decimals.
                    - In financial statements, default "." to thousands separator.
                    - If a number contains ONLY ",":
                    - Treat "," as decimals.

                    Output rules:
                    - Output numeric values as JSON numbers (not strings).
                    - If the value cannot be confidently parsed as a number, output "" (empty string).

                    Now extract the data from the provided png file.

                    """
            )

            content=[
                {"type":"text","text":(prompt)},
                {"type":"image_url","image_url":{"url": self._png_to_data_url(str(png_path))}}
            ]
                    
            resp=self.client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role":"user","content":content}],
                response_format={"type":"json_object"},
            )

            json_response = json.loads(resp.choices[0].message.content or "{}")
                
            print("Job id & page number", job_id, page_number)
            print("Extracted response:", json_response)

            sql = """
                UPDATE job_pages
                SET extracted_json = ?
                WHERE job_id = ? AND page_number = ?
                """

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    sql,
                    (
                        json.dumps(json_response, ensure_ascii=False),
                        int(job_id),
                        int(page_number),
                    )
                )
                conn.commit()

        except Exception as ex:
            print("exception occurred while extracting records from page", ex)

    def process_other_pages(self, job_id, page_number, png_path):
        
        db_path = "data/app.db"
        try:
            prompt = ("""
                    
                    CRITICAL INSTRUCTION:
                    This is a transcription task, NOT a summarization task.
                    You MUST extract EVERY visible table row exactly as it appears.
                    Do NOT omit rows just because they are subtotals, explanations, or indented.

                    The table contains hierarchical sections. Follow these rules strictly:

                    Extract ALL rows including:

                    Section headers

                    Sub-section headers

                    Line items

                    Subtotals (rows containing "í alt")

                    Totals

                    Rows with empty values

                    NEVER collapse multiple rows into one.

                    NEVER return only totals.

                    ROW LABEL RULES:

                    NEVER add artificial tags like [SECTION], [SUBSECTION], [TOTAL] or any other bracketed keywords.

                    Use the label text exactly as printed in the image.

                    Section headings/subheadings should still be included as normal rows, but with empty values for all years.

                    For totals/subtotals (rows like "í alt"), keep the label exactly as printed (e.g., "Ogn í alt") with their numeric values.

                    NEW: TRANSLATION (EXPERIAN VALUE) RULES:

                    For EVERY row label you extract, create an English translation and store it in the field "Experian Value".

                    Keep the original row label as the JSON key (do NOT replace it).

                    "Experian Value" must be a short, business-friendly English phrase.

                    Do NOT add extra explanatory text.

                    Preserve qualifiers like "í alt" in meaning (translate as "total").

                    Preserve punctuation like ":" in the original key, but "Experian Value" should NOT include trailing ":" unless it is meaningful.

                    If the row label is already English, copy it as-is.

                    If you are unsure, provide the best likely English translation and reduce confidence_score accordingly.

                    NUMBER RULES:

                    Keep number formatting EXACTLY as printed (e.g. 1.389.189, (122.158))

                    Do NOT remove thousand separators

                    Do NOT convert negatives

                    If a value is missing, use an empty string ""

                    NOTA RULES:
                        -Pay special attention to the Nota column if its present. Fetch the correct Nota values against each row item.
                        -If no nota is present, use ""
                        -Never use numeric 0 for nota

                    OUTPUT FORMAT (JSON ONLY):
                    {
                        "page_title": "<string>",
                        "years": ["2024", "2023"],
                        "rows": {
                            "<row label>": {
                                "Experian Value": "<english translation of row label>",
                                "2024": "<string>",
                                "2023": "<string>",
                                "nota": "<string>",
                                "confidence_score": <number between 0 and 1>
                            }
                        }
                    }

                    COMPLETENESS CHECK (MANDATORY):
                        Before producing the final JSON:
                            -Count the number of visible table rows in the image.
                            -Check if Nota values are populated properly for each row items
                            -Ensure the JSON contains AT LEAST the same number of row entries.
                            -If not, retry internally and add the missing rows.

                    Return ONLY valid JSON.
                    """
            )

            content=[
                {"type":"text","text":(prompt)},
                {"type":"image_url","image_url":{"url": self._png_to_data_url(str(png_path))}}
            ]
                    
            resp=self.client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role":"user","content":content}],
                response_format={"type":"json_object"},
            )

            json_response = json.loads(resp.choices[0].message.content or "{}")
                
            print("Job id & page number", job_id, page_number)
            print("Extracted response:", json_response)

            return json_response            

        except Exception as ex:
            print("exception occurred while extracting records from page", ex)
            return "complete"

    def _openai_extract_page_json(self, png_path: str, page_number: int) -> Dict[str, Any]:
        """
        Optional: image-based per-page extraction.
        Keeps output schema similar to mock but you can adapt it to your real schema.
        """
        data_url = self._png_to_data_url(png_path)

        prompt = (
            "You are given a financial/report page as an image.\n"
            "Extract key structured information.\n\n"
            "Return ONLY valid JSON with these keys:\n"
            "{\n"
            '  "page_number": number,\n'
            '  "header_preview": string|null,\n'
            '  "rows": [ { "label": string, "value_col_1": string|null, "value_col_2": string|null } ]\n'
            "}\n\n"
            "Rules:\n"
            "- Preserve decimal points exactly.\n"
            "- If a value is missing, return null for that value.\n"
            "- Keep rows concise (max ~50 rows).\n"
            "- No extra keys, no explanations."
        )

        resp = self.client.responses.create(
            model=self.openai_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            text={"format": {"type": "json_object"}},
        )

        raw = getattr(resp, "output_text", None) or ""
        try:
            data = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}

        # Ensure required keys exist
        data.setdefault("page_number", page_number)
        data.setdefault("header_preview", None)
        data.setdefault("rows", [])

        # Add png_path for traceability (matches your existing mock output)
        data["png_path"] = png_path
        return data

    def compare_json(self, ocr_json: Dict[str, Any], llm_json: Dict[str, Any],) -> Dict[str, Any]:
        """
        Merge two extracted financial JSONs.

        Rules:
        1) For any field present in both sources, pick the value from the source whose
        row confidence_score is LOWER.
        2) If a row is missing in llm_json but present in ocr_json, keep the OCR row.
        If missing in ocr_json but present in llm_json, keep the LLM row.
        3) Years are unioned (OCR years + LLM years) in original order preference: OCR first then LLM.

        Assumptions:
        - Both inputs follow structure:
        {
            "page_title": str,
            "years": [str, ...],
            "rows": {
            "<row_key>": {
                "Experian Value": str,
                "<year>": str,
                "nota": str,
                "confidence_score": float|int
            },
            ...
            }
        }
        - confidence_score exists at row level (if missing, treated as +inf).
        """

        def _get_conf(row: Optional[Dict[str, Any]]) -> float:
            if not isinstance(row, dict):
                return float("inf")
            v = row.get("confidence_score")
            try:
                return float(v)
            except Exception:
                return float("inf")

        def _union_years(yrs1: Any, yrs2: Any) -> List[str]:
            out: List[str] = []
            seen: Set[str] = set()
            for src in (yrs1, yrs2):
                if isinstance(src, list):
                    for y in src:
                        ys = str(y)
                        if ys not in seen:
                            seen.add(ys)
                            out.append(ys)
            return out

        def _keys_union(d1: Any, d2: Any) -> Set[str]:
            s: Set[str] = set()
            if isinstance(d1, dict):
                s |= set(d1.keys())
            if isinstance(d2, dict):
                s |= set(d2.keys())
            return s

        # Base output: start with OCR (so "missing in LLM but present in OCR" is naturally preserved)
        merged: Dict[str, Any] = deepcopy(ocr_json) if isinstance(ocr_json, dict) else {}
        if not isinstance(merged, dict):
            merged = {}

        # Page title: prefer OCR, else LLM
        if not merged.get("page_title") and isinstance(llm_json, dict):
            merged["page_title"] = llm_json.get("page_title", "")

        # Years: union
        merged["years"] = _union_years(
            (ocr_json or {}).get("years"),
            (llm_json or {}).get("years"),
        )

        ocr_rows = (ocr_json or {}).get("rows") if isinstance(ocr_json, dict) else {}
        llm_rows = (llm_json or {}).get("rows") if isinstance(llm_json, dict) else {}
        if not isinstance(ocr_rows, dict):
            ocr_rows = {}
        if not isinstance(llm_rows, dict):
            llm_rows = {}

        merged_rows: Dict[str, Any] = {}
        merged["rows"] = merged_rows

        # Union of row keys
        for row_key in sorted(_keys_union(ocr_rows, llm_rows)):
            o_row = ocr_rows.get(row_key)
            l_row = llm_rows.get(row_key)

            # If only one exists, keep it
            if isinstance(o_row, dict) and not isinstance(l_row, dict):
                merged_rows[row_key] = deepcopy(o_row)
                continue
            if isinstance(l_row, dict) and not isinstance(o_row, dict):
                merged_rows[row_key] = deepcopy(l_row)
                continue
            if not isinstance(o_row, dict) and not isinstance(l_row, dict):
                continue  # nothing usable

            # Both exist: decide per-field using LOWER confidence_score at row level
            o_conf = _get_conf(o_row)
            l_conf = _get_conf(l_row)

            # Union of fields across both rows (Experian Value, years, nota, etc.)
            fields = _keys_union(o_row, l_row)

            chosen: Dict[str, Any] = {}
            for f in fields:
                o_has = f in o_row
                l_has = f in l_row

                if o_has and not l_has:
                    chosen[f] = deepcopy(o_row[f])
                elif l_has and not o_has:
                    chosen[f] = deepcopy(l_row[f])
                else:
                    # Both have the field: take the value from the row with LOWER conf
                    if o_conf <= l_conf:
                        chosen[f] = deepcopy(o_row[f])
                    else:
                        chosen[f] = deepcopy(l_row[f])

            # Ensure confidence_score reflects what we chose (the lower one)
            print("Confidance score ", o_conf, l_conf)
            chosen["confidence_score"] = round(min(o_conf, l_conf), 2)
            print("Final json", json.dumps(chosen))
            merged_rows[row_key] = chosen

        return merged


    def verify_ocr_against_llm(self, ocr_json: Dict[str, Any], llm_json: Dict[str, Any],) -> Dict[str, Any]:
        """
        Verify OCR JSON against LLM JSON.

        Rules:
        - Final JSON structure is based ONLY on ocr_json.
        - For each row in ocr_json:
            * If same row key exists in llm_json -> keep confidence_score unchanged
            * If row key NOT found in llm_json -> reduce confidence_score by `penalty`
        - confidence_score is floored at 0.0 and rounded to 2 decimals
        """

        penalty = 0.2
        result = deepcopy(ocr_json)

        ocr_rows = result.get("rows", {})
        llm_rows = llm_json.get("rows", {}) if isinstance(llm_json, dict) else {}

        if not isinstance(ocr_rows, dict):
            return result  # nothing to verify

        for row_key, row_val in ocr_rows.items():
            if not isinstance(row_val, dict):
                continue

            # Current OCR confidence
            try:
                conf = float(row_val.get("confidence_score", 0.0))
            except Exception:
                conf = 0.0

            # If row NOT found in LLM, apply penalty
            if not isinstance(llm_rows, dict) or row_key not in llm_rows:
                conf = conf - penalty

            # Normalize confidence
            conf = max(conf, 0.0)
            conf = round(conf, 2)

            row_val["confidence_score"] = conf
            ocr_rows[row_key] = row_val

        result["rows"] = ocr_rows
        return result


    # --------------------
    # Mock implementations
    # --------------------
    def _mock_company_year(self, text: str, company_guess: Optional[str], year_guess: Optional[str]) -> Dict[str, Any]:
        company = company_guess
        # Try to find a better company line if possible
        for ln in [x.strip() for x in text.splitlines() if x.strip()]:
            if len(ln) >= 3 and not re.search(r"\b\d{4}\b", ln):
                company = company or ln
                break

        year = year_guess
        m = re.search(r"\b(19\d{2}|20\d{2}|2100)\b", text)
        if m:
            year = m.group(0)

        return {
            "company_name": company,
            "publication_year": year
        }

    def _mock_extract_page(self, png_path: str, page_number: int, page_text: str) -> Dict[str, Any]:
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]

        # Pull a likely title/header (first 3 lines)
        header = " | ".join(lines[:3]) if lines else ""

        # Extract table-ish rows that end with two numbers (e.g., 2024 and 2023 columns)
        rows: List[Dict[str, Any]] = []
        row_re = re.compile(r"^(?P<label>.+?)\s+(?P<v1>-?\d[\d\.,]*)\s+(?P<v2>-?\d[\d\.,]*)\s*$")
        for ln in lines:
            m = row_re.match(ln)
            if m:
                rows.append({
                    "label": m.group("label").strip(),
                    "value_col_1": m.group("v1"),
                    "value_col_2": m.group("v2")
                })

        # Fallback: extract all amounts that look like numbers
        nums = re.findall(r"-?\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d+)?", page_text)

        return {
            "page_number": page_number,
            "png_path": png_path,
            "header_preview": header,
            "rows": rows[:50],
            "numbers_found": nums[:200],
            "text_excerpt": "\n".join(lines[:60])
        }
