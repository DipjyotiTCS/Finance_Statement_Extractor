import os
import re
import json
import base64
from typing import Dict, Any, Optional, List
from openai import OpenAI
import sqlite3
import cv2
import pytesseract

from ..db import get_conn, q, e

class LLMClient:

    def __init__(self):
        self.client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").strip())
        self.openai_model = os.environ.get("OPENAI_MODEL_VISION", "gpt-4o-mini").strip()


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
        for ep in extraction_payload:
            try:
                header = self.process_page_header(ep.get("png_path"))
                print("Header:", header)
                header_text = str(header.get("header", "")).lower()

                if "notur" in header_text:
                    self.process_notur_page(job_id, ep.get("page_number"), ep.get("png_path"))
                else:
                    self.process_other_pages(job_id, ep.get("page_number"), ep.get("png_path"))

            except Exception as ex:
                print("exception occurred while extracting records from page", ex)


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

                    If no nota is present, use ""

                    Never use numeric 0 for nota

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

                    Count the number of visible table rows in the image.

                    Ensure the JSON contains AT LEAST the same number of row entries.

                    If not, retry internally and add the missing rows.

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
