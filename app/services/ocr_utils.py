import os
import json
from typing import Dict, Any, List

from PIL import Image
import pytesseract
from pytesseract import Output

def ocr_page_to_json(png_path: str, page_number: int) -> Dict[str, Any]:
    """Run Tesseract OCR and return a compact JSON payload used across the app.

    This is the single source of truth for OCR output so both:
    - data extraction (LLM normalization) and
    - UI highlighting (word boxes)
    use the same OCR configuration and schema.

    Output schema:
    {
      "page_number": <int>,
      "image": {"width": <int>, "height": <int>},
      "text": "<full text>",
      "lines": [{"text":..., "bbox":[x1,y1,x2,y2], "avg_conf":..., "block":..,"par":..,"line":..}],
      "words": [{"text":..., "conf":..., "bbox":[x1,y1,x2,y2], "block":..,"par":..,"line":..,"word":..}]
    }
    """
    img = Image.open(png_path)
    width, height = img.size

    lang = os.environ.get("TESSERACT_LANG", "eng+dan+fao").strip() or "eng"
    config = os.environ.get("TESSERACT_CONFIG", "--oem 1 --psm 6 --dpi 300")

    data = pytesseract.image_to_data(img, output_type=Output.DICT, lang=lang, config=config)

    words: List[Dict[str, Any]] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data.get("text", [""])[i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data.get("conf", [])[i])
        except Exception:
            conf = -1.0

        x = int(data.get("left", [0])[i] or 0)
        y = int(data.get("top", [0])[i] or 0)
        w = int(data.get("width", [0])[i] or 0)
        h = int(data.get("height", [0])[i] or 0)

        words.append({
            "text": txt,
            "conf": conf,
            "bbox": [x, y, x + w, y + h],
            "block": int(data.get("block_num", [0])[i] or 0),
            "par": int(data.get("par_num", [0])[i] or 0),
            "line": int(data.get("line_num", [0])[i] or 0),
            "word": int(data.get("word_num", [0])[i] or 0),
        })

    # Aggregate into lines (block, par, line)
    lines_map: Dict[Any, List[Dict[str, Any]]] = {}
    for w in words:
        k = (w.get("block"), w.get("par"), w.get("line"))
        lines_map.setdefault(k, []).append(w)

    lines: List[Dict[str, Any]] = []
    for k, ws in lines_map.items():
        ws_sorted = sorted(ws, key=lambda t: (t["bbox"][0], t["bbox"][1]))
        line_text = " ".join([t["text"] for t in ws_sorted]).strip()
        if not line_text:
            continue
        x1 = min(t["bbox"][0] for t in ws_sorted)
        y1 = min(t["bbox"][1] for t in ws_sorted)
        x2 = max(t["bbox"][2] for t in ws_sorted)
        y2 = max(t["bbox"][3] for t in ws_sorted)

        confs = [t["conf"] for t in ws_sorted if t.get("conf") is not None and t["conf"] >= 0]
        avg_conf = sum(confs) / len(confs) if confs else -1.0

        lines.append({
            "text": line_text,
            "bbox": [x1, y1, x2, y2],
            "avg_conf": avg_conf,
            "block": k[0],
            "par": k[1],
            "line": k[2],
        })

    full_text = "\n".join([ln["text"] for ln in sorted(lines, key=lambda t: (t["bbox"][1], t["bbox"][0]))])

    return {
        "page_number": page_number,
        "image": {"width": width, "height": height},
        "text": full_text,
        "lines": lines,
        "words": words,
        "lang": lang,
    }

def write_ocr_json(png_path: str, out_json_path: str, page_number: int) -> None:
    payload = ocr_page_to_json(png_path, page_number)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
