# Flask PDF Processor (Jobs + SQLite + PyMuPDF)

## What it does
- Upload a PDF from the homepage and click **Start Processing**.
- Creates a **job** entry in SQLite with status **Work in progress**, then processes the PDF in a background thread.
- Finds the page whose **header line** contains **"Rakstrarroknskapur"**, then splits from that page to the end:
  - Saves each page as a **PNG** using **PyMuPDF**
  - Stores PNGs in a **dedicated temp folder per job**
  - Stores the temp folder path in SQLite
- For each PNG, extracts structured data (default: **mock** extractor) and stores JSON in SQLite.
- Phase 2: extracts **Company Name** and **Year of publication** from the first page and stores it in SQLite.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env  # optional
python run.py
```

Open: http://127.0.0.1:5000

## Notes about LLM
By default, the app uses a **mock extractor** (no API key needed) so the sample PDF works out of the box.

If you want to wire a real LLM later, set:
- `LLM_MODE=openai`
- `OPENAI_API_KEY=...`

and update `app/services/llm_client.py` accordingly.
