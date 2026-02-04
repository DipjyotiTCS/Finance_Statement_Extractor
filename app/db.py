import sqlite3
from typing import Optional, Any, Dict, List

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL,
  pdf_filename TEXT NOT NULL,
  pdf_path TEXT NOT NULL,
  temp_folder TEXT,
  start_page INTEGER,
  end_page INTEGER,
  error TEXT
);

CREATE TABLE IF NOT EXISTS job_pages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id INTEGER NOT NULL,
  page_number INTEGER NOT NULL,
  png_path TEXT NOT NULL,
  extracted_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(job_id) REFERENCES jobs(id)
);

-- Word-level OCR coordinates (per rendered PNG page)
CREATE TABLE IF NOT EXISTS job_page_ocr (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id INTEGER NOT NULL,
  page_number INTEGER NOT NULL,
  ocr_json_path TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(job_id, page_number),
  FOREIGN KEY(job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS job_meta (
  job_id INTEGER PRIMARY KEY,
  company_name TEXT,
  publication_year TEXT,
  publication_date TEXT,
  FOREIGN KEY(job_id) REFERENCES jobs(id)
);
"""

def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str) -> None:
    conn = get_conn(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()

        # Lightweight migration: add new columns if missing
        try:
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(job_meta)").fetchall()]
            if "publication_date" not in cols:
                conn.execute("ALTER TABLE job_meta ADD COLUMN publication_date TEXT")
                conn.commit()
        except Exception:
            # If job_meta doesn't exist yet, SCHEMA will create it
            pass

    finally:
        conn.close()

def q(conn: sqlite3.Connection, sql: str, params: Optional[tuple]=None) -> List[sqlite3.Row]:
    cur = conn.execute(sql, params or ())
    return cur.fetchall()

def e(conn: sqlite3.Connection, sql: str, params: Optional[tuple]=None) -> int:
    cur = conn.execute(sql, params or ())
    conn.commit()
    return cur.lastrowid
