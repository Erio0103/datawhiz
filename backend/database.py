"""
database.py — SQLite database setup
All tables are created automatically on first run.
No installation needed — SQLite is built into Python.

Tables:
  - uploaded_files   → every CSV uploaded (original + cleaned path)
  - chat_history     → every question + answer + chart
  - anomaly_reports  → outliers detected per file
  - session          → last active file (auto-restore on restart)
"""

import sqlite3
import os

DB_FILE = "datalens.db"

# ══════════════════════════════════════════════════════════
# CREATE ALL TABLES
# ══════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Every CSV file ever uploaded
    c.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            filename       TEXT NOT NULL,
            original_path  TEXT,
            cleaned_path   TEXT,
            original_rows  INTEGER,
            cleaned_rows   INTEGER,
            columns        TEXT,
            cleaning_log   TEXT,
            summary        TEXT,
            uploaded_at    TEXT DEFAULT (datetime('now'))
        )
    """)

    # Every question + answer
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id    INTEGER,
            filename   TEXT,
            question   TEXT NOT NULL,
            answer     TEXT NOT NULL,
            chart      TEXT,
            chart_type TEXT,
            asked_at   TEXT DEFAULT (datetime('now'))
        )
    """)

    # Anomaly detection results
    c.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_reports (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id        INTEGER,
            column_name    TEXT,
            outlier_count  INTEGER,
            normal_min     TEXT,
            normal_max     TEXT,
            outlier_values TEXT,
            detected_at    TEXT DEFAULT (datetime('now'))
        )
    """)

    # Session — remembers last active file
    c.execute("""
        CREATE TABLE IF NOT EXISTS session (
            id           INTEGER PRIMARY KEY CHECK (id = 1),
            file_id      INTEGER,
            filename     TEXT,
            cleaned_path TEXT,
            columns      TEXT,
            saved_at     TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()
    print("✅ SQLite database ready — datalens.db")


# ══════════════════════════════════════════════════════════
# CONNECTION HELPER
# ══════════════════════════════════════════════════════════
def get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    return conn
