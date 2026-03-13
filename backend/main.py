"""
backend/main.py — FastAPI Backend with SQLite
=============================================
Run from the datalens_sqlite/ folder:
  uvicorn backend.main:app --reload --port 8000

Everything is stored permanently in datalens.db:
  - uploaded_files   → every CSV uploaded
  - chat_history     → every Q&A + chart
  - anomaly_reports  → outlier detection
  - session          → auto-restores last file on restart
"""

import os
import io
import json
import datetime
from typing import Optional, List

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

from backend.database import init_db, get_conn
from backend.cleaner  import (
    clean_dataframe, get_summary,
    detect_anomalies, suggest_chart_type
)

load_dotenv()

# ── App ───────────────────────────────────────────────────
app = FastAPI(title="DataLens API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Groq AI client ────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# ── Storage folders ───────────────────────────────────────
UPLOADS_DIR = "uploads"
CLEANED_DIR = "cleaned"
EXPORTS_DIR = "exports"
for folder in [UPLOADS_DIR, CLEANED_DIR, EXPORTS_DIR]:
    os.makedirs(folder, exist_ok=True)

# ── In-memory active session ──────────────────────────────
_store: dict = {
    "df":       None,
    "file_id":  None,
    "filename": None,
    "columns":  [],
}

# ══════════════════════════════════════════════════════════
# STARTUP — init DB + restore last session
# ══════════════════════════════════════════════════════════
@app.on_event("startup")
def startup():
    init_db()
    restore_session()

def restore_session():
    """Reload the last active CSV automatically on server start."""
    try:
        conn = get_conn()
        row = conn.execute("SELECT * FROM session WHERE id=1").fetchone()
        conn.close()
        if not row:
            return
        cleaned_path = row["cleaned_path"]
        if cleaned_path and os.path.exists(cleaned_path):
            df = pd.read_csv(cleaned_path)
            _store["df"]       = df
            _store["file_id"]  = row["file_id"]
            _store["filename"] = row["filename"]
            _store["columns"]  = json.loads(row["columns"])
            print(f"✅ Session restored: '{row['filename']}' ({len(df)} rows)")
        else:
            print("⚠️  Previous CSV file not found on disk.")
    except Exception as e:
        print(f"⚠️  Could not restore session: {e}")

def save_session(file_id, filename, cleaned_path, columns):
    """Save current session to SQLite so it survives restarts."""
    conn = get_conn()
    conn.execute("DELETE FROM session")
    conn.execute(
        "INSERT INTO session (id, file_id, filename, cleaned_path, columns) VALUES (1,?,?,?,?)",
        (file_id, filename, cleaned_path, json.dumps(columns))
    )
    conn.commit()
    conn.close()

# ══════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════
class QueryRequest(BaseModel):
    prompt:  str
    columns: Optional[List[str]] = None

class QueryResponse(BaseModel):
    answer:           str
    chart:            Optional[dict] = None
    chart_suggestion: Optional[str]  = None
    error:            Optional[str]  = None

# ══════════════════════════════════════════════════════════
# HELPER — AI system prompt
# ══════════════════════════════════════════════════════════
def build_system_prompt(df: pd.DataFrame) -> str:
    schema  = "\n".join(f"  - {col}: {dtype}" for col, dtype in df.dtypes.items())
    preview = df.head(5).to_dict(orient="records")
    stats   = json.dumps(get_summary(df), indent=2, default=str)
    return f"""You are an expert data analyst.

COLUMNS:
{schema}

FIRST 5 ROWS:
{json.dumps(preview, indent=2, default=str)}

STATISTICS:
{stats}

RULES:
1. Respond ONLY with valid JSON — no markdown, no extra text.
2. Use EXACTLY this format:
{{
  "answer": "your clear text answer with real numbers",
  "chart": {{
    "type": "bar | pie | line",
    "title": "chart title",
    "labels": ["A", "B", "C"],
    "datasets": [
      {{"label": "Series name", "data": [10, 20, 30]}}
    ]
  }}
}}
3. Set "chart" to null if no chart is needed.
4. All values in data arrays MUST be numbers.
5. For pie charts use ONE dataset only.
6. Base answers ONLY on the data provided.
"""

# ══════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════

# ── Health check ──────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":     "ok",
        "csv_loaded": _store["df"] is not None,
        "filename":   _store["filename"],
        "rows":       len(_store["df"]) if _store["df"] is not None else 0,
        "database":   "SQLite (datalens.db)",
    }

# ── Upload CSV ────────────────────────────────────────────
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV → clean → save original + cleaned to disk → save record to SQLite."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(422, f"Failed to parse CSV: {e}")

    if df.empty:
        raise HTTPException(422, "The CSV file is empty.")

    original_rows = len(df)

    # Save original CSV to disk
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name   = file.filename.replace(" ", "_")
    upload_path = os.path.join(UPLOADS_DIR, f"{timestamp}_{safe_name}")
    with open(upload_path, "wb") as f_out:
        f_out.write(contents)

    # Clean the data
    df, cleaning_log = clean_dataframe(df)

    # Save cleaned CSV to disk
    cleaned_path = os.path.join(CLEANED_DIR, f"{timestamp}_cleaned_{safe_name}")
    df.to_csv(cleaned_path, index=False)

    # Build stats
    summary   = get_summary(df)
    anomalies = detect_anomalies(df)

    # Save file record to SQLite
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO uploaded_files
           (filename, original_path, cleaned_path, original_rows,
            cleaned_rows, columns, cleaning_log, summary)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            file.filename, upload_path, cleaned_path,
            original_rows, len(df),
            json.dumps(list(df.columns)),
            json.dumps(cleaning_log),
            json.dumps(summary, default=str),
        )
    )
    file_id = cur.lastrowid

    # Save anomaly reports to SQLite
    for a in anomalies:
        conn.execute(
            """INSERT INTO anomaly_reports
               (file_id, column_name, outlier_count, normal_min, normal_max, outlier_values)
               VALUES (?,?,?,?,?,?)""",
            (
                file_id, a["column"], a["count"],
                str(a["normal_range"]["min"]),
                str(a["normal_range"]["max"]),
                json.dumps(a["outlier_values"]),
            )
        )
    conn.commit()
    conn.close()

    # Update in-memory session
    _store["df"]       = df
    _store["file_id"]  = file_id
    _store["filename"] = file.filename
    _store["columns"]  = list(df.columns)

    # Save session to SQLite for auto-restore
    save_session(file_id, file.filename, cleaned_path, list(df.columns))

    return {
        "message":         "CSV uploaded, cleaned and saved permanently.",
        "file_id":         file_id,
        "filename":        file.filename,
        "original_path":   upload_path,
        "cleaned_path":    cleaned_path,
        "original_rows":   original_rows,
        "cleaned_rows":    len(df),
        "columns":         list(df.columns),
        "cleaning_log":    cleaning_log,
        "anomalies_found": len(anomalies),
    }

# ── List all uploaded files ───────────────────────────────
@app.get("/files")
def list_files():
    """Get all files ever uploaded from SQLite."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM uploaded_files ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return {
        "total": len(rows),
        "files": [
            {
                "id":            r["id"],
                "filename":      r["filename"],
                "original_rows": r["original_rows"],
                "cleaned_rows":  r["cleaned_rows"],
                "columns":       json.loads(r["columns"]),
                "cleaning_log":  json.loads(r["cleaning_log"]),
                "uploaded_at":   r["uploaded_at"],
            }
            for r in rows
        ],
    }

# ── Load a previous file by ID ────────────────────────────
@app.post("/files/load/{file_id}")
def load_file(file_id: int):
    """Reload any previously uploaded CSV by its ID."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM uploaded_files WHERE id=?", (file_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, f"No file with id {file_id}")
    if not os.path.exists(row["cleaned_path"]):
        raise HTTPException(404, "Cleaned file not found on disk.")

    df = pd.read_csv(row["cleaned_path"])
    _store["df"]       = df
    _store["file_id"]  = row["id"]
    _store["filename"] = row["filename"]
    _store["columns"]  = list(df.columns)

    save_session(row["id"], row["filename"], row["cleaned_path"], list(df.columns))

    return {
        "message":  f"'{row['filename']}' loaded successfully.",
        "file_id":  row["id"],
        "filename": row["filename"],
        "rows":     len(df),
        "columns":  list(df.columns),
    }

# ── Download CSV ──────────────────────────────────────────
@app.get("/files/download/{file_type}")
def download_file(file_type: str):
    """Download current CSV. file_type = 'original' or 'cleaned'"""
    if _store["df"] is None:
        raise HTTPException(400, "No CSV loaded.")
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM uploaded_files WHERE id=?", (_store["file_id"],)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "File record not found.")
    path = row["original_path"] if file_type == "original" else row["cleaned_path"]
    if not os.path.exists(path):
        raise HTTPException(404, "File not found on disk.")
    return FileResponse(path, media_type="text/csv", filename=os.path.basename(path))

# ── Data summary ──────────────────────────────────────────
@app.get("/summary")
def get_data_summary():
    """Get full statistics for the current CSV."""
    if _store["df"] is None:
        raise HTTPException(400, "No CSV loaded. Upload a file first.")
    df = _store["df"]
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM uploaded_files WHERE id=?", (_store["file_id"],)
    ).fetchone()
    conn.close()
    return {
        "file_id":      _store["file_id"],
        "filename":     _store["filename"],
        "rows":         len(df),
        "columns":      len(df.columns),
        "column_names": list(df.columns),
        "summary":      get_summary(df),
        "preview":      df.head(5).to_dict(orient="records"),
        "cleaning_log": json.loads(row["cleaning_log"]) if row else [],
        "uploaded_at":  row["uploaded_at"] if row else None,
    }

# ── Anomaly detection ─────────────────────────────────────
@app.get("/anomalies")
def get_anomalies():
    """Get anomaly detection results from SQLite."""
    if _store["file_id"] is None:
        raise HTTPException(400, "No CSV loaded.")
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM anomaly_reports WHERE file_id=?",
        (_store["file_id"],)
    ).fetchall()
    conn.close()
    return {
        "filename":        _store["filename"],
        "anomalies_found": len(rows),
        "anomalies": [
            {
                "column":         r["column_name"],
                "count":          r["outlier_count"],
                "normal_range":   {"min": r["normal_min"], "max": r["normal_max"]},
                "outlier_values": json.loads(r["outlier_values"]),
                "detected_at":    r["detected_at"],
            }
            for r in rows
        ],
    }

# ── Query — ask the AI ────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query_data(request: QueryRequest):
    """Send question to Groq AI and save answer to SQLite."""
    if _store["df"] is None:
        return QueryResponse(answer="⚠️ No CSV loaded. Please upload a file first.")

    df = _store["df"].copy()

    # Apply column filter if provided
    if request.columns:
        valid = [c for c in request.columns if c in df.columns]
        if valid:
            df = df[valid]

    chart_suggestion = suggest_chart_type(request.prompt)

    # Call Groq AI
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": build_system_prompt(df)},
                {"role": "user",   "content": request.prompt},
            ],
        )
        raw = completion.choices[0].message.content
    except Exception as e:
        return QueryResponse(answer="❌ AI call failed. Check your API key.", error=str(e))

    # Parse AI response
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return QueryResponse(answer="❌ AI returned invalid format.", error=raw[:200])

    answer = parsed.get("answer", "No answer returned.")
    chart  = parsed.get("chart")

    # Validate chart data
    if chart:
        try:
            assert isinstance(chart.get("labels"), list)
            assert isinstance(chart.get("datasets"), list)
            for ds in chart["datasets"]:
                ds["data"] = [float(v) for v in ds["data"]]
        except Exception as e:
            answer += f"\n_(Chart could not be rendered: {e})_"
            chart = None

    # Save to SQLite
    conn = get_conn()
    conn.execute(
        """INSERT INTO chat_history
           (file_id, filename, question, answer, chart, chart_type)
           VALUES (?,?,?,?,?,?)""",
        (
            _store["file_id"],
            _store["filename"],
            request.prompt,
            answer,
            json.dumps(chart) if chart else None,
            chart.get("type") if chart else None,
        )
    )
    conn.commit()
    conn.close()

    return QueryResponse(
        answer=answer,
        chart=chart,
        chart_suggestion=chart_suggestion,
    )

# ── Chat history ──────────────────────────────────────────
@app.get("/history")
def get_history(filename: Optional[str] = None, limit: int = 50):
    """Get chat history from SQLite."""
    conn = get_conn()
    if filename:
        rows = conn.execute(
            "SELECT * FROM chat_history WHERE filename=? ORDER BY id DESC LIMIT ?",
            (filename, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM chat_history ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return {
        "total": len(rows),
        "chats": [
            {
                "id":       r["id"],
                "filename": r["filename"],
                "question": r["question"],
                "answer":   r["answer"],
                "chart":    json.loads(r["chart"]) if r["chart"] else None,
                "asked_at": r["asked_at"],
            }
            for r in rows
        ],
    }

# ── Export chat history ───────────────────────────────────
@app.get("/history/export")
def export_history(filename: Optional[str] = None):
    """Export full chat history as a downloadable JSON file."""
    conn = get_conn()
    if filename:
        rows = conn.execute(
            "SELECT * FROM chat_history WHERE filename=? ORDER BY id DESC",
            (filename,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM chat_history ORDER BY id DESC"
        ).fetchall()
    conn.close()

    if not rows:
        raise HTTPException(404, "No chat history found.")

    export_path = os.path.join(
        EXPORTS_DIR,
        f"chat_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    data = [
        {
            "id":       r["id"],
            "filename": r["filename"],
            "question": r["question"],
            "answer":   r["answer"],
            "chart":    json.loads(r["chart"]) if r["chart"] else None,
            "asked_at": r["asked_at"],
        }
        for r in rows
    ]
    with open(export_path, "w") as f:
        json.dump(data, f, indent=2)

    return FileResponse(
        export_path,
        media_type="application/json",
        filename=os.path.basename(export_path)
    )

# ── Delete all chat history ───────────────────────────────
@app.delete("/history")
def delete_history():
    conn = get_conn()
    conn.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()
    return {"message": "All chat history deleted from SQLite."}

# ── Get single chat by ID ─────────────────────────────────
@app.get("/history/{chat_id}")
def get_chat(chat_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM chat_history WHERE id=?", (chat_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Chat not found.")
    return {
        "id":       row["id"],
        "filename": row["filename"],
        "question": row["question"],
        "answer":   row["answer"],
        "chart":    json.loads(row["chart"]) if row["chart"] else None,
        "asked_at": row["asked_at"],
    }

# ── Session info ──────────────────────────────────────────
@app.get("/session")
def get_session():
    """See what's currently loaded."""
    conn = get_conn()
    row = conn.execute("SELECT * FROM session WHERE id=1").fetchone()
    conn.close()
    if not row:
        return {"message": "No session saved yet."}
    return {
        "file_id":     row["file_id"],
        "filename":    row["filename"],
        "columns":     json.loads(row["columns"]),
        "saved_at":    row["saved_at"],
    }
