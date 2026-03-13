# DataLens — AI Data Explorer (SQLite Version)

## Tech Stack
- Frontend  : HTML + Tailwind CSS + Chart.js
- Backend   : Python + FastAPI
- Database  : SQLite (zero setup — built into Python)
- AI        : Groq API (llama-3.3-70b-versatile)

---

## Setup (3 Steps Only)

### Step 1 — Add your Groq API key
Edit the `.env` file:
```
GROQ_API_KEY=gsk_your_key_here
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
pip install httpx==0.27.0
```

### Step 3 — Start the backend
```bash
uvicorn backend.main:app --reload --port 8000
```

### Step 4 — Open the frontend
Open a second terminal in the same folder:
```bash
python -m http.server 5500 --directory frontend
```
Then go to: http://localhost:5500

---

## Project Structure
```
datalens_sqlite/
├── backend/
│   ├── __init__.py
│   ├── main.py        ← FastAPI app + all endpoints
│   ├── database.py    ← SQLite setup + all tables
│   └── cleaner.py     ← Data cleaning + anomaly detection
├── frontend/
│   └── index.html     ← Complete UI
├── uploads/           ← Original CSVs saved here
├── cleaned/           ← Cleaned CSVs saved here
├── exports/           ← Exported chat history
├── datalens.db        ← SQLite database (auto-created)
├── sample_data.csv
├── requirements.txt
└── .env
```

---

## What Gets Stored Permanently

| Data | Where |
|---|---|
| Original CSV file | uploads/ folder |
| Cleaned CSV file | cleaned/ folder |
| File upload record | datalens.db → uploaded_files table |
| Every Q&A + chart | datalens.db → chat_history table |
| Anomaly reports | datalens.db → anomaly_reports table |
| Last active session | datalens.db → session table |

Everything survives server restarts automatically.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Server + DB status |
| POST | /upload | Upload + clean CSV |
| GET | /files | All uploaded files |
| POST | /files/load/{id} | Reload old file by ID |
| GET | /files/download/original | Download original CSV |
| GET | /files/download/cleaned | Download cleaned CSV |
| GET | /summary | Dataset statistics |
| GET | /anomalies | Anomaly detection results |
| POST | /query | Ask AI a question |
| GET | /history | Chat history |
| GET | /history/export | Export history as JSON |
| DELETE | /history | Clear all history |
| GET | /session | Current session info |
