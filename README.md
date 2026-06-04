# NARA · CER / WER OCR Evaluator

Evaluate OCR output (NARA / National Archives records) against human
ground-truth transcriptions. You provide two CSVs and look records up by
**NAID** (National Archives ID); the app computes **CER** (Character Error Rate)
and **WER** (Word Error Rate) and shows a side-by-side comparison where matches
are **green** and errors are **red**.

- **Frontend:** React + Vite (Vercel or Render Static Site)
- **Backend:** FastAPI + [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) (Render / Railway)

## Workflow

1. Upload the two CSVs (parsed entirely in your browser — they never leave your machine):
   - `naid_transcriptions.csv` — columns `naId, transcriptionText` (ground truth)
   - `ocr_extraction.csv` — columns `naId, extracted_text` (OCR output)
2. Type a **NAID** (or hit **Random**) and click **View record**.
3. The app finds that NAID in both CSVs, shows both texts, and computes CER/WER.

The app ships with a small **demo dataset** so it works before any upload;
loading both CSVs unlocks all records.

## Features

- **NAID lookup** across two CSVs, with a **Random** record button
- CER, WER, and accuracy (1 − error rate) scores
- Word-level and character-level error breakdown (substitutions / deletions / insertions / hits)
- Toggle highlighting between **word** and **character** alignment
- **Ignore case** / **Ignore punctuation** toggles to match how "correct" is graded (re-scores live)
- Side-by-side colour-coded diff for verification
- **Handles very large documents** (200k+ characters): exact scores are always
  computed; the green/red diff falls back to a truncated preview + plain-text
  view when a document is too large to align in full

## How the metrics are computed

Both rates use Levenshtein (edit-distance) alignment of the OCR hypothesis
against the ground-truth reference:

```
WER = (S + D + I) / N   over words
CER = (S + D + I) / N   over characters
```

where `S` = substitutions, `D` = deletions, `I` = insertions, and `N` = number
of tokens in the ground truth. Accuracy is reported as `1 − error rate`.

The distance is computed with **rapidfuzz** (C-backed) so even 235k-character
records score in a couple of seconds. The same alignment produces the green/red
highlighting; for documents above `CHAR_ALIGN_MAX` (20k chars) / `WORD_ALIGN_MAX`
(8k words) the exact split is omitted and a truncated preview is shown instead
(scores still cover the full document).

## Big data note

The real CSVs (~105 MB combined, 2,260 records) are **not** committed to the
repo — they're git-ignored under `test-data/`. CSV parsing and NAID matching
happen client-side; only the selected record's two texts are sent to the backend.

---

## Run locally

### Backend

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend runs at http://localhost:8000 (docs at `/docs`).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:5173 and talks to the backend on
`localhost:8000` by default.

---

## Deploy

### Backend → Render

1. Push this repo to GitHub.
2. On Render: **New → Web Service**, point at the repo.
   - Root directory: `backend`
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - (A `render.yaml` is included for one-click Blueprint deploys.)
3. After the frontend is live, set the `ALLOWED_ORIGINS` env var to your Vercel
   URL (e.g. `https://your-app.vercel.app`).

### Frontend → Vercel

1. On Vercel: **New Project**, import the repo.
   - Root directory: `frontend`
   - Framework preset: **Vite** (build `npm run build`, output `dist`)
2. Add an env var `VITE_API_URL` = your Render backend URL.
3. Deploy.

---

## Project structure

```
ocrcomparison/
├── backend/
│   ├── main.py            FastAPI app + CORS
│   ├── metrics.py         CER/WER + alignment (Levenshtein)
│   ├── requirements.txt
│   ├── Procfile           (Railway/Heroku-style)
│   └── render.yaml        (Render Blueprint)
└── frontend/
    ├── src/
    │   ├── App.jsx        UI + state
    │   ├── api.js         backend client
    │   └── index.css      simple skin
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── vercel.json
```
