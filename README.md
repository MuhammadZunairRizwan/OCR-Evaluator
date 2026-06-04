# OCR CER / WER Comparison Tool

Evaluate OCR output against human ground-truth transcriptions. Computes
**CER** (Character Error Rate) and **WER** (Word Error Rate), and shows a
side-by-side comparison where matches are **green** and errors are **red**.

- **Frontend:** React + Vite (deploy to Vercel)
- **Backend:** FastAPI (deploy to Render / Railway)

## Features

- Two text boxes — **Ground Truth** (left) and **OCR Result** (right)
- Upload `.txt` files **or** edit/paste text manually
- CER, WER, and accuracy (1 − error rate) scores
- Word-level and character-level error breakdown (substitutions / deletions / insertions / hits)
- Toggle highlighting between **word** and **character** alignment
- **Ignore case** / **Ignore punctuation** toggles to match how "correct" is graded (re-scores live)
- Side-by-side colour-coded diff for verification

## How the metrics are computed

Both rates use Levenshtein (edit-distance) alignment of the OCR hypothesis
against the ground-truth reference:

```
WER = (S + D + I) / N   over words
CER = (S + D + I) / N   over characters
```

where `S` = substitutions, `D` = deletions, `I` = insertions, and `N` = number
of tokens in the ground truth. Accuracy is reported as `1 − error rate`.
The same alignment produces the green/red highlighting, so the colours always
match the score.

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
