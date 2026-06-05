"""
FastAPI backend for the OCR CER/WER comparison tool.

Endpoints
---------
GET  /              -> health check
POST /api/evaluate  -> compute CER/WER + side-by-side highlighting
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from metrics import evaluate

app = FastAPI(title="OCR CER/WER Comparison API", version="1.0.0")

# Allow the Vercel frontend (and local dev) to call the API.
# Set ALLOWED_ORIGINS env var (comma-separated) in production for a tighter policy.
_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
allow_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvaluateRequest(BaseModel):
    ground_truth: str = ""
    ocr_text: str = ""
    ignore_case: bool = False
    ignore_punct: bool = False
    ignore_space: bool = False
    ignore_newline: bool = False


@app.get("/")
def health():
    return {"status": "ok", "service": "ocr-cer-wer"}


@app.post("/api/evaluate")
def api_evaluate(req: EvaluateRequest):
    return evaluate(
        req.ground_truth,
        req.ocr_text,
        ignore_case=req.ignore_case,
        ignore_punct=req.ignore_punct,
        ignore_space=req.ignore_space,
        ignore_newline=req.ignore_newline,
    )
