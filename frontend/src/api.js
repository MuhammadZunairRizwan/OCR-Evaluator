// Base URL of the FastAPI backend.
// In production (Vercel) set VITE_API_URL to your Render/Railway URL.
// In local dev it falls back to localhost:8000.
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function evaluate(groundTruth, ocrText, options = {}) {
  const res = await fetch(`${API_URL}/api/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ground_truth: groundTruth,
      ocr_text: ocrText,
      ignore_case: !!options.ignoreCase,
      ignore_punct: !!options.ignorePunct,
    }),
  });
  if (!res.ok) {
    throw new Error(`Server error (${res.status})`);
  }
  return res.json();
}
