import { useRef, useState } from "react";
import { evaluate } from "./api.js";

const SAMPLE_GT =
  "The quick brown fox jumps over the lazy dog.\nHandwritten notes are hard to read.";
const SAMPLE_OCR =
  "The qulck brown fox jumps over the lazy dog.\nHandwriten notes are hard to reads.";

function EditorPanel({ title, value, onChange, onFile, accent }) {
  const inputRef = useRef(null);

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => onChange(String(reader.result ?? ""));
    reader.readAsText(file);
    onFile?.(file.name);
    e.target.value = ""; // allow re-uploading the same file
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="dot" style={{ background: accent }} />
        <h2>{title}</h2>
        <button className="upload-btn" onClick={() => inputRef.current?.click()}>
          Upload .txt
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".txt,text/plain"
          onChange={handleFile}
          hidden
        />
      </div>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={`Paste or upload ${title.toLowerCase()}…`}
        spellCheck={false}
      />
      <div className="counts">
        {value.trim() ? value.trim().split(/\s+/).length : 0} words ·{" "}
        {value.length} chars
      </div>
    </div>
  );
}

function HighlightView({ segments }) {
  return (
    <pre className="highlight">
      {segments.map((seg, i) => (
        <span key={i} className={seg.status === "error" ? "seg-error" : "seg-match"}>
          {seg.text}
        </span>
      ))}
    </pre>
  );
}

function Metric({ label, value, hint, good }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${good ? "good" : ""}`}>{value}</div>
      {hint && <div className="metric-hint">{hint}</div>}
    </div>
  );
}

const pct = (x) => `${(x * 100).toFixed(2)}%`;

export default function App() {
  const [gt, setGt] = useState("");
  const [ocr, setOcr] = useState("");
  const [result, setResult] = useState(null);
  const [mode, setMode] = useState("word"); // "word" | "char"
  const [ignoreCase, setIgnoreCase] = useState(false);
  const [ignorePunct, setIgnorePunct] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const run = async (opts = { ignoreCase, ignorePunct }) => {
    setError("");
    setLoading(true);
    try {
      const data = await evaluate(gt, ocr, opts);
      setResult(data);
    } catch (e) {
      setError(e.message || "Failed to evaluate. Is the backend running?");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  // Re-run automatically when a normalization toggle changes (if we have a result).
  const toggleCase = () => {
    const next = !ignoreCase;
    setIgnoreCase(next);
    if (result) run({ ignoreCase: next, ignorePunct });
  };
  const togglePunct = () => {
    const next = !ignorePunct;
    setIgnorePunct(next);
    if (result) run({ ignoreCase, ignorePunct: next });
  };

  const loadSample = () => {
    setGt(SAMPLE_GT);
    setOcr(SAMPLE_OCR);
    setResult(null);
  };

  const clearAll = () => {
    setGt("");
    setOcr("");
    setResult(null);
    setError("");
  };

  const alignment = result
    ? mode === "word"
      ? result.word_alignment
      : result.char_alignment
    : null;
  const counts = result
    ? mode === "word"
      ? result.word_counts
      : result.char_counts
    : null;

  return (
    <div className="app">
      <header>
        <h1>OCR Evaluation — CER &amp; WER</h1>
        <p>
          Compare ground-truth text against OCR output. Matches are{" "}
          <span className="g">green</span>, errors are{" "}
          <span className="r">red</span>.
        </p>
      </header>

      <section className="editors">
        <EditorPanel
          title="Ground Truth"
          value={gt}
          onChange={setGt}
          accent="#16a34a"
        />
        <EditorPanel
          title="OCR Result"
          value={ocr}
          onChange={setOcr}
          accent="#2563eb"
        />
      </section>

      <div className="actions">
        <button className="primary" onClick={() => run()} disabled={loading}>
          {loading ? "Evaluating…" : "Evaluate"}
        </button>
        <button onClick={loadSample}>Load sample</button>
        <button onClick={clearAll}>Clear</button>

        <div className="norm-opts">
          <label>
            <input type="checkbox" checked={ignoreCase} onChange={toggleCase} />
            Ignore case
          </label>
          <label>
            <input
              type="checkbox"
              checked={ignorePunct}
              onChange={togglePunct}
            />
            Ignore punctuation
          </label>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <section className="metrics">
            <Metric
              label="CER"
              value={pct(result.cer)}
              hint="Character Error Rate"
            />
            <Metric
              label="WER"
              value={pct(result.wer)}
              hint="Word Error Rate"
            />
            <Metric
              label="Char Accuracy"
              value={pct(result.char_accuracy)}
              good
              hint="1 − CER"
            />
            <Metric
              label="Word Accuracy"
              value={pct(result.word_accuracy)}
              good
              hint="1 − WER"
            />
          </section>

          <section className="breakdown">
            <strong>{mode === "word" ? "Word" : "Character"} breakdown:</strong>{" "}
            {counts.hits} correct · {counts.substitutions} substitutions ·{" "}
            {counts.deletions} deletions · {counts.insertions} insertions ·{" "}
            reference length {counts.ref_length}
          </section>

          <div className="mode-toggle">
            <span>Highlight by:</span>
            <button
              className={mode === "word" ? "active" : ""}
              onClick={() => setMode("word")}
            >
              Word
            </button>
            <button
              className={mode === "char" ? "active" : ""}
              onClick={() => setMode("char")}
            >
              Character
            </button>
          </div>

          <section className="compare">
            <div className="compare-col">
              <h3>Ground Truth</h3>
              <HighlightView segments={alignment.left} />
            </div>
            <div className="compare-col">
              <h3>OCR Result</h3>
              <HighlightView segments={alignment.right} />
            </div>
          </section>
        </>
      )}

      <footer>
        Frontend: React + Vite · Backend: FastAPI · CER/WER via Levenshtein
        alignment
      </footer>
    </div>
  );
}
