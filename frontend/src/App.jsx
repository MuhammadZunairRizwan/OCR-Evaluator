import { useMemo, useRef, useState } from "react";
import { evaluate } from "./api.js";
import { parseNaraCsv } from "./csv.js";
import { DEMO_RECORDS } from "./demoData.js";

const pct = (x) => `${(x * 100).toFixed(2)}%`;
const fmt = (n) => n.toLocaleString();

// ---- Demo maps (available before any CSV is uploaded) ----
const DEMO_GT = new Map(DEMO_RECORDS.map((r) => [r.naId, r.gt]));
const DEMO_OCR = new Map(DEMO_RECORDS.map((r) => [r.naId, r.ocr]));

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

function CsvPicker({ label, filename, count, accent, onPick, disabled }) {
  const ref = useRef(null);
  return (
    <div className="csv-picker">
      <div className="csv-label">
        <span className="dot" style={{ background: accent }} /> {label}
      </div>
      <button onClick={() => ref.current?.click()} disabled={disabled}>
        Choose CSV
      </button>
      <input
        ref={ref}
        type="file"
        accept=".csv,text/csv"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onPick(f);
          e.target.value = "";
        }}
      />
      <span className="csv-status">
        {filename ? `${filename} — ${fmt(count)} rows` : "No file chosen"}
      </span>
    </div>
  );
}

export default function App() {
  const [naidInput, setNaidInput] = useState("");
  const [current, setCurrent] = useState(null); // {naId, gt, ocr}
  const [result, setResult] = useState(null);
  const [mode, setMode] = useState("word");
  const [ignoreCase, setIgnoreCase] = useState(false);
  const [ignorePunct, setIgnorePunct] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPlain, setShowPlain] = useState(false);

  // Uploaded CSV maps (null until parsed)
  const [gtUpload, setGtUpload] = useState(null); // {name, map}
  const [ocrUpload, setOcrUpload] = useState(null);
  const [parsing, setParsing] = useState(""); // status text while parsing

  const full = gtUpload && ocrUpload;
  const gtMap = full ? gtUpload.map : DEMO_GT;
  const ocrMap = full ? ocrUpload.map : DEMO_OCR;

  const available = useMemo(() => {
    const ids = [];
    for (const id of gtMap.keys()) if (ocrMap.has(id)) ids.push(id);
    return ids;
  }, [gtMap, ocrMap]);

  const runEval = async (gt, ocr, opts) => {
    setError("");
    setLoading(true);
    try {
      const data = await evaluate(gt, ocr, opts);
      setResult(data);
    } catch (e) {
      setError(e.message || "Failed to evaluate. Is the backend awake?");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const viewRecord = (id) => {
    const naId = String(id).trim();
    if (!naId) {
      setError("Enter a National Archives ID (NAID).");
      return;
    }
    const gt = gtMap.get(naId);
    const ocr = ocrMap.get(naId);
    if (gt === undefined && ocr === undefined) {
      setError(
        `NAID ${naId} not found${full ? "" : " in the demo set — upload both CSVs to search all 2,260 records"}.`
      );
      return;
    }
    if (gt === undefined) return setError(`No ground-truth row for NAID ${naId}.`);
    if (ocr === undefined) return setError(`No OCR row for NAID ${naId}.`);

    setCurrent({ naId, gt, ocr });
    setShowPlain(false);
    runEval(gt, ocr, { ignoreCase, ignorePunct });
  };

  const random = () => {
    if (!available.length) return;
    const id = available[Math.floor(Math.random() * available.length)];
    setNaidInput(id);
    viewRecord(id);
  };

  const onToggle = (which) => {
    const next = which === "case" ? !ignoreCase : !ignorePunct;
    if (which === "case") setIgnoreCase(next);
    else setIgnorePunct(next);
    if (current) {
      runEval(current.gt, current.ocr, {
        ignoreCase: which === "case" ? next : ignoreCase,
        ignorePunct: which === "punct" ? next : ignorePunct,
      });
    }
  };

  const handleCsv = async (file, side) => {
    setParsing(`Parsing ${file.name}…`);
    setError("");
    try {
      const map = await parseNaraCsv(file, (n) =>
        setParsing(`Parsing ${file.name}… ${fmt(n)} rows`)
      );
      const payload = { name: file.name, map };
      if (side === "gt") setGtUpload(payload);
      else setOcrUpload(payload);
    } catch (e) {
      setError(`Could not parse ${file.name}: ${e.message}`);
    } finally {
      setParsing("");
    }
  };

  const lvl = result ? (mode === "word" ? result.word : result.char) : null;

  return (
    <div className="app">
      <header>
        <div className="brand">NARA · CER / WER</div>
        <p>
          Enter a National Archives ID to compare its human transcription
          (ground truth) against the OCR extraction. Matches are{" "}
          <span className="g">green</span>, errors are <span className="r">red</span>.
        </p>
      </header>

      {/* ---- Lookup controls ---- */}
      <section className="lookup">
        <div className="naid-row">
          <div className="naid-field">
            <label>National Archives ID</label>
            <input
              value={naidInput}
              onChange={(e) => setNaidInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && viewRecord(naidInput)}
              placeholder="e.g. 111406406"
              inputMode="numeric"
            />
          </div>
          <button className="primary" onClick={() => viewRecord(naidInput)} disabled={loading}>
            {loading ? "Evaluating…" : "View record"}
          </button>
          <button onClick={random} disabled={loading || !available.length}>
            Random
          </button>
        </div>

        <div className="norm-opts">
          <label>
            <input type="checkbox" checked={ignoreCase} onChange={() => onToggle("case")} />
            Ignore case
          </label>
          <label>
            <input type="checkbox" checked={ignorePunct} onChange={() => onToggle("punct")} />
            Ignore punctuation
          </label>
        </div>
      </section>

      {/* ---- Dataset / CSV upload ---- */}
      <section className="dataset">
        <div className="dataset-status">
          <span className="dot" style={{ background: full ? "#16a34a" : "#d97706" }} />
          {full ? (
            <strong>Full dataset loaded ({fmt(available.length)} records)</strong>
          ) : (
            <>
              <strong>Demo data loaded</strong> ({available.length} records) —
              upload both CSVs to unlock the full dataset
            </>
          )}
        </div>

        <div className="csv-row">
          <CsvPicker
            label="Ground truth — naid_transcriptions.csv"
            accent="#16a34a"
            filename={gtUpload?.name}
            count={gtUpload?.map.size ?? 0}
            onPick={(f) => handleCsv(f, "gt")}
            disabled={!!parsing}
          />
          <CsvPicker
            label="OCR output — ocr_extraction.csv"
            accent="#2563eb"
            filename={ocrUpload?.name}
            count={ocrUpload?.map.size ?? 0}
            onPick={(f) => handleCsv(f, "ocr")}
            disabled={!!parsing}
          />
        </div>
        {parsing && <div className="parsing">{parsing}</div>}
        {!full && (gtUpload || ocrUpload) && !parsing && (
          <div className="hint">Select both CSVs to unlock all records.</div>
        )}
      </section>

      {error && <div className="error">{error}</div>}

      {/* ---- Results ---- */}
      {result && current && (
        <>
          <section className="metrics">
            <Metric label="CER" value={pct(result.cer)} hint="Character Error Rate" />
            <Metric label="WER" value={pct(result.wer)} hint="Word Error Rate" />
            <Metric label="Char Accuracy" value={pct(result.char_accuracy)} good hint="1 − CER" />
            <Metric label="Word Accuracy" value={pct(result.word_accuracy)} good hint="1 − WER" />
          </section>

          <section className="breakdown">
            <span className="naid-tag">NAID {current.naId}</span>{" "}
            {lvl.counts ? (
              <>
                <strong>{mode === "word" ? "Word" : "Character"} breakdown:</strong>{" "}
                {fmt(lvl.counts.hits)} correct · {fmt(lvl.counts.substitutions)} substitutions ·{" "}
                {fmt(lvl.counts.deletions)} deletions · {fmt(lvl.counts.insertions)} insertions ·
                reference length {fmt(lvl.ref_length)}
              </>
            ) : (
              <>
                <strong>{mode === "word" ? "Word" : "Character"} breakdown:</strong> edit distance{" "}
                {fmt(lvl.distance)} over reference length {fmt(lvl.ref_length)} — exact
                substitution/deletion/insertion split omitted (document too large).
              </>
            )}
          </section>

          <div className="mode-toggle">
            <span>Highlight by:</span>
            <button className={mode === "word" ? "active" : ""} onClick={() => setMode("word")}>
              Word
            </button>
            <button className={mode === "char" ? "active" : ""} onClick={() => setMode("char")}>
              Character
            </button>
            {lvl.truncated && (
              <button className="plain-toggle" onClick={() => setShowPlain((v) => !v)}>
                {showPlain ? "Show highlighted preview" : "Show full plain text"}
              </button>
            )}
          </div>

          {lvl.truncated && !showPlain && (
            <div className="notice">
              Diff preview — first {fmt(lvl.preview_limit)}{" "}
              {mode === "word" ? "words" : "characters"} shown. The CER/WER and accuracy
              above are exact for the <em>full</em> document.
            </div>
          )}

          <section className="compare">
            <div className="compare-col">
              <h3>Ground Truth</h3>
              {showPlain || !lvl.alignment ? (
                <pre className="highlight plain">{current.gt}</pre>
              ) : (
                <HighlightView segments={lvl.alignment.left} />
              )}
            </div>
            <div className="compare-col">
              <h3>OCR Result</h3>
              {showPlain || !lvl.alignment ? (
                <pre className="highlight plain">{current.ocr}</pre>
              ) : (
                <HighlightView segments={lvl.alignment.right} />
              )}
            </div>
          </section>
        </>
      )}

      <footer>
        Frontend: React + Vite · Backend: FastAPI + rapidfuzz · CER/WER via Levenshtein
        alignment
      </footer>
    </div>
  );
}
