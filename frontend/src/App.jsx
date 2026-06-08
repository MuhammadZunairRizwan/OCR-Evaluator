import { useMemo, useRef, useState } from "react";
import { evaluate } from "./api.js";
import { parseNaraCsv } from "./csv.js";
import { DEMO_RECORDS } from "./demoData.js";

const pct = (x) => `${(x * 100).toFixed(2)}%`;
const fmt = (n) => n.toLocaleString();

// ---- Demo maps (available before any CSV is uploaded) ----
const DEMO_GT = new Map(DEMO_RECORDS.map((r) => [r.naId, r.gt]));
const DEMO_OCR = new Map(DEMO_RECORDS.map((r) => [r.naId, r.ocr]));

const SEG_CLASS = { error: "seg-error", neutral: "seg-neutral", match: "seg-match" };

function HighlightView({ segments }) {
  return (
    <pre className="highlight">
      {segments.map((seg, i) => (
        <span key={i} className={SEG_CLASS[seg.status] || "seg-match"}>
          {seg.text}
        </span>
      ))}
    </pre>
  );
}

function AlignCell({ segs, side }) {
  // segs === null  -> blank yellow filler (this line exists only on the other side)
  // segs === []    -> a genuine blank line
  const cls =
    "align-cell" + (side === "left" ? " left" : "") + (segs === null ? " align-filler" : "");
  return (
    <div className={cls}>
      {segs &&
        segs.map((s, i) => (
          <span key={i} className={SEG_CLASS[s.status] || "seg-match"}>
            {s.text}
          </span>
        ))}
    </div>
  );
}

function AlignedView({ rows }) {
  return (
    <div className="aligned">
      <div className="align-row align-head">
        <div className="align-cell left">Ground Truth</div>
        <div className="align-cell">OCR Result</div>
      </div>
      {rows.map((r, i) => (
        <div className={"align-row" + (r.moved ? " moved" : "")} key={i}>
          <AlignCell segs={r.left} side="left" />
          <AlignCell segs={r.right} side="right" />
        </div>
      ))}
    </div>
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
  const [norm, setNorm] = useState({
    ignoreCase: false,
    ignorePunct: false,
    ignoreSpace: false,
    ignoreNewline: false,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showPlain, setShowPlain] = useState(false);
  const [alignView, setAlignView] = useState(false);

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

  const runEval = async (gt, ocr, normObj, align, level) => {
    setError("");
    setLoading(true);
    try {
      const data = await evaluate(gt, ocr, { ...normObj, align, alignLevel: level });
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
    runEval(gt, ocr, norm, alignView, mode);
  };

  const random = () => {
    if (!available.length) return;
    const id = available[Math.floor(Math.random() * available.length)];
    setNaidInput(id);
    viewRecord(id);
  };

  const toggleNorm = (key) => {
    const next = { ...norm, [key]: !norm[key] };
    setNorm(next);
    if (current) runEval(current.gt, current.ocr, next, alignView, mode);
  };

  const toggleAlign = () => {
    const next = !alignView;
    setAlignView(next);
    if (current) runEval(current.gt, current.ocr, norm, next, mode);
  };

  const changeMode = (m) => {
    setMode(m);
    // The aligned view is colour-rendered on the server, so re-fetch on mode change.
    if (current && alignView) runEval(current.gt, current.ocr, norm, true, m);
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
  const aligned = result?.aligned;
  const showAligned = alignView && aligned && aligned.available;

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
            <input
              type="checkbox"
              checked={norm.ignoreCase}
              onChange={() => toggleNorm("ignoreCase")}
            />
            Ignore case
          </label>
          <label>
            <input
              type="checkbox"
              checked={norm.ignorePunct}
              onChange={() => toggleNorm("ignorePunct")}
            />
            Ignore punctuation
          </label>
          <label>
            <input
              type="checkbox"
              checked={norm.ignoreSpace}
              onChange={() => toggleNorm("ignoreSpace")}
            />
            Ignore space
          </label>
          <label>
            <input
              type="checkbox"
              checked={norm.ignoreNewline}
              onChange={() => toggleNorm("ignoreNewline")}
            />
            Ignore newline
          </label>
          <span className="norm-note">(space &amp; newline affect CER only)</span>
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
            <button
              className={mode === "word" ? "active" : ""}
              onClick={() => changeMode("word")}
            >
              Word
            </button>
            <button
              className={mode === "char" ? "active" : ""}
              onClick={() => changeMode("char")}
            >
              Character
            </button>
            <label className="align-check">
              <input type="checkbox" checked={alignView} onChange={toggleAlign} />
              Horizontal align
            </label>
            {lvl.truncated && !showAligned && (
              <button className="plain-toggle" onClick={() => setShowPlain((v) => !v)}>
                {showPlain ? "Show highlighted preview" : "Show full plain text"}
              </button>
            )}
          </div>

          {alignView && aligned && !aligned.available && (
            <div className="notice">
              Horizontal align isn’t available for very large documents — showing the
              standard view instead. The scores above are still exact.
            </div>
          )}

          {lvl.truncated && !showAligned && !showPlain && (
            <div className="notice">
              Diff preview — first {fmt(lvl.preview_limit)}{" "}
              {mode === "word" ? "words" : "characters"} shown. The CER/WER and accuracy
              above are exact for the <em>full</em> document.
            </div>
          )}

          {showAligned ? (
            <>
              <div className="align-legend">
                <span><i className="sw filler" /> blank filler (added to align — original text never removed)</span>
                <span>matching content stays green even when reordered or re-wrapped</span>
              </div>
              <AlignedView rows={aligned.rows} />
            </>
          ) : (
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
          )}
        </>
      )}

      <footer>
        Frontend: React + Vite · Backend: FastAPI + rapidfuzz · CER/WER via Levenshtein
        alignment
      </footer>
    </div>
  );
}
