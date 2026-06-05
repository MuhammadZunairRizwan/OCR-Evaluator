"""
CER / WER computation and alignment for OCR evaluation.

Definitions
-----------
WER (Word Error Rate)      = (S + D + I) / N      over word tokens
CER (Character Error Rate) = (S + D + I) / N      over character tokens

where, comparing the OCR hypothesis against the ground-truth reference:
    S = substitutions, D = deletions, I = insertions, N = reference tokens.

Accuracy is 1 - error-rate (clamped to 0).

Display units
-------------
Each side is tokenized into a list of *units*, where a unit is
``(display, key)``:

    * ``display`` is the original text shown in the side-by-side view.
    * ``key`` is what gets compared, or ``None`` for an **ignored** unit.

Ignored units (e.g. whitespace when "ignore space" is on, or the spaces
*between* words) are still rendered — shown as **neutral** (no green/red) — but
they take no part in the alignment or the score. This is why turning on
"ignore space" keeps the spaces visible yet stops them counting as errors.

Scale
-----
NARA documents can be very large (200k+ characters). The edit *distance* (all
CER/WER need) is computed with rapidfuzz's C-backed Levenshtein in linear
memory. The green/red alignment needs the full editops, which we compute only
under a size threshold; above it the exact scores still stand and a truncated
preview of the alignment is returned.

Normalization keys only — `ignore_case` / `ignore_punct` / `ignore_space` /
`ignore_newline` change the comparison key (or mark a unit ignored), never the
displayed text.
"""

from __future__ import annotations

import string
from typing import List, Optional, Tuple

from rapidfuzz.distance import Levenshtein

_PUNCT = set(string.punctuation)
_SPACE = {" ", "\t"}
_NEWLINE = {"\n", "\r"}

# Above these sizes we skip the full alignment and show a truncated preview.
CHAR_ALIGN_MAX = 20000   # compared characters
WORD_ALIGN_MAX = 8000    # compared words

# A unit is (display, key); key is None when the unit is ignored (neutral).
Unit = Tuple[str, Optional[str]]


# --------------------------------------------------------------------------- #
# Tokenization into display units
# --------------------------------------------------------------------------- #
def _norm_word_key(w: str, ignore_case: bool, ignore_punct: bool) -> str:
    if ignore_case:
        w = w.lower()
    if ignore_punct:
        w = "".join(c for c in w if c not in _PUNCT)
    return w


def _word_units(text: str, ignore_case: bool, ignore_punct: bool) -> List[Unit]:
    """Words become compared units; the single space between them is neutral."""
    units: List[Unit] = []
    first = True
    for w in text.split():
        if not first:
            units.append((" ", None))  # neutral separator (shown, not compared)
        first = False
        key = _norm_word_key(w, ignore_case, ignore_punct)
        if ignore_punct and key == "":
            units.append((w, None))  # pure-punctuation word: shown, not compared
        else:
            units.append((w, key))
    return units


def _char_units(
    text: str,
    ignore_case: bool,
    ignore_punct: bool,
    ignore_space: bool,
    ignore_newline: bool,
) -> List[Unit]:
    units: List[Unit] = []
    for c in text:
        ignored = (
            (ignore_punct and c in _PUNCT)
            or (ignore_space and c in _SPACE)
            or (ignore_newline and c in _NEWLINE)
        )
        if ignored:
            units.append((c, None))  # shown as neutral, not compared
        else:
            units.append((c, c.lower() if ignore_case else c))
    return units


# --------------------------------------------------------------------------- #
# Alignment helpers
# --------------------------------------------------------------------------- #
def _encode(ref_keys: List[str], hyp_keys: List[str]) -> Tuple[str, str]:
    """Map each distinct key to one code point so rapidfuzz can run on words too."""
    mapping: dict = {}

    def enc(seq: List[str]) -> str:
        return "".join(mapping.setdefault(k, chr(len(mapping))) for k in seq)

    return enc(ref_keys), enc(hyp_keys)


def _statuses(n_ref: int, n_hyp: int, ops) -> Tuple[List[str], List[str]]:
    """Per-compared-token status (match/error) for each side, from editops."""
    ref = ["match"] * n_ref
    hyp = ["match"] * n_hyp
    for op in ops:
        if op.tag == "replace":
            ref[op.src_pos] = "error"
            hyp[op.dest_pos] = "error"
        elif op.tag == "delete":
            ref[op.src_pos] = "error"
        elif op.tag == "insert":
            hyp[op.dest_pos] = "error"
    return ref, hyp


def _segments(units: List[Unit], statuses: List[str]) -> List[dict]:
    """
    Walk display units, colouring compared ones by their status and rendering
    ignored ones as neutral. Adjacent same-status runs are merged.
    """
    segs: List[dict] = []
    si = 0
    for display, key in units:
        if key is None:
            status = "neutral"
        else:
            status = statuses[si]
            si += 1
        if segs and segs[-1]["status"] == status:
            segs[-1]["text"] += display
        else:
            segs.append({"text": display, "status": status})
    return segs


def _prefix_units(units: List[Unit], max_compared: int) -> List[Unit]:
    """First slice of units containing up to `max_compared` compared tokens."""
    out: List[Unit] = []
    c = 0
    for u in units:
        if u[1] is not None:
            if c >= max_compared:
                break
            c += 1
        out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Per-level evaluation
# --------------------------------------------------------------------------- #
def _eval_level(ref_units: List[Unit], hyp_units: List[Unit], align_max: int) -> dict:
    ref_keys = [k for _, k in ref_units if k is not None]
    hyp_keys = [k for _, k in hyp_units if k is not None]
    rs, hs = _encode(ref_keys, hyp_keys)

    distance = Levenshtein.distance(rs, hs)
    n = len(ref_keys)
    if n == 0:
        rate = 0.0 if distance == 0 else 1.0
    else:
        rate = distance / n
    accuracy = max(0.0, 1.0 - rate)

    result = {
        "error_rate": rate,
        "accuracy": accuracy,
        "distance": distance,
        "ref_length": n,
        "hyp_length": len(hyp_keys),
        "included": False,
        "truncated": False,
        "preview_limit": None,
        "counts": None,
        "alignment": None,
    }

    if max(len(rs), len(hs)) <= align_max:
        ops = Levenshtein.editops(rs, hs)
        sub = sum(1 for op in ops if op.tag == "replace")
        dele = sum(1 for op in ops if op.tag == "delete")
        ins = sum(1 for op in ops if op.tag == "insert")
        ref_status, hyp_status = _statuses(n, len(hyp_keys), ops)
        result["included"] = True
        result["counts"] = {
            "substitutions": sub,
            "deletions": dele,
            "insertions": ins,
            "hits": n - sub - dele,
        }
        result["alignment"] = {
            "left": _segments(ref_units, ref_status),
            "right": _segments(hyp_units, hyp_status),
        }
    else:
        # Too large for a full alignment: exact scores stand, show a preview.
        ru = _prefix_units(ref_units, align_max)
        hu = _prefix_units(hyp_units, align_max)
        n_ref_p = sum(1 for _, k in ru if k is not None)
        n_hyp_p = sum(1 for _, k in hu if k is not None)
        ops = Levenshtein.editops(rs[:align_max], hs[:align_max])
        ref_status, hyp_status = _statuses(n_ref_p, n_hyp_p, ops)
        result["truncated"] = True
        result["preview_limit"] = align_max
        result["alignment"] = {
            "left": _segments(ru, ref_status),
            "right": _segments(hu, hyp_status),
        }

    return result


# --------------------------------------------------------------------------- #
# Horizontal (line-by-line) alignment for the side-by-side view
# --------------------------------------------------------------------------- #
# Above this many words on a side, the aligned view is unavailable.
ALIGN_MAX_WORDS = 60_000
# A run of matching words this long becomes its own shared "anchor" row; shorter
# matching runs are absorbed into the surrounding diff block.
_ANCHOR_MIN = 3


def _words_for_align(text: str, ic: bool, ip: bool) -> List[Token]:
    """(display, key) per word. Whitespace is dropped — the layout is re-flowed."""
    return [(w, _norm_word_key(w, ic, ip)) for w in text.split()]


def _row_segs(words):
    """Merge (word, status) pairs into segments, neutral single spaces between."""
    if not words:
        return None  # nothing on this side -> blank (yellow) filler
    segs: List[dict] = []
    for idx, (w, st) in enumerate(words):
        if idx > 0:
            if segs and segs[-1]["status"] == "neutral":
                segs[-1]["text"] += " "
            else:
                segs.append({"text": " ", "status": "neutral"})
        if segs and segs[-1]["status"] == st:
            segs[-1]["text"] += w
        else:
            segs.append({"text": w, "status": st})
    return segs


def _char_all(text: str, ic, ip, isp, inl, status: str):
    """Char segments for one side of a filler/identical row (ignored chars neutral)."""
    segs: List[dict] = []
    for disp, key in _char_units(text, ic, ip, isp, inl):
        st = "neutral" if key is None else status
        if segs and segs[-1]["status"] == st:
            segs[-1]["text"] += disp
        else:
            segs.append({"text": disp, "status": st})
    return segs


def _row_char(lw, rw, ic, ip, isp, inl):
    """Character-level colouring for a row, given its left/right word lists."""
    lt = None if lw is None else " ".join(w for w, _ in lw)
    rt = None if rw is None else " ".join(w for w, _ in rw)
    if lt is None:
        return None, _char_all(rt, ic, ip, isp, inl, "error")
    if rt is None:
        return _char_all(lt, ic, ip, isp, inl, "error"), None
    if lt == rt:
        return (
            _char_all(lt, ic, ip, isp, inl, "match"),
            _char_all(rt, ic, ip, isp, inl, "match"),
        )
    lu = _char_units(lt, ic, ip, isp, inl)
    ru = _char_units(rt, ic, ip, isp, inl)
    lk = [k for _, k in lu if k is not None]
    rk = [k for _, k in ru if k is not None]
    rs, hs = _encode(lk, rk)
    ops = Levenshtein.editops(rs, hs)
    ls, rss = _statuses(len(lk), len(rk), ops)
    return _segments(lu, ls), _segments(ru, rss)


def align_lines(
    ground_truth: str,
    ocr_text: str,
    ignore_case: bool = False,
    ignore_punct: bool = False,
    ignore_space: bool = False,
    ignore_newline: bool = False,
    level: str = "word",
) -> dict:
    """
    Word-alignment-driven side-by-side layout.

    The two sides usually have different line breaks (hand transcription vs OCR
    paragraphs), so pairing original lines leaves big gaps. Instead we align on
    the global word diff: long runs of matching words become shared "anchor"
    rows that keep the two columns in sync, and the changes between anchors are
    grouped into aligned blocks. A block present on only one side gets a blank
    (yellow) filler opposite.

    The *layout* is always word-anchored; the *colouring* follows `level`:
    "word" colours whole words (matching the WER), "char" diffs each row at the
    character level so only the differing characters within a word are red.
    """
    g = _words_for_align(ground_truth, ignore_case, ignore_punct)
    o = _words_for_align(ocr_text, ignore_case, ignore_punct)
    if max(len(g), len(o)) > ALIGN_MAX_WORDS:
        return {"available": False, "reason": "too_large", "rows": []}

    rs, hs = _encode([k for _, k in g], [k for _, k in o])
    ops = Levenshtein.editops(rs, hs)

    # Expand editops into a full tagged item stream (equal/sub/del/ins).
    items = []
    i = j = 0
    for op in ops:
        while i < op.src_pos:
            items.append(("equal", g[i][0], o[j][0]))
            i += 1
            j += 1
        if op.tag == "replace":
            items.append(("sub", g[i][0], o[j][0]))
            i += 1
            j += 1
        elif op.tag == "delete":
            items.append(("del", g[i][0], None))
            i += 1
        else:  # insert
            items.append(("ins", None, o[j][0]))
            j += 1
    while i < len(g):
        items.append(("equal", g[i][0], o[j][0]))
        i += 1
        j += 1

    # Group consecutive items into equal-runs vs diff-runs.
    runs = []
    for it in items:
        kind = "equal" if it[0] == "equal" else "diff"
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(it)
        else:
            runs.append([kind, [it]])

    # Build raw rows as left/right word lists (None = blank filler side).
    raw = []

    def emit_diff(buf):
        if not buf:
            return
        left, right = [], []
        for tag, gw, ow in buf:
            if tag == "equal":
                left.append((gw, "match"))
                right.append((ow, "match"))
            elif tag == "sub":
                left.append((gw, "error"))
                right.append((ow, "error"))
            elif tag == "del":
                left.append((gw, "error"))
            else:
                right.append((ow, "error"))
        raw.append({"left": left or None, "right": right or None})

    buf = []
    for kind, its in runs:
        if kind == "equal" and len(its) >= _ANCHOR_MIN:
            emit_diff(buf)
            buf = []
            raw.append(
                {
                    "left": [(gw, "match") for _, gw, _ in its],
                    "right": [(ow, "match") for _, _, ow in its],
                }
            )
        else:
            buf.extend(its)
    emit_diff(buf)

    # Render rows at the requested granularity.
    rows = []
    if level == "char":
        for r in raw:
            left, right = _row_char(
                r["left"], r["right"],
                ignore_case, ignore_punct, ignore_space, ignore_newline,
            )
            rows.append({"left": left, "right": right})
    else:
        for r in raw:
            rows.append({"left": _row_segs(r["left"]), "right": _row_segs(r["right"])})

    return {"available": True, "rows": rows}


def evaluate(
    ground_truth: str,
    ocr_text: str,
    ignore_case: bool = False,
    ignore_punct: bool = False,
    ignore_space: bool = False,
    ignore_newline: bool = False,
) -> dict:
    """Compute CER, WER, accuracies and (where feasible) alignments."""
    word = _eval_level(
        _word_units(ground_truth, ignore_case, ignore_punct),
        _word_units(ocr_text, ignore_case, ignore_punct),
        WORD_ALIGN_MAX,
    )
    char = _eval_level(
        _char_units(ground_truth, ignore_case, ignore_punct, ignore_space, ignore_newline),
        _char_units(ocr_text, ignore_case, ignore_punct, ignore_space, ignore_newline),
        CHAR_ALIGN_MAX,
    )

    return {
        "cer": char["error_rate"],
        "wer": word["error_rate"],
        "char_accuracy": char["accuracy"],
        "word_accuracy": word["accuracy"],
        "char": char,
        "word": word,
    }
