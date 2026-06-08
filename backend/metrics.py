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

import re
import string
from typing import List, Optional, Tuple

from rapidfuzz.distance import Levenshtein

_PUNCT = set(string.punctuation)
_SPACE = {" ", "\t"}
_NEWLINE = {"\n", "\r"}
_WS_RE = re.compile(r"(\s+)")  # split keeping whitespace runs

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
    """
    Split into words (compared) and whitespace runs (neutral). The original
    whitespace — every space, tab and newline — is preserved exactly as a
    neutral unit, so the displayed text keeps its real formatting; only the
    words take part in the comparison.
    """
    units: List[Unit] = []
    for tok in _WS_RE.split(text):
        if not tok:
            continue
        if tok.strip() == "":  # a run of whitespace (spaces / tabs / newlines)
            units.append((tok, None))
            continue
        key = _norm_word_key(tok, ignore_case, ignore_punct)
        if ignore_punct and key == "":
            units.append((tok, None))  # pure-punctuation word: shown, not compared
        else:
            units.append((tok, key))
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
# Above this many line cells (n*m) the aligned view is unavailable.
LINE_ALIGN_MAX_CELLS = 600_000
_LINE_GAP = -0.4  # Needleman-Wunsch penalty for leaving a line unpaired
# Reordering: an unmatched line may be pulled to a matching unmatched line on the
# other side if within this many rows and at least this similar.
REORDER_WINDOW = 4
REORDER_MIN_SIM = 0.55
# Many-to-one line merging: up to this many short lines on one side may merge to
# match one long line on the other (handles OCR that joins/splits lines).
_MAX_MERGE = 3


def _line_key(s: str, ic: bool, ip: bool, isp: bool, inl: bool) -> str:
    if ic:
        s = s.lower()
    if ip:
        s = "".join(c for c in s if c not in _PUNCT)
    if isp:
        s = s.replace(" ", "").replace("\t", "")
    return s.strip()


def _line_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return Levenshtein.normalized_similarity(a, b)


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


def _word_all(text: str, ic, ip, status: str):
    """Word segments for one line at a single status (whitespace neutral)."""
    segs: List[dict] = []
    for disp, key in _word_units(text, ic, ip):
        st = "neutral" if key is None else status
        if segs and segs[-1]["status"] == st:
            segs[-1]["text"] += disp
        else:
            segs.append({"text": disp, "status": st})
    return segs


def _char_diff_lines(a, b, ic, ip, isp, inl):
    """Character-level colouring of two lines against each other."""
    if a == b:
        return _char_all(a, ic, ip, isp, inl, "match"), _char_all(b, ic, ip, isp, inl, "match")
    lu = _char_units(a, ic, ip, isp, inl)
    ru = _char_units(b, ic, ip, isp, inl)
    lk = [k for _, k in lu if k is not None]
    rk = [k for _, k in ru if k is not None]
    rs, hs = _encode(lk, rk)
    ops = Levenshtein.editops(rs, hs)
    ls, rss = _statuses(len(lk), len(rk), ops)
    return _segments(lu, ls), _segments(ru, rss)


def _word_diff_lines(a, b, ic, ip):
    """Word-level colouring of two lines against each other (for moved rows)."""
    res = _eval_level(_word_units(a, ic, ip), _word_units(b, ic, ip), WORD_ALIGN_MAX)
    al = res["alignment"]
    if al:
        return al["left"], al["right"]
    left = [{"text": a, "status": "match"}] if a else []
    right = [{"text": b, "status": "match"}] if b else []
    return left, right


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
    Line-preserving side-by-side layout.

    Each side keeps its ORIGINAL lines — every space, indent and blank line is
    shown exactly. Lines are paired by a Needleman-Wunsch alignment over line
    similarity; a line with no partner gets a blank YELLOW filler opposite so the
    columns stay level (we only ADD blank space, never remove text). A short
    reorder pass can pull an unmatched ground-truth line up/down a few rows to
    meet a matching line on the other side — those rows are marked "moved"
    (rendered blue). Reordering is a visual aid only; the CER/WER are unchanged
    (they remain the standard in-order scores).

    Colouring follows `level`: "word" uses the GLOBAL word alignment (so red
    matches the WER); "char" diffs each paired line at the character level.
    """
    ic, ip, isp, inl = ignore_case, ignore_punct, ignore_space, ignore_newline
    g_lines = [x.rstrip("\r") for x in ground_truth.split("\n")]
    o_lines = [x.rstrip("\r") for x in ocr_text.split("\n")]
    n, m = len(g_lines), len(o_lines)
    if n * m > LINE_ALIGN_MAX_CELLS:
        return {"available": False, "reason": "too_large", "rows": []}

    # Line-structure alignment with many-to-one merging (Needleman-Wunsch).
    gk = [_line_key(x, ic, ip, isp, inl) for x in g_lines]
    ok = [_line_key(x, ic, ip, isp, inl) for x in o_lines]
    glen = [len(x) for x in gk]
    olen = [len(x) for x in ok]
    NEG = float("-inf")
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    bp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _LINE_GAP
        bp[i][0] = ("del", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _LINE_GAP
        bp[0][j] = ("ins", 0, j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = dp[i - 1][j] + _LINE_GAP
            bb = ("del", i - 1, j)
            v = dp[i][j - 1] + _LINE_GAP
            if v > best:
                best, bb = v, ("ins", i, j - 1)
            v = dp[i - 1][j - 1] + (2 * _line_sim(gk[i - 1], ok[j - 1]) - 1)
            if v > best:
                best, bb = v, ("pair", i - 1, j - 1)
            oj_len = olen[j - 1]
            clen = glen[i - 1]
            for k in range(2, _MAX_MERGE + 1):  # k GT lines -> 1 OCR line
                if i - k < 0:
                    break
                clen += glen[i - k] + 1
                if oj_len and clen > 1.6 * oj_len:
                    break  # only grows from here
                if oj_len and clen < 0.5 * oj_len:
                    continue  # too short yet — try more lines
                v = dp[i - k][j - 1] + (2 * _line_sim(" ".join(gk[i - k:i]), ok[j - 1]) - 1)
                if v > best:
                    best, bb = v, ("mergeG", i - k, j - 1, k)
            gi_len = glen[i - 1]
            clen = olen[j - 1]
            for k in range(2, _MAX_MERGE + 1):  # 1 GT line -> k OCR lines
                if j - k < 0:
                    break
                clen += olen[j - k] + 1
                if gi_len and clen > 1.6 * gi_len:
                    break
                if gi_len and clen < 0.5 * gi_len:
                    continue
                v = dp[i - 1][j - k] + (2 * _line_sim(gk[i - 1], " ".join(ok[j - k:j])) - 1)
                if v > best:
                    best, bb = v, ("mergeO", i - 1, j - k, k)
            dp[i][j] = best
            bp[i][j] = bb

    line_ops = []
    i, j = n, m
    while i > 0 or j > 0:
        tag = bp[i][j]
        if tag[0] == "pair":
            line_ops.append(["pair", tag[1], tag[2]])
            i, j = tag[1], tag[2]
        elif tag[0] == "del":
            line_ops.append(["del", i - 1, None])
            i, j = tag[1], tag[2]
        elif tag[0] == "ins":
            line_ops.append(["ins", None, j - 1])
            i, j = tag[1], tag[2]
        elif tag[0] == "mergeG":
            pi, pj, k = tag[1], tag[2], tag[3]
            line_ops.append(["mergeG", list(range(pi, pi + k)), pj])
            i, j = pi, pj
        else:  # mergeO
            pi, pj, k = tag[1], tag[2], tag[3]
            line_ops.append(["mergeO", pi, list(range(pj, pj + k))])
            i, j = pi, pj
    line_ops.reverse()

    # Reorder pass: pull an unmatched ground-truth line up/down to a nearby
    # matching OCR line, rendering that row as "moved" (blue). Visual aid only.
    used = set()
    moved = {}  # ins op-index -> ground-truth line index pulled here
    for p, op in enumerate(line_ops):
        if op[0] != "ins":
            continue
        oj = op[2]
        best, best_sim = None, REORDER_MIN_SIM
        for q in range(max(0, p - REORDER_WINDOW), min(len(line_ops), p + REORDER_WINDOW + 1)):
            cand = line_ops[q]
            if cand[0] != "del" or q in used:
                continue
            sim = _line_sim(gk[cand[1]], ok[oj])
            if sim >= best_sim:
                best_sim, best = sim, q
        if best is not None:
            used.add(best)
            moved[p] = line_ops[best][1]

    def pair_segs(gline, oline):
        if level == "char":
            return _char_diff_lines(gline, oline, ic, ip, isp, inl)
        return _word_diff_lines(gline, oline, ic, ip)

    def one_side(text, is_left):
        if text.strip() == "":
            return [], []  # blank line is just spacing — no "missing content" filler
        seg = _char_all(text, ic, ip, isp, inl, "error") if level == "char" else _word_all(text, ic, ip, "error")
        return (seg, None) if is_left else (None, seg)

    rows = []
    for p, op in enumerate(line_ops):
        tag = op[0]
        if tag == "del" and p in used:
            continue  # this ground-truth line was moved into a 'moved' row below
        if tag == "pair":
            left, right = pair_segs(g_lines[op[1]], o_lines[op[2]])
            rows.append({"left": left, "right": right})
        elif tag == "mergeG":  # several GT lines stacked against one OCR line
            gt_text = "\n".join(g_lines[x] for x in op[1])
            left, right = pair_segs(gt_text, o_lines[op[2]])
            rows.append({"left": left, "right": right})
        elif tag == "mergeO":  # one GT line against several OCR lines
            oc_text = "\n".join(o_lines[x] for x in op[2])
            left, right = pair_segs(g_lines[op[1]], oc_text)
            rows.append({"left": left, "right": right})
        elif tag == "del":
            left, right = one_side(g_lines[op[1]], True)
            rows.append({"left": left, "right": right})
        else:  # ins
            if p in moved:
                left, right = pair_segs(g_lines[moved[p]], o_lines[op[2]])
                rows.append({"left": left, "right": right, "moved": True})
            else:
                left, right = one_side(o_lines[op[2]], False)
                rows.append({"left": left, "right": right})

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
