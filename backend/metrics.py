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
# Cap the line-alignment DP so it stays fast; above this we report unavailable.
LINE_ALIGN_MAX_CELLS = 300_000
_LINE_GAP = -0.4  # penalty for leaving a line unpaired (insert/delete)


def _line_key(s: str, ic: bool, ip: bool, isp: bool) -> str:
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


def _line_segs(text: str, status: str) -> List[dict]:
    return [{"text": text, "status": status}] if text else []


def _pair_row(lg: str, lo: str, kg: str, ko: str, ic: bool, ip: bool) -> dict:
    """A row where both sides have a line: colour by within-line word diff."""
    if kg == ko:
        return {"left": _line_segs(lg, "match"), "right": _line_segs(lo, "match")}
    res = _eval_level(_word_units(lg, ic, ip), _word_units(lo, ic, ip), WORD_ALIGN_MAX)
    al = res["alignment"]
    if not al:
        return {"left": _line_segs(lg, "match"), "right": _line_segs(lo, "match")}
    return {"left": al["left"], "right": al["right"]}


def align_lines(
    ground_truth: str,
    ocr_text: str,
    ignore_case: bool = False,
    ignore_punct: bool = False,
    ignore_space: bool = False,
    ignore_newline: bool = False,
) -> dict:
    """
    Needleman-Wunsch alignment over lines so matching blocks line up vertically.
    Returns rows of ``{left, right}`` segment lists; a ``None`` side is a blank
    (yellow) filler. Lines pair up fuzzily (by similarity) so OCR errors within a
    line don't break the pairing.
    """
    gl = [x.rstrip("\r") for x in ground_truth.split("\n")]
    ol = [x.rstrip("\r") for x in ocr_text.split("\n")]
    n, m = len(gl), len(ol)
    if n * m > LINE_ALIGN_MAX_CELLS:
        return {"available": False, "reason": "too_large", "rows": []}

    gk = [_line_key(x, ignore_case, ignore_punct, ignore_space) for x in gl]
    ok = [_line_key(x, ignore_case, ignore_punct, ignore_space) for x in ol]

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _LINE_GAP
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _LINE_GAP
    for i in range(1, n + 1):
        gi = gk[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            pair = prev[j - 1] + (2 * _line_sim(gi, ok[j - 1]) - 1)
            row[j] = max(pair, prev[j] + _LINE_GAP, row[j - 1] + _LINE_GAP)

    # Backtrace
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = _line_sim(gk[i - 1], ok[j - 1])
            if abs(dp[i][j] - (dp[i - 1][j - 1] + (2 * s - 1))) < 1e-9:
                ops.append(("pair", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + _LINE_GAP)) < 1e-9:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()

    rows = []
    for tag, i0, j0 in ops:
        if tag == "pair":
            rows.append(_pair_row(gl[i0], ol[j0], gk[i0], ok[j0], ignore_case, ignore_punct))
        elif tag == "del":
            rows.append({"left": _line_segs(gl[i0], "error"), "right": None})
        else:
            rows.append({"left": None, "right": _line_segs(ol[j0], "error")})

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
