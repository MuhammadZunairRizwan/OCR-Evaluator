"""
CER / WER computation and alignment for OCR evaluation.

Definitions
-----------
WER (Word Error Rate)      = (S + D + I) / N      over word tokens
CER (Character Error Rate) = (S + D + I) / N      over character tokens

where, comparing the OCR hypothesis against the ground-truth reference:
    S = substitutions, D = deletions, I = insertions, N = reference tokens.

Accuracy is 1 - error-rate (clamped to 0).

Scale
-----
NARA documents can be very large (200k+ characters). A naive O(n*m) DP matrix
is impossible at that size, so the edit *distance* (which is all CER/WER need)
is computed with rapidfuzz's C-backed Levenshtein in linear memory — fast even
for the biggest records.

The green/red side-by-side highlighting needs the full alignment (editops). We
always produce it for documents under a size threshold. For larger documents we
still report the exact full-document scores, plus a *truncated preview* of the
alignment (first N tokens) so there is something to eyeball; the precise
substitution/deletion/insertion split is omitted in that case.

Normalization
-------------
Tokens carry a `display` string (original text, for highlighting) and a `key`
string (for comparison). `ignore_case` / `ignore_punct` only affect the key, so
the diff always shows real text while the score reflects the chosen leniency.
"""

from __future__ import annotations

import string
from typing import List, Optional, Tuple

from rapidfuzz.distance import Levenshtein

_PUNCT = set(string.punctuation)

# Above these sizes we skip the full alignment and show a truncated preview.
CHAR_ALIGN_MAX = 20000   # characters
WORD_ALIGN_MAX = 8000    # words

# A token is (key, display).
Token = Tuple[str, str]


# --------------------------------------------------------------------------- #
# Tokenization + normalization
# --------------------------------------------------------------------------- #
def _norm_key(s: str, ignore_case: bool, ignore_punct: bool) -> str:
    if ignore_case:
        s = s.lower()
    if ignore_punct:
        s = "".join(c for c in s if c not in _PUNCT)
    return s


def _word_tokens(text: str, ignore_case: bool, ignore_punct: bool) -> List[Token]:
    # Word tokenization already drops all whitespace (space/tab/newline) via
    # str.split(), so ignore_space / ignore_newline have no effect at word level.
    tokens: List[Token] = []
    for w in text.split():
        key = _norm_key(w, ignore_case, ignore_punct)
        if ignore_punct and key == "":
            continue  # pure-punctuation token ignored when stripping punctuation
        tokens.append((key, w))
    return tokens


def _char_tokens(
    text: str,
    ignore_case: bool,
    ignore_punct: bool,
    ignore_space: bool,
    ignore_newline: bool,
) -> List[Token]:
    skip: set = set()
    if ignore_punct:
        skip |= _PUNCT
    if ignore_space:
        skip |= {" ", "\t"}
    if ignore_newline:
        skip |= {"\n", "\r"}
    tokens: List[Token] = []
    for c in text:
        if c in skip:
            continue
        key = c.lower() if ignore_case else c
        tokens.append((key, c))
    return tokens


def _encode(ref_keys: List[str], hyp_keys: List[str]) -> Tuple[str, str]:
    """
    Map each distinct token key to a single Unicode code point so we can run
    rapidfuzz's fast string Levenshtein on word tokens too (not just chars).
    Unique keys per document pair are far below the Unicode limit.
    """
    mapping: dict = {}

    def enc(seq: List[str]) -> str:
        return "".join(mapping.setdefault(k, chr(len(mapping))) for k in seq)

    return enc(ref_keys), enc(hyp_keys)


# --------------------------------------------------------------------------- #
# Alignment -> coloured segments
# --------------------------------------------------------------------------- #
def _segments(ref_tokens, hyp_tokens, ops, joiner: str):
    """Walk editops to build coloured segments for the left/right panels."""
    left: List[list] = []
    right: List[list] = []

    def push(arr, text, status):
        if joiner and arr:
            arr.append([joiner, "match"])
        arr.append([text, status])

    i = j = 0
    for op in ops:
        sp, dp = op.src_pos, op.dest_pos
        while i < sp:
            push(left, ref_tokens[i][1], "match")
            i += 1
        while j < dp:
            push(right, hyp_tokens[j][1], "match")
            j += 1
        if op.tag == "replace":
            push(left, ref_tokens[i][1], "error")
            push(right, hyp_tokens[j][1], "error")
            i += 1
            j += 1
        elif op.tag == "delete":
            push(left, ref_tokens[i][1], "error")
            i += 1
        elif op.tag == "insert":
            push(right, hyp_tokens[j][1], "error")
            j += 1
    while i < len(ref_tokens):
        push(left, ref_tokens[i][1], "match")
        i += 1
    while j < len(hyp_tokens):
        push(right, hyp_tokens[j][1], "match")
        j += 1

    return _merge(left), _merge(right)


def _merge(segments: List[list]) -> List[dict]:
    """Collapse adjacent same-status segments for a smaller payload."""
    out: List[dict] = []
    for text, status in segments:
        if out and out[-1]["status"] == status:
            out[-1]["text"] += text
        else:
            out.append({"text": text, "status": status})
    return out


# --------------------------------------------------------------------------- #
# Per-level evaluation
# --------------------------------------------------------------------------- #
def _eval_level(ref_tokens, hyp_tokens, align_max: int, joiner: str) -> dict:
    ref_keys = [t[0] for t in ref_tokens]
    hyp_keys = [t[0] for t in hyp_tokens]
    rs, hs = _encode(ref_keys, hyp_keys)

    distance = Levenshtein.distance(rs, hs)
    n = len(ref_tokens)
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
        "hyp_length": len(hyp_tokens),
        "included": False,
        "truncated": False,
        "preview_limit": None,
        "counts": None,
        "alignment": None,
    }

    big = max(len(rs), len(hs))
    if big <= align_max:
        # Full alignment with exact substitution/deletion/insertion split.
        ops = Levenshtein.editops(rs, hs)
        sub = sum(1 for op in ops if op.tag == "replace")
        dele = sum(1 for op in ops if op.tag == "delete")
        ins = sum(1 for op in ops if op.tag == "insert")
        left, right = _segments(ref_tokens, hyp_tokens, ops, joiner)
        result["included"] = True
        result["counts"] = {
            "substitutions": sub,
            "deletions": dele,
            "insertions": ins,
            "hits": n - sub - dele,
        }
        result["alignment"] = {"left": left, "right": right}
    else:
        # Too large for a full alignment: exact scores stand, show a preview.
        rt = ref_tokens[:align_max]
        ht = hyp_tokens[:align_max]
        ops = Levenshtein.editops(rs[:align_max], hs[:align_max])
        left, right = _segments(rt, ht, ops, joiner)
        result["truncated"] = True
        result["preview_limit"] = align_max
        result["alignment"] = {"left": left, "right": right}

    return result


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
        _word_tokens(ground_truth, ignore_case, ignore_punct),
        _word_tokens(ocr_text, ignore_case, ignore_punct),
        WORD_ALIGN_MAX,
        joiner=" ",
    )
    char = _eval_level(
        _char_tokens(ground_truth, ignore_case, ignore_punct, ignore_space, ignore_newline),
        _char_tokens(ocr_text, ignore_case, ignore_punct, ignore_space, ignore_newline),
        CHAR_ALIGN_MAX,
        joiner="",
    )

    return {
        "cer": char["error_rate"],
        "wer": word["error_rate"],
        "char_accuracy": char["accuracy"],
        "word_accuracy": word["accuracy"],
        "char": char,
        "word": word,
    }
