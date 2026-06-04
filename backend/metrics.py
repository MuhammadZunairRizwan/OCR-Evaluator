"""
CER / WER computation and alignment for OCR evaluation.

Definitions
-----------
WER (Word Error Rate)      = (S + D + I) / N      over word tokens
CER (Character Error Rate) = (S + D + I) / N      over character tokens

where, comparing the OCR hypothesis against the ground-truth reference:
    S = substitutions
    D = deletions   (token present in reference, missing in hypothesis)
    I = insertions  (token present in hypothesis, absent in reference)
    N = number of tokens in the reference (ground truth)

Accuracy is reported as 1 - error-rate (clamped to 0), i.e. how close the
OCR output is to the human transcription.

Normalization
-------------
Tokens carry a `display` string (the original text, used for highlighting) and
a `key` string (used for comparison). Optional `ignore_case` / `ignore_punct`
flags only affect the key, so the side-by-side view always shows the real text
while the score reflects the chosen leniency.

The same Levenshtein DP that yields the counts also yields an alignment,
which we turn into colour-coded segments for the side-by-side view.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

Status = Literal["match", "error"]

_PUNCT = set(string.punctuation)

# A token is (key, display). key="" means "ignored" and is filtered out.
Token = Tuple[str, str]


@dataclass
class Counts:
    substitutions: int
    deletions: int
    insertions: int
    hits: int          # correct matches
    ref_length: int    # N (tokens in ground truth)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        if self.ref_length == 0:
            # Empty reference: any hypothesis token is an insertion error.
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.ref_length

    @property
    def accuracy(self) -> float:
        return max(0.0, 1.0 - self.error_rate)


@dataclass
class Segment:
    text: str
    status: Status


def _align(ref: Sequence[Token], hyp: Sequence[Token]):
    """
    Levenshtein DP over two token sequences, comparing on token keys.

    Returns (counts, ops) where ops is a list of (type, ref_tok, hyp_tok)
    with type in {"equal", "sub", "del", "ins"}; tokens are (key, display).
    """
    n, m = len(ref), len(hyp)

    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1][0] == hyp[j - 1][0]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(
                    cost[i - 1][j - 1],  # substitution
                    cost[i - 1][j],      # deletion
                    cost[i][j - 1],      # insertion
                )

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and ref[i - 1][0] == hyp[j - 1][0]
            and cost[i][j] == cost[i - 1][j - 1]
        ):
            ops.append(("equal", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()

    s = sum(1 for t, _, _ in ops if t == "sub")
    d = sum(1 for t, _, _ in ops if t == "del")
    ins = sum(1 for t, _, _ in ops if t == "ins")
    hits = sum(1 for t, _, _ in ops if t == "equal")
    counts = Counts(substitutions=s, deletions=d, insertions=ins, hits=hits, ref_length=len(ref))
    return counts, ops


def _segments_from_ops(ops, joiner: str):
    """Build colour-coded segments for the left (ref) and right (hyp) panels."""
    left: List[Segment] = []
    right: List[Segment] = []

    def push(segments: List[Segment], text: str, status: Status):
        if joiner and segments:
            segments.append(Segment(text=joiner, status="match"))
        segments.append(Segment(text=text, status=status))

    for typ, ref_tok, hyp_tok in ops:
        if typ == "equal":
            push(left, ref_tok[1], "match")
            push(right, hyp_tok[1], "match")
        elif typ == "sub":
            push(left, ref_tok[1], "error")
            push(right, hyp_tok[1], "error")
        elif typ == "del":
            push(left, ref_tok[1], "error")
        elif typ == "ins":
            push(right, hyp_tok[1], "error")

    return left, right


def _merge_segments(segments: List[Segment]) -> List[Segment]:
    """Collapse adjacent segments with the same status for a smaller payload."""
    merged: List[Segment] = []
    for seg in segments:
        if merged and merged[-1].status == seg.status:
            merged[-1] = Segment(text=merged[-1].text + seg.text, status=seg.status)
        else:
            merged.append(Segment(text=seg.text, status=seg.status))
    return merged


def _norm_key(s: str, ignore_case: bool, ignore_punct: bool) -> str:
    if ignore_case:
        s = s.lower()
    if ignore_punct:
        s = "".join(c for c in s if c not in _PUNCT)
    return s


def _word_tokens(text: str, ignore_case: bool, ignore_punct: bool) -> List[Token]:
    tokens: List[Token] = []
    for w in text.split():
        key = _norm_key(w, ignore_case, ignore_punct)
        if ignore_punct and key == "":
            # Pure-punctuation token; ignored entirely when stripping punctuation.
            continue
        tokens.append((key, w))
    return tokens


def _char_tokens(text: str, ignore_case: bool, ignore_punct: bool) -> List[Token]:
    tokens: List[Token] = []
    for c in text:
        if ignore_punct and c in _PUNCT:
            continue
        key = c.lower() if ignore_case else c
        tokens.append((key, c))
    return tokens


def evaluate(
    ground_truth: str,
    ocr_text: str,
    ignore_case: bool = False,
    ignore_punct: bool = False,
) -> dict:
    """Compute CER, WER, accuracies and both alignments."""
    # ---- Word level (WER) ----
    ref_words = _word_tokens(ground_truth, ignore_case, ignore_punct)
    hyp_words = _word_tokens(ocr_text, ignore_case, ignore_punct)
    word_counts, word_ops = _align(ref_words, hyp_words)
    word_left, word_right = _segments_from_ops(word_ops, joiner=" ")

    # ---- Character level (CER) ----
    ref_chars = _char_tokens(ground_truth, ignore_case, ignore_punct)
    hyp_chars = _char_tokens(ocr_text, ignore_case, ignore_punct)
    char_counts, char_ops = _align(ref_chars, hyp_chars)
    char_left, char_right = _segments_from_ops(char_ops, joiner="")

    def counts_dict(c: Counts) -> dict:
        return {
            "substitutions": c.substitutions,
            "deletions": c.deletions,
            "insertions": c.insertions,
            "hits": c.hits,
            "ref_length": c.ref_length,
            "errors": c.errors,
        }

    def segs(segments: List[Segment]) -> List[dict]:
        return [{"text": s.text, "status": s.status} for s in _merge_segments(segments)]

    return {
        "wer": word_counts.error_rate,
        "cer": char_counts.error_rate,
        "word_accuracy": word_counts.accuracy,
        "char_accuracy": char_counts.accuracy,
        "word_counts": counts_dict(word_counts),
        "char_counts": counts_dict(char_counts),
        "word_alignment": {"left": segs(word_left), "right": segs(word_right)},
        "char_alignment": {"left": segs(char_left), "right": segs(char_right)},
    }
