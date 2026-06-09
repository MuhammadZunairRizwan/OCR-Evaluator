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
# Above this many words on a side, the aligned view is unavailable.
ALIGN_MAX_WORDS = 60_000
# A run of matching words this long becomes a shared "anchor" that keeps the
# columns in sync; shorter matches are absorbed into the surrounding diff block.
_ANCHOR_MIN = 3
_WORD_RE = re.compile(r"\S+")
# Block-level reordering: a one-sided ground-truth block may pair with a similar
# one-sided OCR block within this many blocks (rendered blue). Visual aid only.
REORDER_BLOCK_WINDOW = 12
REORDER_BLOCK_SIM = 0.45


def _line_key(s: str, ic: bool, ip: bool, isp: bool, inl: bool) -> str:
    if ic:
        s = s.lower()
    if ip:
        s = "".join(c for c in s if c not in _PUNCT)
    if isp:
        s = s.replace(" ", "").replace("\t", "")
    return " ".join(s.split())


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
    Word-alignment-driven, formatting-preserving side-by-side layout.

    The two sides often break lines very differently (hand transcription in short
    lines vs OCR paragraphs), so we do NOT pair raw lines. Instead we align on the
    global word diff: long runs of matching words become shared "anchor" blocks
    that keep the columns in sync, and the changes between anchors form their own
    blocks. For every block we slice the ORIGINAL text back out, so every space,
    indent and newline is preserved exactly (we only ADD blank yellow fillers
    where a block exists on one side only). Each block is coloured by diffing its
    two sides, so matching content is green even where the document was reordered
    or the OCR merged/split lines. The CER/WER are computed separately and
    unaffected.
    """
    ic, ip, isp, inl = ignore_case, ignore_punct, ignore_space, ignore_newline
    gspan = [mt.span() for mt in _WORD_RE.finditer(ground_truth)]
    ospan = [mt.span() for mt in _WORD_RE.finditer(ocr_text)]
    if max(len(gspan), len(ospan)) > ALIGN_MAX_WORDS:
        return {"available": False, "reason": "too_large", "rows": []}

    gkeys = [_norm_word_key(ground_truth[s:e], ic, ip) for s, e in gspan]
    okeys = [_norm_word_key(ocr_text[s:e], ic, ip) for s, e in ospan]
    rs, hs = _encode(gkeys, okeys)
    ops = Levenshtein.editops(rs, hs)

    # Expand editops into a tagged stream of (kind, gt_word_idx, ocr_word_idx).
    items = []
    i = j = 0
    for op in ops:
        while i < op.src_pos:
            items.append(("equal", i, j)); i += 1; j += 1
        if op.tag == "replace":
            items.append(("diff", i, j)); i += 1; j += 1
        elif op.tag == "delete":
            items.append(("diff", i, None)); i += 1
        else:
            items.append(("diff", None, j)); j += 1
    while i < len(gspan):
        items.append(("equal", i, j)); i += 1; j += 1

    def render(lt, rt):
        # lt / rt are the original-text slices for this block (None = filler).
        if lt is None and rt is None:
            return None
        if lt is None:
            seg = _char_all(rt, ic, ip, isp, inl, "error") if level == "char" else _word_all(rt, ic, ip, "error")
            return {"left": None, "right": seg}
        if rt is None:
            seg = _char_all(lt, ic, ip, isp, inl, "error") if level == "char" else _word_all(lt, ic, ip, "error")
            return {"left": seg, "right": None}
        if level == "char":
            left, right = _char_diff_lines(lt, rt, ic, ip, isp, inl)
        else:
            left, right = _word_diff_lines(lt, rt, ic, ip)
        return {"left": left, "right": right}

    # Build blocks by type so deletions and insertions are SEPARATE one-sided
    # blocks (this lets the reorder pass pair a moved GT block with its OCR copy).
    #   paired = equal or substitution (both sides, rendered side-by-side)
    #   del    = ground-truth-only run    ins = OCR-only run
    def itype(it):
        if it[0] == "equal" or (it[1] is not None and it[2] is not None):
            return "paired"
        return "del" if it[1] is not None else "ins"

    blocks = []
    cur_t, cur_g, cur_o = None, [], []
    for it in items:
        t = itype(it)
        if t != cur_t:
            if cur_g or cur_o:
                blocks.append({"g": cur_g, "o": cur_o, "type": cur_t})
            cur_t, cur_g, cur_o = t, [], []
        if it[1] is not None:
            cur_g.append(it[1])
        if it[2] is not None:
            cur_o.append(it[2])
    if cur_g or cur_o:
        blocks.append({"g": cur_g, "o": cur_o, "type": cur_t})

    # Absorb tiny one-sided blocks (1-2 stray tokens, e.g. OCR markup "##", "}")
    # into the previous block so they render inline instead of as a filler row.
    _ONESIDE_MIN = 3
    merged = []
    for b in blocks:
        if (merged and b["type"] in ("del", "ins")
                and len(b["g"]) + len(b["o"]) < _ONESIDE_MIN):
            merged[-1]["g"] += b["g"]
            merged[-1]["o"] += b["o"]
            merged[-1]["type"] = "paired"
        else:
            merged.append({"g": list(b["g"]), "o": list(b["o"]), "type": b["type"]})
    blocks = merged

    # Slice the ORIGINAL text for a block — leading whitespace (back to the prior
    # word) is included so every space/newline lands in exactly one block, and
    # the slice depends only on the block's own words (so reordering is safe).
    def g_slice(idxs):
        a, b = idxs[0], idxs[-1]
        return ground_truth[(gspan[a - 1][1] if a > 0 else 0):gspan[b][1]]

    def o_slice(idxs):
        a, b = idxs[0], idxs[-1]
        return ocr_text[(ospan[a - 1][1] if a > 0 else 0):ospan[b][1]]

    # Block-level reorder: pair a moved OCR-only block with a nearby GT-only
    # block of similar text, shown as one "moved" (blue) row. Score unaffected.
    skip, moved_to, used_del = set(), {}, set()
    for s, b in enumerate(blocks):
        if b["type"] != "ins":
            continue
        okey = _line_key(o_slice(b["o"]), ic, ip, isp, inl)
        if not okey:
            continue
        best, best_sim = None, REORDER_BLOCK_SIM
        for d in range(max(0, s - REORDER_BLOCK_WINDOW), min(len(blocks), s + REORDER_BLOCK_WINDOW + 1)):
            db = blocks[d]
            if db["type"] != "del" or d in used_del:
                continue
            sim = _line_sim(_line_key(g_slice(db["g"]), ic, ip, isp, inl), okey)
            if sim >= best_sim:
                best_sim, best = sim, d
        if best is not None:
            used_del.add(best)
            skip.add(best)
            moved_to[s] = best

    rows = []
    k, N = 0, len(blocks)
    while k < N:
        if k in skip:
            k += 1
            continue  # ground-truth block was pulled to a 'moved' row elsewhere
        b = blocks[k]
        if k in moved_to:
            d = blocks[moved_to[k]]
            r = render(g_slice(d["g"]), o_slice(b["o"]))
            if r:
                r["moved"] = True
                rows.append(r)
            k += 1
            continue
        # A local deletion immediately next to an insertion is a substitution —
        # render the pair side-by-side instead of as two stacked filler rows.
        nxt = k + 1
        if nxt < N and nxt not in skip and nxt not in moved_to:
            b2 = blocks[nxt]
            if b["type"] == "del" and b2["type"] == "ins":
                r = render(g_slice(b["g"]), o_slice(b2["o"]))
                if r:
                    rows.append(r)
                k += 2
                continue
            if b["type"] == "ins" and b2["type"] == "del":
                r = render(g_slice(b2["g"]), o_slice(b["o"]))
                if r:
                    rows.append(r)
                k += 2
                continue
        lt = g_slice(b["g"]) if b["g"] else None
        rt = o_slice(b["o"]) if b["o"] else None
        r = render(lt, rt)
        if r:
            rows.append(r)
        k += 1

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
