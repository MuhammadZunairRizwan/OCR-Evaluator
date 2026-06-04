# Test data — expected results

Upload the **ground_truth** file into the left box and the matching
**ocr_result** file into the right box, then click **Evaluate**.

| Pair | What it tests | WER | CER | Word Acc | Char Acc |
|------|----------------|-----|-----|----------|----------|
| **1** `1_ground_truth.txt` / `1_ocr_result.txt` | Character-level typos (quick→qulck, etc.) — note CER ≪ WER | 17.39% | 2.76% | 82.61% | 97.24% |
| **2** `2_ground_truth.txt` / `2_ocr_result.txt` | Mix of substitutions, a deletion ("the"), an insertion ("today") | 20.45% | 6.61% | 79.55% | 93.39% |
| **3** `3_perfect_ground_truth.txt` / `3_perfect_ocr_result.txt` | Identical text → baseline | 0.00% | 0.00% | 100% | 100% |
| **4** `4_case_punct_ground_truth.txt` / `4_case_punct_ocr_result.txt` | Differs only in case + punctuation | 106.67%* | 21.11% | 0% | 78.89% |
| **4 + toggles** (tick *Ignore case* **and** *Ignore punctuation*) | Same files, lenient scoring | 13.33% | 1.19% | 86.67% | 98.81% |

\* WER can exceed 100% — it's `(S + D + I) / N`, and insertions can push the
error count above the reference length. Word Accuracy is clamped to 0%. This is
expected, standard behaviour (same as ASR/Kaldi WER).

**Pair 4 is the key demo:** with the toggles off, every word is "wrong" (case +
punctuation differ). Tick **Ignore case** and **Ignore punctuation** and watch
the score jump to ~87% — proving the normalization options work and that the
green/red highlighting re-aligns live.

Tip: also toggle **Word ↔ Character** highlighting on Pair 1 to see how the same
errors look at each granularity.
