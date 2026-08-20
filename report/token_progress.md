# Token Progress: rough, post-purity-filter, post-dedup

Rough whitespace/regex tokenization, not the real BPE count (that comes once
the tokenizer is trained). Each language's section below is recomputed
independently by its own `token_tracker.py` (`hindi/data/scripts/`,
`assamese/data/scripts/`) via a full rescan of that language's
`data/raw/**` each run.

## Caveats

- Sangraha's `unverified` config is automated-perplexity-filtered, not
  human-verified like `verified` -- treat it as a lower-confidence tier, not
  equivalent-quality text. `synthetic` is never pulled at all
  (machine-translated/romanized text, excluded from collection entirely).
- Counts are rough (whitespace tokenization). Real fertility-based counts
  come from the trained tokenizer.
- Word-level foreign-word stripping (per course guidance: digits are fine,
  but embedded non-language words must go) and script-purity filtering both
  happen inside each language's `clean_text.py`/`text_clean.py`, not in this
  tracker -- this script only counts what survives.

## Hindi

_Last updated: 2026-08-20 21:01:31_

- **Real tokens (kept):** 566,315,453 / 500,000,000 target (113.3%)
  - Manual (ocr + scrape): 135,301,892 (23.9%)
  - Downloaded (sangraha): 431,013,561 (76.1%)
    - of which from `unverified` (automated perplexity-filtered, **not** human-verified, lower-confidence tier): 266
  - Manual quota (>=20% of 500,000,000 = 100,000,000): MET (135,301,892 so far)

- **Filtering:** 1,427,149 segments scanned across 5,993 files.
  - Dropped for script impurity / fully word-stripped (clean_text.py: NFC normalize, strip embedded Latin words, drop non-Devanagari-majority lines): 108 segments, 37,482 rough tokens
  - Dropped as exact duplicates: 111,301 segments, 33,133,609 rough tokens

## Assamese

_Last updated: 2026-08-20 21:01:57_

- **Real tokens (kept):** 324,063,401 / 500,000,000 target (64.8%), rough whitespace count, see spec-corrected estimate below
  - Manual (ocr + scrape): 78,037,234 (24.1%)
  - Downloaded (sangraha): 246,026,167 (75.9%)
    - of which from `unverified` (automated perplexity-filtered, **not** human-verified, lower-confidence tier): 7,948,008
  - Manual quota (>=20% of 500,000,000 = 100,000,000): NOT MET (78,037,234 so far)

- **Spec-corrected estimate (real BPE tokens, per line 121's "after tokenization"):** measured fertility 1.7437 real tokens per rough word (8,000-vocab production SentencePiece BPE, trained on the real final train split, measured on the real held-out val split; see constant comment above for method). Applying it:
  - Total: ~565,069,352 / 500,000,000 target (113.0%)
  - Manual: ~136,073,525 / 100,000,000 floor (136.1%)
  - This is the final number: real production tokenizer, real held-out split, UNK rate 0.0. No longer a proxy estimate.

- **Filtering:** 3,956,585 segments scanned across 14,968 files.
  - Dropped for script impurity / fully word-stripped (text_clean.py: strip embedded Latin words, drop non-Bengali-Assamese-block-majority lines): 23,685 segments, 9,975,282 rough tokens
  - Dropped as suspected Bengali-not-Assamese (ৰ/ৱ heuristic, scrape+sangraha only; see caveat below): 10,520 segments, 182,268 rough tokens
  - Dropped as exact duplicates: 130,601 segments, 15,848,931 rough tokens

**Note on the ৰ/ৱ heuristic:** this is a frequency heuristic, not a real language-ID classifier; two letters' frequency is a weak signal on its own. A lightweight tool like `langid.py`, fastText `lid.176`, or `cld3` would do a properly calibrated job; not added now, just flagging it as a future option. It's skipped entirely for the ocr/ source since the OCR pipeline's Bengali-to-Assamese correction pass already force-converts every Bengali "র" to "ৰ", which would make this check trivially pass on anything.
