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

_Last updated: 2026-08-19 12:41:47_

- **Real tokens (kept):** 565,974,997 / 500,000,000 target (113.2%)
  - Manual (ocr + scrape): 134,961,436 (23.8%)
  - Downloaded (sangraha): 431,013,561 (76.2%)
    - of which from `unverified` (automated perplexity-filtered, **not** human-verified -- lower-confidence tier): 266
  - Manual quota (>=20% of 500,000,000 = 100,000,000): MET (134,961,436 so far)

- **Filtering:** 1,425,976 segments scanned across 5,993 files.
  - Dropped for script impurity / fully word-stripped (clean_text.py -- NFC normalize, strip embedded Latin words, drop non-Devanagari-majority lines): 108 segments, 37,482 rough tokens
  - Dropped as exact duplicates: 111,301 segments, 33,133,609 rough tokens

## Assamese

_Last updated: 2026-08-19 16:43:46_

- **Real tokens (kept):** 286,543,906 / 500,000,000 target (57.3%) -- rough whitespace count, see spec-corrected estimate below
  - Manual (ocr + scrape): 40,385,896 (14.1%)
  - Downloaded (sangraha): 246,158,010 (85.9%)
    - of which from `unverified` (automated perplexity-filtered, **not** human-verified -- lower-confidence tier): 7,948,050
  - Manual quota (>=20% of 500,000,000 = 100,000,000): NOT MET (40,385,896 so far)

- **Spec-corrected estimate (real BPE tokens, per line 121's "after tokenization"):** measured fertility 1.4398 real tokens per rough word (32k-vocab SentencePiece BPE, own corpus sample, held-out eval -- see constant comment above for method). Applying it:
  - Total: ~412,565,916 / 500,000,000 target (82.5%)
  - Manual: ~58,147,613 / 100,000,000 floor (58.1%)
  - This is a snapshot, not final -- the real tokenizer isn't trained yet and vocab size isn't chosen (spec's own fertility/UNK-rate sweep is still open). Re-measure once it is.

- **Filtering:** 3,793,248 segments scanned across 10,341 files.
  - Dropped for script impurity / fully word-stripped (text_clean.py -- strip embedded Latin words, drop non-Bengali-Assamese-block-majority lines): 23,685 segments, 9,975,282 rough tokens
  - Dropped as suspected Bengali-not-Assamese (ৰ/ৱ heuristic, scrape+sangraha only -- see caveat below): 10,519 segments, 182,230 rough tokens
  - Dropped as exact duplicates: 116,343 segments, 12,788,075 rough tokens

**Note on the ৰ/ৱ heuristic:** this is a frequency heuristic, not a real language-ID classifier -- two letters' frequency is a weak signal on its own. A lightweight tool like `langid.py`, fastText `lid.176`, or `cld3` would do a properly calibrated job; not added now, just flagging it as a future option. It's skipped entirely for the ocr/ source since the OCR pipeline's Bengali-to-Assamese correction pass already force-converts every Bengali "র" to "ৰ", which would make this check trivially pass on anything.
