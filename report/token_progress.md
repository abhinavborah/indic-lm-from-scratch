# Token Progress — rough, post-purity-filter, post-dedup

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

_Last updated: 2026-08-17 04:17:17_

- **Real tokens (kept):** 480,537,862 / 500,000,000 target (96.1%)
  - Manual (ocr + scrape): 49,524,301 (10.3%)
  - Downloaded (sangraha): 431,013,561 (89.7%)
    - of which from `unverified` (automated perplexity-filtered, **not** human-verified -- lower-confidence tier): 266
  - Manual quota (>=20% of 500,000,000 = 100,000,000): NOT MET (49,524,301 so far)

- **Filtering:** 1,342,962 segments scanned across 102 files.
  - Dropped for script impurity / fully word-stripped (clean_text.py -- NFC normalize, strip embedded Latin words, drop non-Devanagari-majority lines): 108 segments, 37,482 rough tokens
  - Dropped as exact duplicates: 110,922 segments, 32,836,115 rough tokens

## Assamese

_Last updated: 2026-08-17 04:20:34_

- **Real tokens (kept):** 153,477,463 / 500,000,000 target (30.7%)
  - Manual (ocr + scrape): 13,355,380 (8.7%)
  - Downloaded (sangraha): 140,122,083 (91.3%)
    - of which from `unverified` (automated perplexity-filtered, **not** human-verified -- lower-confidence tier): 7,948,050
  - Manual quota (>=20% of 500,000,000 = 100,000,000): NOT MET (13,355,380 so far)

- **Filtering:** 380,033 segments scanned across 65 files.
  - Dropped for script impurity / fully word-stripped (text_clean.py -- strip embedded Latin words, drop non-Bengali-Assamese-block-majority lines): 15,678 segments, 9,833,104 rough tokens
  - Dropped as suspected Bengali-not-Assamese (ৰ/ৱ heuristic, scrape+sangraha only -- see caveat below): 66 segments, 17,505 rough tokens
  - Dropped as exact duplicates: 10,306 segments, 4,591,268 rough tokens

**Note on the ৰ/ৱ heuristic:** this is a frequency heuristic, not a real language-ID classifier -- two letters' frequency is a weak signal on its own. A lightweight tool like `langid.py`, fastText `lid.176`, or `cld3` would do a properly calibrated job; not added now, just flagging it as a future option. It's skipped entirely for the ocr/ source since the OCR pipeline's Bengali-to-Assamese correction pass already force-converts every Bengali "র" to "ৰ", which would make this check trivially pass on anything.
