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

_Last updated: 2026-08-17 23:37:31_

- **Real tokens (kept):** 492,318,773 / 500,000,000 target (98.5%)
  - Manual (ocr + scrape): 61,305,212 (12.5%)
  - Downloaded (sangraha): 431,013,561 (87.5%)
    - of which from `unverified` (automated perplexity-filtered, **not** human-verified -- lower-confidence tier): 266
  - Manual quota (>=20% of 500,000,000 = 100,000,000): NOT MET (61,305,212 so far)

- **Filtering:** 1,365,711 segments scanned across 123 files.
  - Dropped for script impurity / fully word-stripped (clean_text.py -- NFC normalize, strip embedded Latin words, drop non-Devanagari-majority lines): 108 segments, 37,482 rough tokens
  - Dropped as exact duplicates: 111,017 segments, 32,893,574 rough tokens

## Assamese

_Last updated: 2026-08-18 01:09:20_

- **Real tokens (kept):** 192,725,169 / 500,000,000 target (38.5%)
  - Manual (ocr + scrape): 14,573,069 (7.6%)
  - Downloaded (sangraha): 178,152,100 (92.4%)
    - of which from `unverified` (automated perplexity-filtered, **not** human-verified -- lower-confidence tier): 7,948,050
  - Manual quota (>=20% of 500,000,000 = 100,000,000): NOT MET (14,573,069 so far)

- **Filtering:** 1,998,481 segments scanned across 66 files.
  - Dropped for script impurity / fully word-stripped (text_clean.py -- strip embedded Latin words, drop non-Bengali-Assamese-block-majority lines): 18,035 segments, 9,880,999 rough tokens
  - Dropped as suspected Bengali-not-Assamese (ৰ/ৱ heuristic, scrape+sangraha only -- see caveat below): 9,778 segments, 175,734 rough tokens
  - Dropped as exact duplicates: 14,947 segments, 4,752,100 rough tokens

**Note on the ৰ/ৱ heuristic:** this is a frequency heuristic, not a real language-ID classifier -- two letters' frequency is a weak signal on its own. A lightweight tool like `langid.py`, fastText `lid.176`, or `cld3` would do a properly calibrated job; not added now, just flagging it as a future option. It's skipped entirely for the ocr/ source since the OCR pipeline's Bengali-to-Assamese correction pass already force-converts every Bengali "র" to "ৰ", which would make this check trivially pass on anything.
