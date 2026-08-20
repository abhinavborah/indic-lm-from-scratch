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

_Last updated: 2026-08-20 14:17:48_

- **Real tokens (kept):** 566,315,453 / 500,000,000 target (113.3%)
  - Manual (ocr + scrape): 135,301,892 (23.9%)
  - Downloaded (sangraha): 431,013,561 (76.1%)
    - of which from `unverified` (automated perplexity-filtered, **not** human-verified -- lower-confidence tier): 266
  - Manual quota (>=20% of 500,000,000 = 100,000,000): MET (135,301,892 so far)

- **Filtering:** 1,427,149 segments scanned across 5,993 files.
  - Dropped for script impurity / fully word-stripped (clean_text.py -- NFC normalize, strip embedded Latin words, drop non-Devanagari-majority lines): 108 segments, 37,482 rough tokens
  - Dropped as exact duplicates: 111,301 segments, 33,133,609 rough tokens

## Assamese

_Last updated: 2026-08-20 14:10:50_

- **Real tokens (kept):** 320,478,882 / 500,000,000 target (64.1%) -- rough whitespace count, see spec-corrected estimate below
  - Manual (ocr + scrape): 74,452,715 (23.2%)
  - Downloaded (sangraha): 246,026,167 (76.8%)
    - of which from `unverified` (automated perplexity-filtered, **not** human-verified -- lower-confidence tier): 7,948,008
  - Manual quota (>=20% of 500,000,000 = 100,000,000): NOT MET (74,452,715 so far)

- **Spec-corrected estimate (real BPE tokens, per line 121's "after tokenization"):** measured fertility 2.3690 real tokens per rough word (5,000-vocab SentencePiece BPE, own corpus sample, held-out eval -- see constant comment above for method). Applying it:
  - Total: ~759,214,471 / 500,000,000 target (151.8%)
  - Manual: ~176,378,482 / 100,000,000 floor (176.4%)
  - This is a snapshot, not final -- vocab size is chosen (5,000) but the real tokenizer isn't trained yet, this fertility is still from a proxy sample, not the final split corpus. Re-measure once the real tokenizer trains.

- **Filtering:** 3,951,952 segments scanned across 10,345 files.
  - Dropped for script impurity / fully word-stripped (text_clean.py -- strip embedded Latin words, drop non-Bengali-Assamese-block-majority lines): 23,685 segments, 9,975,282 rough tokens
  - Dropped as suspected Bengali-not-Assamese (ৰ/ৱ heuristic, scrape+sangraha only -- see caveat below): 10,519 segments, 182,230 rough tokens
  - Dropped as exact duplicates: 130,593 segments, 15,799,841 rough tokens

**Note on the ৰ/ৱ heuristic:** this is a frequency heuristic, not a real language-ID classifier -- two letters' frequency is a weak signal on its own. A lightweight tool like `langid.py`, fastText `lid.176`, or `cld3` would do a properly calibrated job; not added now, just flagging it as a future option. It's skipped entirely for the ocr/ source since the OCR pipeline's Bengali-to-Assamese correction pass already force-converts every Bengali "র" to "ৰ", which would make this check trivially pass on anything.
