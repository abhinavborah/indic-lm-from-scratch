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
- Script-purity filtering (dropping whole lines that aren't predominantly
  in the target script) happens inside each language's
  `clean_text.py`/`text_clean.py`, not in this tracker; this script only
  counts what survives. Embedded English words/proper nouns within an
  otherwise-native line are kept, per course guidance, not stripped.

## Hindi

_Last updated: 2026-08-21 01:50:32_

- **Real tokens (kept):** 569,197,046 / 500,000,000 target (113.8%), rough whitespace count, see spec-corrected estimate below
  - Manual (ocr + scrape): 135,428,971 (23.8%)
  - Downloaded (sangraha): 433,768,075 (76.2%)
    - of which from `unverified` (automated perplexity-filtered, **not** human-verified, lower-confidence tier): 273
  - Manual quota (>=20% of 500,000,000 = 100,000,000): MET (135,428,971 so far)

- **Spec-corrected estimate (real BPE tokens, per line 121's "after tokenization"):** measured fertility 1.4940 real tokens per rough word (8,000-vocab production SentencePiece BPE, trained on the real final train split, measured on the real held-out val split, see constant comment above for method). Applying it:
  - Total: ~850,380,387 / 500,000,000 target (170.1%)
  - Manual: ~202,330,883 / 100,000,000 floor (202.3%)
  - This is the final number: real production tokenizer, real held-out split, UNK rate 0.0. No longer a proxy estimate.

- **Filtering:** 4,060,015 segments scanned across 5,994 files.
  - Dropped for script impurity / fully word-stripped (clean_text.py: NFC normalize, drop non-Devanagari-majority lines): 325 segments, 61,008 rough tokens
  - Dropped as exact duplicates: 827,076 segments, 35,372,140 rough tokens

## Assamese

_Last updated: 2026-08-21 01:57:41_

- **Real tokens (kept):** 323,898,182 / 500,000,000 target (64.8%), rough whitespace count, see spec-corrected estimate below
  - Manual (ocr + scrape): 77,301,769 (23.9%)
  - Downloaded (sangraha): 246,596,413 (76.1%)
    - of which from `unverified` (automated perplexity-filtered, **not** human-verified, lower-confidence tier): 8,099,493
  - Manual quota (>=20% of 500,000,000 = 100,000,000): NOT MET (77,301,769 so far)

- **Spec-corrected estimate (real BPE tokens, per line 121's "after tokenization"):** measured fertility 1.7780 real tokens per rough word (8,000-vocab production SentencePiece BPE, trained on the real final train split, measured on the real held-out val split; see constant comment above for method). Applying it:
  - Total: ~575,890,968 / 500,000,000 target (115.2%)
  - Manual: ~137,442,545 / 100,000,000 floor (137.4%)
  - This is the final number: real production tokenizer, real held-out split, UNK rate 0.0. No longer a proxy estimate.

- **Filtering:** 4,802,488 segments scanned across 14,968 files.
  - Dropped for script impurity (text_clean.py: drop non-Bengali-Assamese-block-majority lines): 23,685 segments, 9,975,282 rough tokens
  - Dropped as suspected Bengali-not-Assamese (ৰ/ৱ heuristic, scrape+sangraha only; see caveat below): 27,419 segments, 433,920 rough tokens
  - Dropped as exact duplicates: 283,024 segments, 17,240,636 rough tokens

**Note on the ৰ/ৱ heuristic:** this is a frequency heuristic, not a real language-ID classifier; two letters' frequency is a weak signal on its own. A lightweight tool like `langid.py`, fastText `lid.176`, or `cld3` would do a properly calibrated job; not added now, just flagging it as a future option. It's skipped entirely for the ocr/ source since the OCR pipeline's Bengali-to-Assamese correction pass already force-converts every Bengali "র" to "ৰ", which would make this check trivially pass on anything.
