# Hindi Data Sources (hindi-data teammate)

Per-source list, distinct from the per-item `COLLECTION_LOG.md`. Orchestrator
folds this into the root-level `SOURCES.md` periodically — this file is not
edited by anyone else.

**Raw vs clean split:** every script writes the original unfiltered extracted
text to `raw/ocr/`/`raw/scrape/` and `clean_text()`'s output to the parallel
`clean/ocr/`/`clean/scrape/` tree (same filenames). **Caveat:** items
collected before this split was added (early NCERT OCR batches) only have
the cleaned version sitting in `raw/ocr/` — the true raw text for those was
never saved and is not recoverable. Forward-only fix, not retroactive.

## OCR (manual collection)

- **NCERT Hindi-medium textbooks, Class VI-XII** (`ncert.nic.in/textbook/pdf/`)
  — status: **in progress**, batched (200 items or 1hr per batch, whichever
  first), full catalog is 1617 chapter/prelim codes across 153 books,
  checkpointed in `.state.json`. Catalog extracted from `textbook.php`'s
  client-side book-picker lookup table (see `scripts/ncert_catalog.py`), not
  scraped page-by-page. Roughly 30-40% of catalog codes 404 — old-curriculum
  books retired from the server as NCERT rolls out its revised NCF textbooks
  class-by-class (confirmed: old Class VI books like Bal Ram Katha/Vasant are
  gone, replaced by Curiosity/Ganita Prakash; Class VII-XII old books still
  mostly live). Handled gracefully (marked `no_pdf`, skipped) — not a
  pipeline bug. Extraction: pdftotext first, Devanagari-density check,
  Tesseract `hin` OCR fallback. Downloads run 6-way parallel with a 45-minute
  per-file retry budget (ncert.nic.in resets connections mid-transfer often).

## Scrape (manual collection)

- **Hindi Wikipedia** (`hi.wikipedia.org`) — status: **done, dump-based
  extraction, complete**. Course clarification: a dump counts as manual
  collection as long as *we* write the preprocessing (extraction + cleaning).
  Uses `dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles-
  multistream.xml.bz2` + its byte-offset index, with a hand-rolled,
  WikiExtractor-style wikitext stripper (`scripts/wiki_dump_extract_parallel.py`)
  rather than live-scraping the MediaWiki API. Multistream format lets
  independent worker processes seek+decompress disjoint byte ranges in
  parallel — full 3271-chunk corpus extracts in ~20s. Final: 3271/3271 chunks,
  ~48.1M words (1 malformed chunk skipped, ~100 pages, non-fatal XML parse
  error in that one stream).
  (History: an earlier single-threaded script against the plain sequential
  dump, `scripts/wiki_dump_extract.py`, hit multi-hour runtimes since that
  dump format can't be split for parallel decompression. A smoke test of the
  new parallel script briefly ran concurrently with the old script before it
  was retired, duplicating ~2.68M words into the shared output via two
  uncoordinated state namespaces — caught before finalizing, output wiped and
  rebuilt clean in one full parallel run. `wiki_dump_extract.py` is retained
  for reference but superseded; don't run it again against this output.
  Earlier still, a live-API scraping approach was tried first and worked but
  hit intermittent connection-level fetch failures — same general network
  flakiness seen against ncert.nic.in, not Wikipedia rate-limiting; moot now.)
- **Hindi news sites** — 10 candidates found and verified reachable via their
  sitemap.xml/news-sitemap.xml (`scripts/scraper.py`, one concurrent worker
  thread per source), status **found, not yet scraped at scale** (only a
  smoke-test batch run so far):
  | Source | Sitemap confirmed | Scraped so far |
  |---|---|---|
  | jagran | yes (200) | smoke-tested |
  | amarujala | yes (200) | smoke-tested |
  | bbc_hindi | yes (200) | not yet |
  | livehindustan | yes (200, `news-sitemap.xml`) | not yet |
  | abplive | yes (200, `news-sitemap.xml`) | not yet |
  | prabhatkhabar | yes (200, `sitemap-index.xml`) | not yet |
  | indiatv | yes (200) | not yet |
  | patrika | yes (200) | not yet |
  | zeenews | yes (200, `/hindi/sitemap.xml`) | not yet |
  | aajtak | yes (301 → `/rssfeeds/news-sitemap`, 200) | not yet |
