# Hindi Data Sources

Per-source list, distinct from the per-item `COLLECTION_LOG.md`. This file
is folded into the root-level `report/SOURCES.md`.

**Raw vs clean split:** every script writes the original unfiltered extracted
text to `raw/ocr/` / `raw/scrape/` and `clean_text()`'s output to the
parallel `clean/ocr/` / `clean/scrape/` tree (same filenames). Caveat: items
collected before this split was added (early NCERT OCR batches) only have
the cleaned version sitting in `raw/ocr/`. The true raw text for those was
never saved and is not recoverable. This is a forward-only fix, not
retroactive.

## OCR (manual collection)

- **NCERT Hindi-medium textbooks, Class VI-XII** (`ncert.nic.in/textbook/pdf/`). Status: in progress, batched (200 items or 1 hour per batch, whichever comes first). Full catalog is 1617 chapter/prelim codes across 153 books, checkpointed in `.state.json`. Catalog extracted from `textbook.php`'s client-side book-picker lookup table (see `scripts/ncert_catalog.py`), not scraped page by page. Roughly 30-40% of catalog codes 404, since old-curriculum books are retired from the server as NCERT rolls out its revised NCF textbooks class by class (confirmed: old Class VI books such as Bal Ram Katha and Vasant are gone, replaced by Curiosity and Ganita Prakash; Class VII-XII old books are still mostly live). This is handled gracefully (marked `no_pdf`, skipped), not a pipeline bug. Extraction: pdftotext first, Devanagari-density check, Tesseract `hin` OCR fallback. Downloads run 6-way parallel with a 45-minute per-file retry budget (ncert.nic.in resets connections mid-transfer often). 388/1617 codes resolved so far.

## Scrape (manual collection)

- **Hindi Wikipedia** (`hi.wikipedia.org`). Status: done, dump-based extraction, complete. A dump counts as manual collection as long as the preprocessing (extraction and cleaning) is done in-house. Uses `dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles-multistream.xml.bz2` and its byte-offset index, with a hand-rolled, WikiExtractor-style wikitext stripper (`scripts/wiki_dump_extract_parallel.py`) rather than live-scraping the MediaWiki API. The multistream format lets independent worker processes seek and decompress disjoint byte ranges in parallel: the full 3271-chunk corpus extracts in approximately 20 seconds. Final: 3271/3271 chunks, approximately 48.1M words (1 malformed chunk skipped, approximately 100 pages, non-fatal XML parse error in that one stream).

  History: an earlier single-threaded script against the plain sequential dump, `scripts/wiki_dump_extract.py`, hit multi-hour runtimes since that dump format cannot be split for parallel decompression. A smoke test of the new parallel script briefly ran concurrently with the old script before it was retired, duplicating approximately 2.68M words into the shared output via two uncoordinated state namespaces. This was caught before finalizing; the output was wiped and rebuilt clean in one full parallel run. `wiki_dump_extract.py` is retained for reference but superseded, and should not be run again against this output. Earlier still, a live-API scraping approach was tried first and worked but hit intermittent connection-level fetch failures, similar to the general network flakiness seen against ncert.nic.in, not Wikipedia rate-limiting. This is moot now.

- **Hindi news sites.** 10 candidates found and verified reachable via their sitemap.xml or news-sitemap.xml (`scripts/scraper.py`, one concurrent worker thread per source). Two real-scale batches run 2026-08-17 (`--sources abplive indiatv patrika zeenews prabhatkhabar aajtak`, 300 articles/3600s cap per source):

  | Source | Sitemap confirmed | Status |
  |---|---|---|
  | jagran | yes (200) | smoke-tested only, not in active rotation |
  | amarujala | yes (200) | smoke-tested only, not in active rotation |
  | bbc_hindi | yes (200) | not yet run |
  | livehindustan | yes (200, `news-sitemap.xml`) | not yet run |
  | abplive | yes (200, `news-sitemap.xml`) | working, steady yield each batch |
  | indiatv | yes (200) | working, steady yield each batch |
  | patrika | yes (200) | working, hits the 300-article cap each batch |
  | zeenews | yes (200, `/hindi/sitemap.xml`) | working, hits the 300-article cap each batch |
  | prabhatkhabar | yes (200, `sitemap-index.xml`) | blocked. The top-level `sitemap-index.xml` fetches fine, but every one of its approximately 50 child sitemaps fails to fetch (curl failure, all retries). This looks like CDN or IP-level blocking on the actual content sitemaps. 0 articles across 2 consecutive batches. Not a pipeline bug: the script correctly does not mark it drained, and retries automatically each run in case the block lifts. |
  | aajtak | yes (301 to `/rssfeeds/news-sitemap`, 200) | blocked. The top-level sitemap fetch itself fails all 3 discovery retries. 0 articles across 2 consecutive batches. Same self-healing retry behavior as prabhatkhabar. |
