# Hindi Data Sources

Per-source list, distinct from the per-item `COLLECTION_LOG.md`. This file
is folded into the root-level `report/SOURCES.md`.

**Raw vs clean split:** every script writes the original unfiltered extracted
text to `raw/<source>/` and `clean_text()`'s output to the parallel
`clean/<source>/` tree (same filenames). Caveat: items collected before this
split was added (early NCERT OCR batches) only have the cleaned version
sitting in `raw/ocr/`. The true raw text for those was never saved and is
not recoverable. This is a forward-only fix, not retroactive.

*Last updated: 2026-08-21. Corpus is frozen: final splits built, production
tokenizer trained. Token totals in `report/SOURCES.md` are from
`report/token_progress.md`'s latest rescan.*

## Manual collection, complete or stopped at the corpus freeze

- **NCERT Hindi-medium textbooks, Class VI-XII** (`ncert.nic.in/textbook/pdf/`). Full catalog is 1617 chapter/prelim codes across 153 books, checkpointed in `.state.json`. Catalog extracted from `textbook.php`'s client-side book-picker lookup table (see `scripts/ncert_catalog.py`), not scraped page by page. Roughly 30 to 40% of catalog codes 404, since old-curriculum books are retired from the server as NCERT rolls out its revised NCF textbooks class by class (confirmed: old Class VI books such as Bal Ram Katha and Vasant are gone, replaced by Curiosity and Ganita Prakash; Class VII to XII old books are still mostly live). Handled gracefully (marked `no_pdf`, skipped), not a pipeline bug. Extraction: pdftotext first, Devanagari-density check, Tesseract `hin` OCR fallback. Stopped at the corpus freeze: 113 items extracted, about 7.1M words.
- **Digital Library of India, Hindi books** (archive.org, `collection:digitallibraryindia AND language:hin`). 53,172 real Hindi-language books, novels, essays, drama, history, and biography, mostly early to mid 1900s. archive.org already OCRs everything on upload and publishes a plain-text `_djvu.txt` file per item, so no PDF download or local OCR is needed here, the script just fetches that file directly. Same approach and same script (adapted) as the Assamese `dli_books_download.py` that found this pattern first. Stopped at the corpus freeze: 434 items done, about 41.7M words. Public-domain status is likely for most titles but not verified per item.
- **Hindi Wikipedia** (`hi.wikipedia.org`). Dump-based extraction, complete. A dump counts as manual collection as long as the preprocessing (extraction and cleaning) is done in-house. Uses `dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles-multistream.xml.bz2` and its byte-offset index, with a hand-rolled, WikiExtractor-style wikitext stripper (`scripts/wiki_dump_extract_parallel.py`) rather than live-scraping the MediaWiki API. The multistream format lets independent worker processes seek and decompress disjoint byte ranges in parallel: the full 3271-chunk corpus extracts in about 20 seconds. Final: 3271/3271 chunks, about 72.2M words (1 malformed chunk skipped, about 100 pages, non-fatal XML parse error in that one stream).

  History: an earlier single-threaded script against the plain sequential dump, `scripts/wiki_dump_extract.py`, hit multi-hour runtimes since that dump format cannot be split for parallel decompression. A smoke test of the new parallel script briefly ran concurrently with the old script before it was retired, duplicating about 2.68M words into the shared output via two uncoordinated state namespaces. This was caught before finalizing; the output was wiped and rebuilt clean in one full parallel run. `wiki_dump_extract.py` is retained for reference but superseded, and should not be run again against this output. Earlier still, a live-API scraping approach was tried first and worked but hit intermittent connection-level fetch failures, similar to the general network flakiness seen against ncert.nic.in, not Wikipedia rate-limiting. This is moot now.

- **abplive.com.** Hindi news site, sitemap-driven scrape (`news-sitemap.xml`). Catalog fully drained: 2749 articles, about 6.7M words.
- **indiatv.in.** Hindi news site, sitemap-driven scrape. Fully drained: about 4.0M words.
- **patrika.com.** Hindi news site, sitemap-driven scrape. Stopped at the corpus freeze: about 10.6M words.
- **zeenews.india.com.** Hindi news site, sitemap-driven scrape (`/hindi/sitemap.xml`). Stopped at the corpus freeze: about 12.1M words. Largest remaining catalog at freeze time (238,571 articles), not exhausted.
- **livehindustan.com.** Hindi news site, sitemap-driven scrape (`news-sitemap.xml`). Stopped at the corpus freeze: about 0.8M words.
- **hi.vikaspedia.in.** Government of India (C-DAC/MeitY) multilingual public-policy portal. Article body ships inline as JSON in a `__NEXT_DATA__` script tag even though the page shell has no visible body text (client-rendered). Discovery via `sitemap.xml` (6829 URLs). Stopped at the corpus freeze: 5345 items extracted (2235+ URLs attempted; the rest are folder/category pages or stale redirects, correctly skipped), about 7.6M words. A slug-collision bug found and fixed this session: non-Latin URL slugs used to collapse to the same filename per category and language, silently overwriting almost everything; fixed by appending a hash of the full URL to the filename.

## Checked, smoke-tested only, or blocked (not in the frozen corpus)

- **jagran.com, amarujala.com.** Sitemaps confirmed reachable (200), smoke-tested only, never promoted to active rotation before the freeze.
- **bbc_hindi (bbc.com/hindi).** Sitemap confirmed reachable, never run.
- **prabhatkhabar.com.** The top-level `sitemap-index.xml` fetches fine, but every one of its about 50 child sitemaps fails to fetch (curl failure, all retries). Looks like CDN or IP-level blocking on the actual content sitemaps, not a pipeline bug.
- **aajtak.in.** The top-level sitemap fetch itself fails all discovery retries. Same blocked pattern as prabhatkhabar.
- **shabdkosh.com, rajbhasha.gov.in.** Candidate Hindi dictionary sources, never actually checked. Flagged as unchecked, not a decision either way.
- **PIB archive** (`archive.pib.gov.in`). Real, reachable, no antibot block (unlike the main `pib.gov.in`, which is antibot-blocked). Not built against: Hindi comfortably cleared its manual floor without it.
- **Doordarshan News** (`ddnews.gov.in`). Real Hindi article site (WordPress, `/category/top-stories/`), confirmed reachable. Not built against for the same reason as PIB archive.
- **newsonair.gov.in (Akashvani).** Checked for Hindi specifically: no single unified Hindi bulletin page exists (unlike Assamese's `/bulletins-city/assamese/`). Hindi bulletins are split across many per-regional-station pages (Ranchi, Jaipur, Chandigarh, and others) instead. Not chased given Hindi's status.
