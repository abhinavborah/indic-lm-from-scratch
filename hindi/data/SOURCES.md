# Hindi Data Sources

Per-source list, distinct from the per-item `COLLECTION_LOG.md`. This file
is folded into the root-level `report/SOURCES.md`.

**Raw vs clean split:** every script writes the original unfiltered extracted
text to `raw/<source>/` and `clean_text()`'s output to the parallel
`clean/<source>/` tree (same filenames). Caveat: items collected before this
split was added (early NCERT OCR batches) only have the cleaned version
sitting in `raw/ocr/`. The true raw text for those was never saved and is
not recoverable. This is a forward-only fix, not retroactive.

*Last updated: 2026-08-18, ~17:55.*

## OCR (manual collection)

- **NCERT Hindi-medium textbooks, Class VI-XII** (`ncert.nic.in/textbook/pdf/`). Status: in progress, batched (200 items or 1 hour per batch, whichever comes first). Full catalog is 1617 chapter/prelim codes across 153 books, checkpointed in `.state.json`. Catalog extracted from `textbook.php`'s client-side book-picker lookup table (see `scripts/ncert_catalog.py`), not scraped page by page. Roughly 30 to 40% of catalog codes 404, since old-curriculum books are retired from the server as NCERT rolls out its revised NCF textbooks class by class (confirmed: old Class VI books such as Bal Ram Katha and Vasant are gone, replaced by Curiosity and Ganita Prakash; Class VII to XII old books are still mostly live). This is handled gracefully (marked `no_pdf`, skipped), not a pipeline bug. Extraction: pdftotext first, Devanagari-density check, Tesseract `hin` OCR fallback. Downloads run 6-way parallel with a 45-minute per-file retry budget (ncert.nic.in resets connections mid-transfer often). About 1200 codes attempted, around 800 extracted so far.
- **Digital Library of India, Hindi books** (archive.org, `collection:digitallibraryindia AND language:hin`). 53,172 real Hindi-language books, novels, essays, drama, history, and biography, mostly early to mid 1900s. archive.org already OCRs everything on upload and publishes a plain-text `_djvu.txt` file per item, so no PDF download or local OCR is needed here, the script just fetches that file directly. Added this session, same approach and same script (adapted) as the Assamese `dli_books_download.py` that found this pattern first. Smoke test yielded 317,890 words from 5 items. Promoted to production immediately: about 20 items done, about 2.5M words, within minutes of starting. Given Hindi's manual floor is the one gap left, this is expected to close most or all of it on its own. Public-domain status is likely for most titles but not verified per item.

## Scrape (manual collection)

- **Hindi Wikipedia** (`hi.wikipedia.org`). Status: done, dump-based extraction, complete. A dump counts as manual collection as long as the preprocessing (extraction and cleaning) is done in-house. Uses `dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles-multistream.xml.bz2` and its byte-offset index, with a hand-rolled, WikiExtractor-style wikitext stripper (`scripts/wiki_dump_extract_parallel.py`) rather than live-scraping the MediaWiki API. The multistream format lets independent worker processes seek and decompress disjoint byte ranges in parallel: the full 3271-chunk corpus extracts in about 20 seconds. Final: 3271/3271 chunks, about 48.1M words (1 malformed chunk skipped, about 100 pages, non-fatal XML parse error in that one stream).

  History: an earlier single-threaded script against the plain sequential dump, `scripts/wiki_dump_extract.py`, hit multi-hour runtimes since that dump format cannot be split for parallel decompression. A smoke test of the new parallel script briefly ran concurrently with the old script before it was retired, duplicating about 2.68M words into the shared output via two uncoordinated state namespaces. This was caught before finalizing; the output was wiped and rebuilt clean in one full parallel run. `wiki_dump_extract.py` is retained for reference but superseded, and should not be run again against this output. Earlier still, a live-API scraping approach was tried first and worked but hit intermittent connection-level fetch failures, similar to the general network flakiness seen against ncert.nic.in, not Wikipedia rate-limiting. This is moot now.

- **Hindi news sites.** 10 candidates found and verified reachable via their sitemap.xml or news-sitemap.xml (`scripts/scraper.py`, one concurrent worker thread per source). Currently running as an open-ended loop (`--sources abplive patrika zeenews`, no article cap, capped only by wall clock per invocation):

  | Source | Sitemap confirmed | Status |
  |---|---|---|
  | jagran | yes (200) | smoke-tested only, not in active rotation |
  | amarujala | yes (200) | smoke-tested only, not in active rotation |
  | bbc_hindi | yes (200) | not yet run |
  | livehindustan | yes (200, `news-sitemap.xml`) | not yet run |
  | abplive | yes (200, `news-sitemap.xml`) | its own catalog fully drained this session (2749 articles, about 6.4M words total ever), dropped from active rotation |
  | indiatv | yes (200) | fully drained in an earlier session, dropped from active rotation |
  | patrika | yes (200) | working, steady yield, about 3.5M words total ever so far |
  | zeenews | yes (200, `/hindi/sitemap.xml`) | working, steady yield, about 2.7M words total ever so far, largest remaining catalog (238,571 articles) |
  | prabhatkhabar | yes (200, `sitemap-index.xml`) | blocked. The top-level `sitemap-index.xml` fetches fine, but every one of its about 50 child sitemaps fails to fetch (curl failure, all retries). This looks like CDN or IP-level blocking on the actual content sitemaps. Not a pipeline bug: the script correctly does not mark it drained, and retries automatically each run in case the block lifts. |
  | aajtak | yes (301 to `/rssfeeds/news-sitemap`, 200) | blocked. The top-level sitemap fetch itself fails all discovery retries. Same self-healing retry behavior as prabhatkhabar. |

- **Vikaspedia** (`hi.vikaspedia.in`). Government of India (C-DAC/MeitY) multilingual public-policy portal. Status: in progress, running as an open-ended loop. Article body ships inline as JSON in a `__NEXT_DATA__` script tag even though the page shell has no visible body text (client-rendered). Discovery via `sitemap.xml` (6829 URLs). 2235 URLs attempted so far, 1632 real articles extracted (rest are folder/category pages or stale redirects, correctly skipped). A slug-collision bug found and fixed this session: non-Latin URL slugs used to collapse to the same filename per category and language, silently overwriting almost everything; fixed by appending a hash of the full URL to the filename.

## Not pursued or low priority

- **shabdkosh.com, rajbhasha.gov.in.** Candidate Hindi dictionary sources, never actually checked. Flagged as unchecked, not a decision either way.
- **PIB archive** (`archive.pib.gov.in`). Real, reachable, no antibot block (unlike the main `pib.gov.in`, which is antibot-blocked). Not built against: Hindi is already effectively at target, so the marginal value is low.
- **Doordarshan News** (`ddnews.gov.in`). Real Hindi article site (WordPress, `/category/top-stories/`), confirmed reachable. Not built against for the same reason as PIB archive: Hindi does not need more manual tokens right now.
- **newsonair.gov.in (Akashvani).** Checked for Hindi specifically: no single unified Hindi bulletin page exists (unlike Assamese's `/bulletins-city/assamese/`). Hindi bulletins are split across many per-regional-station pages (Ranchi, Jaipur, Chandigarh, and others) instead. Not chased given Hindi's status.
