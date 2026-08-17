# Corpus Sources

Combined view across both languages. Per-language detail (source lines,
status, notes) lives in `hindi/data/SOURCES.md` and `assamese/data/SOURCES.md`.
Per-document detail (counts, timestamps) lives in each language's
`COLLECTION_LOG.md`. This file tracks which sites and collections were
tapped, not per-document detail.

## Hindi (Model H)

| Source | Type | Status |
|---|---|---|
| NCERT Hindi-medium textbooks (ncert.nic.in), 1617 chapter/prelim codes across 153 books | Manual, OCR/pdftotext + Tesseract `hin` fallback | In progress, batched (200 items per hour). Approximately 30-40% of codes 404 (retired old-curriculum books, handled gracefully). 388/1617 codes resolved so far. |
| hi.wikipedia.org | Manual, dump-based extraction (own preprocessing) | Complete. 3271/3271 chunks processed via multistream parallel extraction, approximately 48.1M words. |
| Hindi news sites (abplive, indiatv, patrika, zeenews in active rotation; jagran, amarujala smoke-tested only; prabhatkhabar, aajtak confirmed blocked at the CDN/IP level) | Manual, scrape (concurrent, one thread per source) | In progress, real-scale batches running. |
| shabdkosh.com, rajbhasha.gov.in | Manual, dictionary scrape | Candidate, not yet added. |
| `ai4bharat/sangraha` (HF datasets), `verified` and `unverified`, `hin` | Public, downloaded | Complete. 431M tokens. |

## Assamese (Model L)

| Source | Type | Status |
|---|---|---|
| `docs/assamese/books/`, SCERT/NCERT Class IX-X textbooks, 29 PDFs | Manual, OCR (Tesseract `ben` + ৰ/ৱ correction) | Complete. 4627/4627 pages. |
| niyomiyabarta.com | Manual, scrape (concurrent) | In progress, batched (2000 articles or 1 hour per run). 10,458 articles scraped so far, approximately 39,542 remaining. |
| asomiyapratidin.in | Manual, scrape (concurrent) | Drained for its current sitemap window (0 new). Dual raw and clean writes resume when new articles appear. |
| as.wikipedia.org | Manual, dump-based extraction (own preprocessing) | Complete. 25,110 articles, raw approximately 10.79M words, clean approximately 10.30M words, 0 suspect-language exclusions. |
| as.wikisource.org dump | Manual, dump-based extraction (own preprocessing) | Not yet run. First attempt was interrupted mid-download, retrying manually. |
| xobdo.org | Manual, dictionary scrape | Complete. 47,972 dictionary entries. |
| `ai4bharat/sangraha` (HF datasets), `verified` and `unverified`, `asm` | Public, downloaded | Complete. 140M tokens. Ceiling is well below the 500M target, expected under the spec's allowed shortfall for lower-resource languages. |
| `MWirelabs/assamese-monolingual-corpus` (HF) | Public, downloaded | Complete. 38.03M real, dedup-aware tokens kept. Downloaded manually and converted 2026-08-18. |

## Real token totals (from `report/token_progress.md`, last rescan)

- Hindi: 492.32M/500M total (98.5%). Manual 61.31M (needs a 100M floor, gap approximately 38.7M).
- Assamese: 192.73M/500M total (38.5%). Manual 14.57M (needs a 100M floor as currently tracked; this floor has been found mathematically unreachable with current sources, and whether it should be re-based to a percentage of Assamese's realistic total instead of a flat 100M is still an open question).

*Last updated: 2026-08-18.*
