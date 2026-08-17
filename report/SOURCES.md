# Corpus Sources

Combined view across both languages. Owned by the orchestrator, folded in
from each teammate's own `hindi/data/SOURCES.md` / `assamese/data/SOURCES.md`
when checking in on progress. Per-item detail (counts, timestamps) lives in
each teammate's `COLLECTION_LOG.md` — this file tracks *which* sites/
collections were tapped, not per-document detail.

## Hindi (Model H)

| Source | Type | Status |
|---|---|---|
| NCERT Hindi-medium textbooks (ncert.nic.in), 1617 chapter/prelim codes across 153 books | Manual — OCR/pdftotext + Tesseract `hin` fallback | In progress, batched (200 items/1hr). ~30-40% of codes 404 (retired old-curriculum books, handled gracefully) |
| hi.wikipedia.org | Manual — dump-based extraction (own preprocessing) | In progress, batched (1hr/3600s per run). Cumulative ~160K pages / ~36.2M words as of last report. Single-threaded — multistream+index parallelization fix relayed 2026-08-17, not yet applied |
| Hindi news sites (10 candidates found: jagran, amarujala, bbc_hindi, livehindustan, abplive, prabhatkhabar, indiatv, patrika, zeenews, aajtak) | Manual — scrape (concurrent, one thread/source) | Found + sitemap-verified; smoke-tested only, first real-scale batch launched 2026-08-17 (in progress) |
| shabdkosh.com, rajbhasha.gov.in | Manual — dictionary scrape | Candidate, relayed 2026-08-17, not yet added |
| `ai4bharat/sangraha` (HF datasets), `verified`+`unverified`, `hin` | Public/downloaded | Done — 431M tokens, near its ~450M cap |

## Assamese (Model L)

| Source | Type | Status |
|---|---|---|
| `docs/assamese/books/` — SCERT/NCERT Class IX-X textbooks, 29 PDFs | Manual — OCR (Tesseract `ben` + ৰ/ৱ correction) | Complete — 4627/4627 pages |
| niyomiyabarta.com | Manual — scrape (concurrent) | In progress, batched (2000 articles/1hr). 5,043 articles scraped so far, 44,957 remaining |
| asomiyapratidin.in | Manual — scrape (concurrent) | Drained for current sitemap window (0 new); dual raw+clean writes resume when new articles appear |
| as.wikipedia.org | Manual — dump-based extraction (own preprocessing) | Complete — 25,110 articles, raw ~10.79M words / clean ~10.30M words, 0 suspect-language exclusions |
| xobdo.org (lead), assamesedictionary.org (lower-confidence alternative) | Manual — dictionary scrape | Candidate, relayed 2026-08-17, not yet added |
| `ai4bharat/sangraha` (HF datasets), `verified`+`unverified`, `asm` | Public/downloaded | Done — 140M tokens (ceiling well below 500M target, expected per spec's allowed shortfall) |

## Real token totals (from `report/token_progress.md`, last rescan)

- Hindi: 459.95M/500M total (92.0%) — manual 28.94M (needs 100M floor, gap ~71M)
- Assamese: 152.27M/500M total (30.5%) — manual 12.15M (needs 100M floor as tracked; see CONTEXT.md's note that this may need re-basing to Assamese's realistic final total instead of a flat 100M)

*Last updated: 2026-08-16, mid Phase 1 data collection — orchestrator handoff pending, see `.planning/phase-1/handoff/orchestrator/`.*
