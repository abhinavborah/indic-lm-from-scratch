# Corpus Sources

Combined view across both languages. Per-language detail (source lines,
status, notes) lives in `hindi/data/SOURCES.md` and `assamese/data/SOURCES.md`.
Per-document detail (counts, timestamps) lives in each language's
`COLLECTION_LOG.md`. This file tracks which sites and collections were
tapped, not per-document detail.

## Hindi (Model H)

| Source | Type | Status |
|---|---|---|
| NCERT Hindi-medium textbooks (ncert.nic.in), 1617 chapter/prelim codes across 153 books | Manual, OCR/pdftotext plus Tesseract `hin` fallback | In progress. 1212/1617 codes attempted, 789 extracted. About 30 to 40% of codes 404 (retired old-curriculum books, handled gracefully). |
| hi.wikipedia.org | Manual, dump-based extraction (own preprocessing) | Complete. 3271/3271 chunks, about 48.1M words. |
| Hindi news sites (patrika, zeenews active; abplive and indiatv fully drained; jagran, amarujala smoke-tested only; prabhatkhabar, aajtak blocked at the CDN/IP level) | Manual, scrape (open-ended loop) | In progress. |
| hi.vikaspedia.in | Manual, scrape (own JSON extraction, open-ended loop) | In progress. 2235 URLs attempted, 1632 real articles. Added this session, slug-collision bug found and fixed. |
| Digital Library of India, Hindi books (archive.org) | Manual, plain-text OCR derivative, no local OCR needed | In progress, open-ended loop. Added this session. 53,172 total items available (24 times the Assamese pool), same script as the Assamese version. About 20 items done, about 2.5M words, within minutes of starting. Expected to close most or all of Hindi's remaining manual gap on its own. |
| shabdkosh.com, rajbhasha.gov.in | Manual, dictionary scrape | Candidate, never checked. |
| PIB archive, Doordarshan News | Manual, scrape | Checked, real and reachable, not built against: Hindi does not need more manual tokens right now. |
| `ai4bharat/sangraha` (HF datasets), `verified` and `unverified`, `hin` | Downloaded | Complete. 431M tokens. |

## Assamese (Model L)

| Source | Type | Status |
|---|---|---|
| `docs/assamese/books/`, SCERT/NCERT Class IX-X textbooks, 29 PDFs | Manual, OCR (Tesseract `ben` plus ৰ/ৱ correction) | Complete. 4627/4627 pages. |
| niyomiyabarta.com | Manual, scrape (open-ended loop) | In progress. About 34,000 articles discovered, well over 8M words collected so far. |
| asomiyapratidin.in | Manual, scrape | Current sitemap window mostly drained, small trickle of new articles as they publish. |
| assamese.sentinelassam.com | Manual, scrape (open-ended loop) | In progress. About 17,000 articles discovered. A checkpoint bug found and fixed this session reset its progress counter partway through; the collected text was not lost, but some already-collected articles are being re-fetched before new ones resume. |
| dy365live.com | Manual, scrape (open-ended loop) | Added this session. Mostly caught up to its sitemap's current window, new articles trickle in as they publish. Affected by the same checkpoint bug as sentinelassam, now fixed. |
| assam.news18.com | Manual, scrape with a custom JSON extractor | Added this session. About 40 articles processed from the daily sitemap so far. |
| YouTube auto-generated Assamese captions | Manual, machine transcription (not human transcription, see notes in `assamese/data/SOURCES.md`) | Added this session. Three channels (aboyobbhuyan, JSSUnsscripted, Unfiltered with Krishnakshi), all early in collection, paused partway through this session. |
| dainandinbartagroup.in (Dainandin Barta) | Manual, scrape (open-ended loop) | Added this session. Large catalog, well past 10,000 collected. |
| assam.nenow.in (Northeast Now) | Manual, scrape (open-ended loop) | Added this session. Large catalog, well past 14,000 collected. |
| saneki.in | Manual, scrape (open-ended loop) | Added this session. Weekly literary magazine, well past 11,000 collected. |
| as.wikipedia.org | Manual, dump-based extraction (own preprocessing) | Complete. 25,110 articles, raw about 10.79M words, clean about 10.30M words. |
| as.wikisource.org dump | Manual, dump-based extraction (own preprocessing) | Complete. 403 pages, about 222K words. |
| as.wikiquote.org dump | Manual, dump-based extraction (own preprocessing) | Complete. 1,729 pages, about 437K words. |
| xobdo.org | Manual, dictionary scrape | Complete. 47,972 dictionary entries. |
| archive.org, broader Assamese-language query | Manual, plain-text OCR derivative | Complete and stopped. About 98 genuinely new items, most overlapped with the Digital Library of India source below. Hit archive.org's 10,000-result pagination cap. |
| as.vikaspedia.in | Manual, scrape (own JSON extraction, open-ended loop) | In progress. Added this session. 2334 URLs attempted, 1447 real articles. Same slug-collision bug as Hindi's copy, found and fixed. |
| jibonorshongram.in, ebook listing (Wayback Machine) | Manual, Google Drive PDF plus pdftotext/OCR fallback | In progress. Added this session. 26/78 books finished. Accepted copyright risk, explicit user decision, third-party republished works, must be described accurately in the report. |
| Digital Library of India, Assamese books (archive.org) | Manual, plain-text OCR derivative, no local OCR needed | In progress, open-ended loop. Added this session. Strongest Assamese source found this session: 116 items done, about 5.46M words, and climbing. 2,179 total items available. A filename URL-encoding bug was found and fixed (same fix applied to the Hindi copy above). |
| newsonair.gov.in (Akashvani) | Manual, PDF plus pdftotext/OCR fallback | In progress, open-ended loop with a 30-minute recheck interval. Added this session. 15/16 processed, small catalog, no working pagination found, accumulates new daily bulletins on re-run. |
| jibonorshongram.in, general articles (Wayback Machine) | Manual, scrape | In progress, open-ended loop. Smoke test passed (274 to 383 words per article in a small sample). A minimum-word floor plus the purity filter screens out thin boilerplate posts automatically. |
| Assam Extraordinary Gazette (archive.org) | Manual, plain-text OCR derivative, no local OCR needed | In progress, open-ended loop. Weak yield (about 15 to 19 words per item, mostly English legal text), promoted anyway since it is free, not a priority source. |
| dipr.assam.gov.in | Manual, PDF plus pdftotext/OCR fallback | Complete and stopped. Added this session. 822 items done, about 342K words, all 161 listing pages traversed, catalog fully exhausted. |
| `ai4bharat/sangraha` (HF datasets), `verified` and `unverified`, `asm` | Downloaded | Complete. 140M tokens. Ceiling is well below the 500M target, expected under the spec's allowed shortfall for lower-resource languages. |
| `MWirelabs/assamese-monolingual-corpus` (HF) | Downloaded | Complete. 38.03M real, dedup-aware tokens kept. |
| CC-100, Assamese slice | Downloaded | Added this session, complete. About 4.3M real tokens. |
| `ai4bharat/IndicCorpV2`, `as.txt` (HF) | Downloaded | Added this session, complete. About 63.7M real tokens. |
| Bhashini/ULCA, AIKosh's restricted Akashvani set, PIB main site, PIB archive's regional Assamese page, enewspapers.co.in, Doordarshan News Assamese, LDC-IL, mC4, Sarvam AI, DIKSHA, shodhganga.inflibnet.ac.in | Checked, gated, redundant, or not pursued | See `assamese/data/SOURCES.md` for the reason on each. |
| assamarchive.org ("Digitizing Assam", about 1.28M claimed pages) | Checked, blocked | Login succeeded once, part of the API was reverse-engineered, but the content-listing endpoint failed and the account was then locked out from investigation traffic. Not pursued further without the user's go-ahead. |

## Real token totals

From `report/token_progress.md`. Hindi's collection is halted (both
thresholds already cleared), so its numbers are stable. Assamese collection
is still active, so a fresher rescan will read higher; check
`report/token_progress.md` directly for the latest.

- **Hindi:** 565.97M/500M total (113.2%, target cleared). Manual 134.96M/100M (both thresholds met).
- **Assamese:** 287.94M/500M rough total (57.6%). Manual: 41.79M/100M rough (41.8% of the floor, not met yet). Spec-corrected estimates depend on the tokenizer vocabulary size, not yet finalized. See `report/token_progress.md` for the current per-vocab-size breakdown.

*Last updated: 2026-08-19, ~19:00.*
