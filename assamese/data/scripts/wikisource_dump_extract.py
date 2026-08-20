#!/usr/bin/env python3
"""Assamese Wikisource manual-collection source: dump-based, same logic as
wiki_dump_extract.py (Assamese Wikipedia): a dump counts as manual
collection as long as *we* do the preprocessing. Wikisource holds
proofread, public-domain literary/document text (poetry, prose, official
documents), a different register from Wikipedia's encyclopedic prose.

Dump source: https://dumps.wikimedia.org/aswikisource/latest/
             aswikisource-latest-pages-articles.xml.bz2 (~41.6MiB compressed)
Downloaded automatically to raw/wikisource_dump/ on first run if missing.

Resumable: assamese/data/.state.json under the "wikisourcedump" namespace
tracks the last fully-processed page index. Output: unfiltered prose to
assamese/data/raw/scrape/as_wikisource.txt, purity-filtered text to
assamese/data/clean/scrape/as_wikisource.txt (same shape as as_wikipedia.txt
so token_tracker.py picks it up automatically via its existing clean/scrape
glob, so no tracker changes needed for this source).
"""

import urllib.request
from datetime import datetime
from pathlib import Path

import state_io
from text_clean import clean_text, is_likely_assamese
from wiki_dump_extract import iter_dump_pages, wikitext_to_prose

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
DUMP_URL = ("https://dumps.wikimedia.org/aswikisource/latest/"
            "aswikisource-latest-pages-articles.xml.bz2")
DUMP_PATH = DATA_DIR / "raw" / "wikisource_dump" / "aswikisource-latest-pages-articles.xml.bz2"
RAW_OUT_PATH = DATA_DIR / "raw" / "scrape" / "as_wikisource.txt"
CLEAN_OUT_PATH = DATA_DIR / "clean" / "scrape" / "as_wikisource.txt"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "wikisourcedump"
LOG_EVERY_N_PAGES = 200


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def ensure_dump():
    if DUMP_PATH.exists():
        return
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {DUMP_URL} -> {DUMP_PATH}")
    req = urllib.request.Request(
        DUMP_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp, DUMP_PATH.open("wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    log(f"Downloaded dump: {DUMP_PATH.stat().st_size} bytes")


def main():
    ensure_dump()

    state = state_io.load_namespace(
        STATE_FILE, NAMESPACE,
        {"last_page_index": 0, "pages_written": 0, "total_words": 0,
         "dropped_lines": 0, "suspect_language": 0},
    )
    start_index = state["last_page_index"]
    log(f"Resuming wikisource dump extraction from page index {start_index}")

    mode = "a" if start_index > 0 else "w"
    RAW_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_OUT_PATH.open(mode, encoding="utf-8") as raw_f, \
         CLEAN_OUT_PATH.open(mode, encoding="utf-8") as clean_f:
        for i, (title, wikitext) in enumerate(iter_dump_pages(DUMP_PATH)):
            if i < start_index:
                continue

            prose = wikitext_to_prose(wikitext)
            if prose:
                raw_f.write(prose + "\n\n")
                raw_f.flush()

            cleaned, dropped = clean_text(prose)
            if cleaned and not is_likely_assamese(cleaned):
                log(f"SUSPECT-LANGUAGE (low ৰ/ৱ frequency): {title}")
                state["suspect_language"] += 1
                cleaned = ""

            if cleaned:
                clean_f.write(cleaned + "\n\n")
                clean_f.flush()
                state["pages_written"] += 1
                state["total_words"] += len(cleaned.split())
            state["dropped_lines"] += dropped
            state["last_page_index"] = i + 1
            state_io.save_namespace(STATE_FILE, NAMESPACE, state)

            if (i + 1) % LOG_EVERY_N_PAGES == 0:
                log(f"as_wikisource (dump): {i + 1} pages scanned, "
                    f"{state['pages_written']} written, ~{state['total_words']} words so far")

    log(f"DONE as_wikisource (dump): {state['pages_written']} pages written, "
        f"~{state['total_words']} words total, {state['suspect_language']} suspect-language pages excluded")


if __name__ == "__main__":
    main()
