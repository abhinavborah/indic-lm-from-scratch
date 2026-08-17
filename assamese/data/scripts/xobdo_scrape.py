#!/usr/bin/env python3
"""xobdo.org Assamese dictionary manual-collection source.

xobdo.org (est. 2006, first online Assamese dictionary, wiki-based/
community-edited) exposes its whole alphabetical word index as ~361
precomputed (start, end) lexicographic range pairs baked into the /alpha
page's HTML (one per `loadWordRange(this, start, end, langId, posId)` call).
Each range maps directly to a clean JSON API --
/2025-web/api/wordsalpha.php?start=..&end=..&l=2 (2 = Assamese) -- that
returns word + part-of-speech + full definition text with no HTML parsing
needed. This script scrapes that JSON API range by range (own preprocessing
of a live site = manual collection, same logic as the Wikipedia dump).

Each API response (one range) is treated as one checkpointed unit: word +
definitions are joined into prose ("word: meaning1; meaning2"), then run
through the same text_clean purity filters as every other source. The
language-ID heuristic is applied per range-batch rather than per individual
word, since a single word+gloss is usually too short for the ৰ/ৱ-frequency
check to be meaningful.

Resumable: assamese/data/.state.json under the "xobdo" namespace tracks the
last fully-processed range index (range order is stable -- baked into the
static HTML). Output: unfiltered prose to assamese/data/raw/scrape/xobdo.txt,
purity-filtered to assamese/data/clean/scrape/xobdo.txt.
"""

import json as jsonlib
import re
import subprocess
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import state_io
from text_clean import clean_text, is_likely_assamese

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
RAW_OUT_PATH = DATA_DIR / "raw" / "scrape" / "xobdo.txt"
CLEAN_OUT_PATH = DATA_DIR / "clean" / "scrape" / "xobdo.txt"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "xobdo"

USER_AGENT = "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"
ALPHA_PAGE_URL = "https://www.xobdo.org/alpha"
API_URL = "https://www.xobdo.org/2025-web/api/wordsalpha.php"
ASSAMESE_LANG_ID = 2
REQUEST_DELAY_S = 1.5
MAX_RETRIES = 3
LOG_EVERY_N_RANGES = 20

_RANGE_RE = re.compile(
    r"loadWordRange\(this,\s*'([^']*)',\s*'([^']*)',\s*(\d+),\s*\d+\)"
)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def fetch(url):
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(REQUEST_DELAY_S)
        r = subprocess.run(
            ["curl", "-s", "-L", "-A", USER_AGENT, "--max-time", "30",
             "-w", "\n%{http_code}", url],
            capture_output=True, stdin=subprocess.DEVNULL,
        )
        if r.returncode != 0:
            continue
        body, _, status = r.stdout.rpartition(b"\n")
        status = status.decode().strip()
        if status == "200":
            return body
        if status == "429":
            time.sleep(REQUEST_DELAY_S * (2 ** attempt))
            continue
        return None
    log(f"FETCH FAILED after {MAX_RETRIES} attempts: {url}")
    return None


def discover_ranges():
    """Parses the /alpha page's baked-in loadWordRange(...) calls for the
    Assamese language id. Range order is the site's own alphabetical order."""
    raw = fetch(ALPHA_PAGE_URL)
    if raw is None:
        return None
    html = raw.decode("utf-8", errors="replace")
    ranges = [
        (start, end) for start, end, lang_id in _RANGE_RE.findall(html)
        if int(lang_id) == ASSAMESE_LANG_ID
    ]
    return ranges or None


def fetch_range_words(start, end):
    url = (f"{API_URL}?start={urllib.parse.quote(start)}"
           f"&end={urllib.parse.quote(end)}&l={ASSAMESE_LANG_ID}&p=0")
    raw = fetch(url)
    if raw is None:
        return None
    try:
        data = jsonlib.loads(raw)
    except jsonlib.JSONDecodeError:
        return None
    if not data.get("success"):
        return None
    return data.get("data", {}).get("words", [])


def words_to_prose(words):
    lines = []
    for w in words:
        meanings = "; ".join(m["text"] for m in w.get("meanings", []) if m.get("text"))
        if meanings:
            lines.append(f"{w['word']}: {meanings}")
    return "\n".join(lines)


def main():
    RAW_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ranges = discover_ranges()
    if ranges is None:
        log("xobdo: could not discover alphabetical ranges from /alpha -- aborting")
        return
    log(f"xobdo: discovered {len(ranges)} alphabetical ranges")

    state = state_io.load_namespace(
        STATE_FILE, NAMESPACE,
        {"last_range_index": 0, "words_written": 0, "total_words": 0,
         "dropped_lines": 0, "suspect_language": 0},
    )
    start_index = state["last_range_index"]
    if start_index >= len(ranges):
        log(f"xobdo: already fully drained ({start_index}/{len(ranges)} ranges)")
        return

    mode = "a" if start_index > 0 else "w"
    with RAW_OUT_PATH.open(mode, encoding="utf-8") as raw_f, \
         CLEAN_OUT_PATH.open(mode, encoding="utf-8") as clean_f:
        for i in range(start_index, len(ranges)):
            start, end = ranges[i]
            words = fetch_range_words(start, end)
            if not words:
                state["last_range_index"] = i + 1
                state_io.save_namespace(STATE_FILE, NAMESPACE, state)
                continue

            prose = words_to_prose(words)
            if prose:
                raw_f.write(prose + "\n\n")
                raw_f.flush()

            cleaned, dropped = clean_text(prose)
            if cleaned and not is_likely_assamese(cleaned):
                log(f"SUSPECT-LANGUAGE (low ৰ/ৱ frequency) range {start!r}-{end!r}")
                state["suspect_language"] += 1
                cleaned = ""

            if cleaned:
                clean_f.write(cleaned + "\n\n")
                clean_f.flush()
                state["words_written"] += len(words)
                state["total_words"] += len(cleaned.split())
            state["dropped_lines"] += dropped
            state["last_range_index"] = i + 1
            state_io.save_namespace(STATE_FILE, NAMESPACE, state)

            if (i + 1) % LOG_EVERY_N_RANGES == 0 or i + 1 == len(ranges):
                log(f"xobdo: {i + 1}/{len(ranges)} ranges, "
                    f"{state['words_written']} dictionary entries, "
                    f"~{state['total_words']} clean words so far")

    log(f"DONE xobdo: {state['words_written']} dictionary entries, "
        f"~{state['total_words']} clean words total, "
        f"{state['suspect_language']} suspect-language ranges excluded")


if __name__ == "__main__":
    main()
