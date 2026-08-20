#!/usr/bin/env python3
"""Downloader for the Assam Extraordinary Gazette on archive.org (manual-collection corpus).

`in.gazette.assam_extraordinary.*` on archive.org: 12,744+ bilingual
(English+Assamese) government gazette notifications, 2017-2026, uploaded by
the National Informatics Centre. archive.org already runs Tesseract OCR
(lang-eng;lang-asm) on upload and publishes a plain-text `_djvu.txt`
derivative per item, so no PDF download or local OCR needed, just fetch that
file directly.

Caveat found during a manual spot-check: gazette notices are often
majority-English legal text with Assamese confined to the masthead, so
real Assamese yield per item may be low. The purity filter (clean_text)
already drops non-Assamese-majority lines, so this self-corrects at count
time; still worth watching the kept-word rate once real numbers land.

Discovery: archive.org's advancedsearch API, paginated.
Resumable: assamese/data/.state.json tracks per-identifier completion.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import state_io
from text_clean import clean_text

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
RAW_OUT_DIR = DATA_DIR / "raw" / "assam_gazette"
CLEAN_OUT_DIR = DATA_DIR / "clean" / "assam_gazette"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "assam_gazette"

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{id}"
DOWNLOAD_URL = "https://archive.org/download/{id}/{fname}"
USER_AGENT = "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"
REQUEST_TIMEOUT_S = 30
PAGE_SIZE = 100
DEFAULT_MAX_NEW_ITEMS = 100


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def load_state():
    return state_io.load_namespace(STATE_FILE, NAMESPACE, {})


def save_state(state):
    state_io.save_namespace(STATE_FILE, NAMESPACE, state)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return resp.read()
    except Exception as e:
        log(f"fetch failed {url}: {e}")
        return None


def correct_ra(text):
    return text.replace("র", "ৰ")


def discover_identifiers(start):
    """Tried archive.org's cursor-based scrape API to page past
    advancedsearch.php's 10,000-result cap, but its cursor mechanism proved
    unreliable here (page-2 requests consistently rejected the page-1
    cursor with "Bad cursor", reproduced across multiple attempts,
    encodings, and a raw rtk-proxy bypass). Reverted to advancedsearch.php:
    this collection's real size under this query (6,024 items, confirmed
    via the scrape API's own `total` field) is under the cap anyway, so the
    cap was never actually the constraint for this source."""
    params = {
        "q": "identifier:in.gazette.assam_extraordinary.* AND mediatype:texts",
        "fl[]": "identifier",
        "sort[]": "identifier asc",
        "rows": PAGE_SIZE,
        "start": start,
        "output": "json",
    }
    url = SEARCH_URL + "?" + urllib.parse.urlencode(params)
    body = fetch(url)
    if body is None:
        return None
    try:
        data = json.loads(body)
        return {
            "ids": [d["identifier"] for d in data["response"]["docs"]],
            "num_found": data["response"]["numFound"],
        }
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # archive.org returns this same malformed/error shape for two very
        # different reasons that used to be indistinguishable in the log:
        # a genuine transient hiccup (rate-limit blip, server error) at any
        # start, or, specifically once start >= 10,000, the documented,
        # permanent advancedsearch.php pagination cap (see docstring above).
        # main() tells these apart using `start` at the call site.
        log(f"discover_identifiers: malformed search response at start={start}: {e}")
        return None


def djvu_txt_filename(identifier):
    body = fetch(METADATA_URL.format(id=identifier))
    if body is None:
        return None
    data = json.loads(body)
    for f in data.get("files", []):
        if f.get("format") == "DjVuTXT":
            return f["name"]
    return None


def process_item(identifier, state):
    if identifier in state:
        return False

    fname = djvu_txt_filename(identifier)
    if fname is None:
        state[identifier] = {"status": "no_djvu_txt"}
        save_state(state)
        return True

    body = fetch(DOWNLOAD_URL.format(id=identifier, fname=urllib.parse.quote(fname)))
    if body is None:
        return False  # transient: leave off state, retry next run

    raw_text = correct_ra(body.decode("utf-8", errors="ignore"))
    cleaned, _dropped = clean_text(raw_text)

    with (RAW_OUT_DIR / f"{identifier}.txt").open("w", encoding="utf-8") as f:
        f.write(raw_text + "\n\n")
    with (CLEAN_OUT_DIR / f"{identifier}.txt").open("w", encoding="utf-8") as f:
        f.write(cleaned + "\n\n")

    words = len(cleaned.split())
    state[identifier] = {"status": "done", "words": words}
    save_state(state)
    log(f"assam_gazette {identifier}: ~{words} words")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-items", type=int, default=DEFAULT_MAX_NEW_ITEMS)
    args = parser.parse_args()

    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    start = state.get("_last_search_start", 0)
    new_count = 0
    words_total = 0
    last_num_found = None

    while new_count < args.max_new_items:
        result = discover_identifiers(start)
        if result is None:
            # Same failure shape, two different meanings; start tells them
            # apart: below 10,000 it's a genuine transient hiccup (state
            # isn't advanced, next run retries this exact position); at or
            # above 10,000 it's the permanent advancedsearch.php cap, and no
            # amount of retrying will get past it.
            if start >= 10000:
                log(f"assam_gazette: hit archive.org's advancedsearch.php 10,000-result pagination cap "
                    f"at start={start}: collection has more items but this method cannot reach them")
            else:
                log(f"assam_gazette: discovery request failed at start={start} (transient); "
                    f"will retry from same position next run")
            break
        ids, last_num_found = result["ids"], result["num_found"]
        if not ids:
            log(f"assam_gazette: reached genuine end of search results "
                f"({last_num_found} total items in collection, all covered)")
            break
        for identifier in ids:
            if new_count >= args.max_new_items:
                break
            attempted = process_item(identifier, state)
            if attempted:
                new_count += 1
                words_total += state.get(identifier, {}).get("words", 0)
            time.sleep(0.3)
        start += PAGE_SIZE
        state["_last_search_start"] = start
        save_state(state)

    found_note = f", {last_num_found} total items in collection" if last_num_found is not None else ""
    log(f"assam_gazette BATCH COMPLETE: {new_count} new items processed, ~{words_total} words this batch{found_note}")


if __name__ == "__main__":
    main()
