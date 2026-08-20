#!/usr/bin/env python3
"""Downloader for Assamese-language texts across ALL of archive.org, not
scoped to one collection (manual-collection corpus).

`language:(asm) AND mediatype:texts` matches 15,385 items across archive.org
(vs. the 2,179 already covered via collection:digitallibraryindia and the
6,024 via the Assam Gazette). Top collections in a sample: opensource,
magazine_rack, community, folkscanomy_religion, plus assorted personal
uploads, a heterogeneous mix unlike DLI's curated book catalog, so yield
and quality per item are unproven until real batches land. Same
archive.org auto-OCR `_djvu.txt` mechanism as the other two sources, no
local OCR needed.

Cross-source dedup: this query's result set very likely overlaps with
identifiers already collected via dli_books (digitallibraryindia is itself
tagged language:asm, so it's a strict subset of this broader query) and
possibly assam_gazette. process_item() checks all three namespaces before
downloading anything, so nothing gets double-counted or re-fetched.

advancedsearch.php hard-caps sorted pagination at 10,000; with 15,385
total matches, roughly 5,000 items past the cap are unreachable via this
method (same documented limitation as dli_books_download.py; see that
script's discover_identifiers docstring for why the cursor-based scrape API
alternative was tried and reverted).

Discovery: archive.org's advancedsearch API, paginated.
Resumable: assamese/data/.state.json tracks per-identifier completion.
"""

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import state_io
from text_clean import clean_text

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
RAW_OUT_DIR = DATA_DIR / "raw" / "archive_broad"
CLEAN_OUT_DIR = DATA_DIR / "clean" / "archive_broad"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "archive_broad"
# Other namespaces in the same state file whose already-downloaded
# identifiers must not be re-fetched or double-counted by this broader query.
OTHER_NAMESPACES = ("dli_books", "assam_gazette")

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{id}"
DOWNLOAD_URL = "https://archive.org/download/{id}/{fname}"
USER_AGENT = "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"
REQUEST_TIMEOUT_S = 30
PAGE_SIZE = 100
DEFAULT_MAX_NEW_ITEMS = 100
# archive.org is a large, CDN-backed service built for heavy concurrent
# traffic (unlike the small news sites this project also scrapes), so a
# handful of concurrent connections is well within normal usage, not
# aggressive scraping. Each worker still paces its own requests (see
# process_item's sleep), so this multiplies throughput without bursting.
WORKERS = 4
# Real state-file writes (state_io) are already flock-serialized across
# processes; this lock only protects the in-memory counters below, which
# multiple threads in *this* process update concurrently.
_counters_lock = threading.Lock()


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def load_state():
    return state_io.load_namespace(STATE_FILE, NAMESPACE, {})


def save_state(state):
    state_io.save_namespace(STATE_FILE, NAMESPACE, state)


def load_other_seen_identifiers():
    """Union of identifiers already downloaded under sibling archive.org
    sources sharing this same state file, so this broader query never
    re-fetches or double-counts them."""
    seen = set()
    for ns in OTHER_NAMESPACES:
        ns_state = state_io.load_namespace(STATE_FILE, ns, {})
        seen.update(k for k in ns_state if not k.startswith("_"))
    return seen


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
    """Same 10,000-result advancedsearch.php cap as dli_books_download.py
    and assam_gazette_download.py; see those scripts for why the
    cursor-based scrape API alternative was tried and reverted."""
    params = {
        "q": "language:(asm) AND mediatype:texts",
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


def fetch_download(url):
    """Like fetch(), but distinguishes a permanent access restriction
    (401/403: archive.org lending-library items that require their own
    separate archive.org login, unrelated to this project and unfixable
    here) from a transient failure. Observed live: without this, the same
    couple of restricted items get retried on every page forever (they
    keep reappearing across different `start` offsets, a pagination quirk
    on archive.org's side), pure wasted requests and log noise. Marking
    them permanently skips that."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return resp.read(), False
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return None, True
        log(f"fetch failed {url}: {e}")
        return None, False
    except Exception as e:
        log(f"fetch failed {url}: {e}")
        return None, False


def process_item(identifier, state, other_seen):
    if identifier in state or identifier in other_seen:
        return False

    fname = djvu_txt_filename(identifier)
    if fname is None:
        state[identifier] = {"status": "no_djvu_txt"}
        save_state(state)
        return True

    body, restricted = fetch_download(DOWNLOAD_URL.format(id=identifier, fname=urllib.parse.quote(fname)))
    if restricted:
        state[identifier] = {"status": "access_restricted"}
        save_state(state)
        return True
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
    log(f"archive_broad {identifier}: ~{words} words")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-items", type=int, default=DEFAULT_MAX_NEW_ITEMS)
    args = parser.parse_args()

    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)

    other_seen = load_other_seen_identifiers()
    state = load_state()
    start = state.get("_last_search_start", 0)
    new_count = 0
    words_total = 0
    skipped_dup = 0
    last_num_found = None

    while new_count < args.max_new_items:
        result = discover_identifiers(start)
        if result is None:
            if start >= 10000:
                log(f"archive_broad: hit archive.org's advancedsearch.php 10,000-result pagination cap "
                    f"at start={start}: collection has more items but this method cannot reach them")
            else:
                log(f"archive_broad: discovery request failed at start={start} (transient); "
                    f"will retry from same position next run")
            break
        ids, last_num_found = result["ids"], result["num_found"]
        if not ids:
            log(f"archive_broad: reached genuine end of search results "
                f"({last_num_found} total items, all covered)")
            break

        to_process = []
        for identifier in ids:
            if identifier in other_seen:
                skipped_dup += 1
            else:
                to_process.append(identifier)

        def handle_one(identifier):
            # Each worker still paces its own requests: WORKERS concurrent
            # workers each doing ~0.3s-spaced requests multiplies aggregate
            # throughput without any single connection bursting.
            attempted = process_item(identifier, state, other_seen)
            time.sleep(0.3)
            return identifier, attempted

        # Process this whole page concurrently, but only advance `start`
        # (the resume checkpoint) once every item in it has been attempted,
        # the same page-boundary checkpoint guarantee as the sequential
        # version, so a kill mid-page just re-attempts that page's
        # not-yet-in-state items next run, nothing silently skipped.
        if to_process:
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                for identifier, attempted in executor.map(handle_one, to_process):
                    if attempted:
                        with _counters_lock:
                            new_count += 1
                            words_total += state.get(identifier, {}).get("words", 0)

        start += PAGE_SIZE
        state["_last_search_start"] = start
        save_state(state)

    found_note = f", {last_num_found} total items" if last_num_found is not None else ""
    log(f"archive_broad BATCH COMPLETE: {new_count} new items processed, ~{words_total} words this batch, "
        f"{skipped_dup} skipped as already collected via other sources{found_note}")


if __name__ == "__main__":
    main()
