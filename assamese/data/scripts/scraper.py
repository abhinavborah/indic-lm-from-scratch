#!/usr/bin/env python3
"""Assamese news scraper (manual-collection corpus).

Sources are discovered via each site's own sitemap.xml (a sitemapindex is
followed one level down into its child <urlset> sitemaps). Article text is
pulled from the page's <article> container via BeautifulSoup, then run
through a language-purity filter: any line whose alphabetic characters
aren't majority Bengali-Assamese-block Unicode (U+0980-U+09FF) is dropped --
strips the English bylines, "Advertisment"/"Follow Us"/date-stamp boilerplate
that WordPress/Publive templates interleave with the real article text.

robots.txt Disallow rules are not honored here (bypass authorized for this
project) but requests are still throttled -- REQUEST_DELAY_S between every
HTTP call, regardless of site.

Resumable: assamese/data/.state.json under the "scrape" namespace tracks
every article URL already written (success or permanent failure), so a
rerun only fetches new URLs. Shares the state file with ocr_pipeline.py via
state_io's flock-protected read-modify-write -- safe to run concurrently.
"""

import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

import state_io
from text_clean import clean_text, is_likely_assamese

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
RAW_OUT_DIR = DATA_DIR / "raw" / "scrape"
CLEAN_OUT_DIR = DATA_DIR / "clean" / "scrape"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "scrape"

USER_AGENT = "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"
REQUEST_DELAY_S = 1.5
MAX_RETRIES = 3
MAX_CHILD_SITEMAPS = 1500  # sentinelassam's sitemap.xml alone lists 1,346 daily
                            # child sitemaps (one per publishing day since ~2018);
                            # the old 50-document cap silently missed all but the
                            # most recent ~50 days for any site with that many
                            # children, including niyomiyabarta -- raising it also
                            # widens niyomiyabarta's own reachable backlog, not
                            # just unlocking this new source.
LOG_EVERY_N_ARTICLES = 20
BATCH_MAX_ARTICLES = 1_000_000  # effectively uncapped -- run until new_urls is exhausted
BATCH_MAX_SECONDS = 86400       # 24h safety net, not a real pacing cap anymore
# These are small news sites, not CDN-backed archives -- unlike archive.org,
# a burst of connections risks tripping a basic rate-limiter/WAF. Kept
# deliberately low (vs. e.g. archive_broad_download.py's 4): each worker
# still waits REQUEST_DELAY_S between its own requests, so this multiplies
# throughput a few times over without a sudden spike in concurrent hits.
WORKERS_PER_SOURCE = 3

# as.wikipedia.org is handled by wiki_dump_extract.py (dump-based, not live
# scraping here) -- see that script for why.
SOURCES = [
    {
        "name": "niyomiyabarta",
        "sitemap": "https://niyomiyabarta.com/sitemap.xml",
        "content_selector": "article",
    },
    {
        "name": "asomiyapratidin",
        "sitemap": "https://asomiyapratidin.in/news-sitemap.xml",
        "content_selector": "article",
    },
    {
        "name": "sentinelassam",
        "sitemap": "https://assamese.sentinelassam.com/sitemap.xml",
        # Modern React app, CSS-module hashed classnames (e.g.
        # "text-story-m_story-content-inner-wrapper__s3KPp") -- matched by
        # stable prefix via [class*=], not the volatile hash suffix. Two
        # elements share that prefix per page: the real text body, and an
        # empty hero-image caption wrapper (extra "hero-image" class) --
        # :not() excludes the latter. Verified against a live article
        # (World Environment Day piece, 05 Jun 2026): real Assamese prose,
        # not boilerplate.
        "content_selector": 'div[class*="text-story-m_story-content-inner-wrapper"]:not([class*="hero-image"])',
    },
]


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def fetch(url, user_agent=USER_AGENT):
    # Shells out to curl rather than urllib.request: urllib sockets hang
    # indefinitely against these sites in this environment's network
    # sandbox, while curl (used for pdftotext/pdftoppm elsewhere in this
    # project) works reliably.
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(REQUEST_DELAY_S)
        r = subprocess.run(
            ["curl", "-s", "-L", "-A", user_agent, "--max-time", "20",
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


def discover_urls(sitemap_url):
    """Returns None on total discovery failure (network/parse) so callers
    can distinguish "couldn't check" from "checked, nothing new" -- a
    transient fetch failure must never look like a fully-drained source."""
    raw = fetch(sitemap_url)
    if raw is None:
        return None
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        log(f"SKIP unparseable sitemap {sitemap_url}: {e}")
        return None

    tag = root.tag.split("}")[-1]
    ns = {"s": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

    def locs(node):
        return [e.text.strip() for e in node.findall(".//s:loc", ns)] if ns else \
               [e.text.strip() for e in node.findall(".//loc")]

    if tag == "sitemapindex":
        child_sitemaps = locs(root)[:MAX_CHILD_SITEMAPS]
        urls = []
        for child in child_sitemaps:
            child_raw = fetch(child)
            if child_raw is None:
                log(f"SKIP child sitemap (fetch failed): {child}")
                continue
            try:
                child_root = ET.fromstring(child_raw)
            except ET.ParseError as e:
                log(f"SKIP unparseable child sitemap {child}: {e}")
                continue
            urls.extend(locs(child_root))
        return urls
    return locs(root)




def scrape_article(url, content_selector):
    """Returns None on a transient fetch failure (network/timeout -- retry
    later, never mark done), "" when the page loaded but the selector found
    no article container (confirmed non-article page -- permanent skip), or
    the extracted text otherwise. Distinct from `fetch`'s own None, which
    means "no bytes came back at all"."""
    raw = fetch(url)
    if raw is None:
        return None
    soup = BeautifulSoup(raw, "html.parser")
    container = soup.select_one(content_selector)
    if container is None:
        return ""
    return container.get_text(separator="\n", strip=True)


def discover_items(source):
    return discover_urls(source["sitemap"])


def fetch_item_text(source, item):
    return scrape_article(item, source["content_selector"])


def process_source(source, scrape_state):
    name = source["name"]
    entry = scrape_state.setdefault(
        name, {"done_urls": [], "total_words": 0, "dropped_lines": 0, "suspect_language": 0}
    )
    entry.setdefault("suspect_language", 0)  # back-compat for entries saved before this field existed
    done = set(entry["done_urls"])

    urls = None
    for attempt in range(3):
        urls = discover_items(source)
        if urls is not None:
            break
        log(f"{name}: discovery attempt {attempt + 1}/3 failed, retrying")
        time.sleep(REQUEST_DELAY_S * (attempt + 1))
    if urls is None:
        log(f"{name}: discovery failed 3/3 times -- skipping this run, "
            f"NOT marking as drained ({len(entry['done_urls'])} done so far)")
        return

    new_urls = [u for u in urls if u not in done]
    log(f"{name}: discovery found {len(urls)} items, {len(new_urls)} new")

    raw_path = RAW_OUT_DIR / f"{name}.txt"
    clean_path = CLEAN_OUT_DIR / f"{name}.txt"
    processed_this_run = 0
    batch_start = time.time()
    hit_batch_cap = False
    # fetch_item_text() (the network call) runs fully concurrently across
    # WORKERS_PER_SOURCE threads -- that's the slow part worth parallelizing.
    # Everything that touches the shared raw_f/clean_f file handles or the
    # entry dict is serialized under this lock so writes never interleave;
    # the lock is only ever held for fast in-memory/disk-append work, so it
    # doesn't erase the concurrency gain from the network calls above it.
    write_lock = threading.Lock()

    with raw_path.open("a", encoding="utf-8") as raw_f, \
         clean_path.open("a", encoding="utf-8") as clean_f:

        def handle_one(url):
            nonlocal processed_this_run, hit_batch_cap
            if hit_batch_cap:
                return

            text = fetch_item_text(source, url)
            if text is None:
                return  # transient fetch failure -- leave off done_urls, retry next run

            with write_lock:
                if not text:
                    entry["done_urls"].append(url)  # confirmed non-article page -- permanent skip
                    state_io.save_namespace(STATE_FILE, NAMESPACE, scrape_state)
                    return

                raw_f.write(text + "\n\n")
                raw_f.flush()

                cleaned, dropped = clean_text(text)
                if cleaned and not is_likely_assamese(cleaned):
                    log(f"SUSPECT-LANGUAGE (low ৰ/ৱ frequency, likely Bengali not Assamese): {url}")
                    entry["suspect_language"] += 1
                    cleaned = ""  # excluded from the clean corpus, but URL still marked done below

                if cleaned:
                    clean_f.write(cleaned + "\n\n")
                    clean_f.flush()

                entry["done_urls"].append(url)
                entry["total_words"] += len(cleaned.split())
                entry["dropped_lines"] += dropped
                # ponytail: full state file rewritten every article (O(n) per
                # save) -- fine up to tens of thousands of URLs; if it becomes
                # the bottleneck, switch done_urls to an append-only JSONL log.
                state_io.save_namespace(STATE_FILE, NAMESPACE, scrape_state)

                processed_this_run += 1
                if processed_this_run % LOG_EVERY_N_ARTICLES == 0:
                    log(f"{name}: {processed_this_run}/{len(new_urls)} articles this run, "
                        f"~{entry['total_words']} words total so far, "
                        f"{entry['dropped_lines']} non-Assamese lines dropped so far")
                if time.time() - batch_start >= BATCH_MAX_SECONDS:
                    hit_batch_cap = True

        with ThreadPoolExecutor(max_workers=WORKERS_PER_SOURCE) as executor:
            list(executor.map(handle_one, new_urls))

    remaining = len(new_urls) - processed_this_run
    if hit_batch_cap:
        log(f"BATCH STOP {name}: {processed_this_run} new articles this batch "
            f"({time.time() - batch_start:.0f}s), {remaining} still remaining, "
            f"{len(entry['done_urls'])} total ever, ~{entry['total_words']} words total")
    else:
        log(f"DONE {name}: {processed_this_run} new articles, "
            f"{len(entry['done_urls'])} total ever, ~{entry['total_words']} words total")


def main():
    import sys
    from concurrent.futures import ThreadPoolExecutor

    # Comma-separated source names to run concurrently, e.g.
    # "niyomiyabarta,asomiyapratidin". No arg = all SOURCES.
    requested = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    selected = [s for s in SOURCES if not requested or s["name"] in requested]
    if not selected:
        log(f"No matching sources for {requested!r} -- known: {[s['name'] for s in SOURCES]}")
        return

    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    scrape_state = state_io.load_namespace(STATE_FILE, NAMESPACE, {})

    # fetch() is blocking (subprocess curl calls), so plain OS threads give
    # real I/O concurrency here -- no asyncio/aiohttp needed. Each thread
    # only ever touches its own source's sub-dict in scrape_state (keyed by
    # distinct source name), and state_io's per-call flock serializes the
    # actual file writes, so this is safe without extra locking.
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        list(pool.map(lambda s: process_source(s, scrape_state), selected))

    log("Scraper run complete.")


if __name__ == "__main__":
    main()
