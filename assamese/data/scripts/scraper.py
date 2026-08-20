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

import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
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
# Each source gets its OWN top-level state_io namespace, not sub-keys under
# one shared "scrape" dict. Found this session (19 Aug 2026): sentinelassam
# and dy365 each run as their own separate OS process (own herdr pane), and
# each process loads the whole "scrape" namespace once at startup -- a
# process that doesn't know about a sibling source's sub-key wipes it on
# its next save (last writer wins for the WHOLE namespace). This actually
# happened: sentinelassam's checkpoint (7280+ done_urls) was wiped to zero
# by a save from the niyomiyabarta/asomiyapratidin process, which had never
# seen sentinelassam's entry. The raw/clean text files were unaffected
# (append-only, separate from state.json), only the checkpoint pointer was
# lost -- but a resume would have re-fetched and re-appended all 7280+
# already-done articles. Same root cause and same fix as
# youtube_captions.py's namespace-per-channel fix earlier this session.
NAMESPACE_PREFIX = "scrape"

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
    {
        "name": "dy365",
        "sitemap": "https://dy365live.com/news-sitemap.xml",
        # Plain WordPress-style <article> wrapper -- verified against a live
        # article (CNG pump explosion story, 19 Aug 2026): clean, coherent
        # Assamese prose, same shape as niyomiyabarta/asomiyapratidin.
        "content_selector": "article",
        # news-sitemap.xml alone only exposes a narrow ~170-article recent
        # window (confirmed this session). The category listing page
        # (dy365live.com/assamese?page=N) goes much deeper -- pages 20, 50,
        # 75, 90, 95 all returned 10 fresh unique articles each when checked
        # this session, so this is combined with the sitemap rather than
        # replacing it.
        "discover": "dy365_pagination",
    },
    {
        "name": "dainandinbartagroup",
        "sitemap": "https://www.dainandinbartagroup.in/sitemap_index.xml",
        # Plain WordPress <article> wrapper, same as niyomiyabarta. Verified
        # against a live article (sports story, 29 Nov 2024): real, coherent
        # Assamese. Found via firecrawl search this session, not previously
        # in SOURCES.md.
        "content_selector": "article",
    },
    {
        "name": "nenow",
        "sitemap": "https://assam.nenow.in/sitemap_index.xml",
        # Northeast Now, Assamese edition. WordPress theme uses
        # div.entry-content, not a plain <article> tag (that selector only
        # matched a short reused excerpt widget, not the real body).
        # Verified against a live article (govt appointment brief): real,
        # coherent Assamese. Found via firecrawl search this session.
        "content_selector": "div.entry-content",
    },
    {
        "name": "saneki",
        "sitemap": "https://www.saneki.in/sitemap.xml",
        # Weekly Assamese digital literary magazine (poetry, essays). Plain
        # <article> wrapper. Verified against a live poem: real, natural
        # Assamese literary prose, good source for the "natural phrasing"
        # quality the spec cares about, not just news boilerplate. Found via
        # firecrawl search this session. Paginated sitemap (Blogger-style
        # ?page=N children), same shape as thereveal.co.in's, but this site
        # is Assamese-primary where thereveal sampled as English-primary.
        "content_selector": "article",
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


REFRESH_RECENT_CHILDREN = 5  # always re-fetch the N most-recent child sitemaps (today's
                             # daily sitemap can still gain new URLs intraday); everything
                             # older than that is immutable once published, so a cached
                             # child sitemap's URL list is trusted rather than re-fetched


def discover_urls(sitemap_url, sitemap_cache=None):
    """Returns None on total discovery failure (network/parse) so callers
    can distinguish "couldn't check" from "checked, nothing new" -- a
    transient fetch failure must never look like a fully-drained source.

    sitemap_cache (optional, mutated in place, persisted by the caller in
    the source's checkpoint entry): a dict of child-sitemap-url -> its
    article URL list. Found this session: a large sitemapindex (sites with
    daily child sitemaps going back years, e.g. sentinelassam) was being
    walked in full from scratch on every single restart -- 1000+ child
    fetches, 40-60+ minutes, just to rebuild a URL list that's almost
    entirely unchanged run to run. Only REFRESH_RECENT_CHILDREN children are
    re-fetched; older ones reuse their cached result."""
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
        cache = sitemap_cache if sitemap_cache is not None else {}
        urls = []
        for i, child in enumerate(child_sitemaps):
            if child in cache and i >= REFRESH_RECENT_CHILDREN:
                urls.extend(cache[child])
                continue
            child_raw = fetch(child)
            if child_raw is None:
                if child in cache:
                    urls.extend(cache[child])  # transient fetch failure -- fall back to last-known-good
                else:
                    log(f"SKIP child sitemap (fetch failed): {child}")
                continue
            try:
                child_root = ET.fromstring(child_raw)
            except ET.ParseError as e:
                log(f"SKIP unparseable child sitemap {child}: {e}")
                continue
            child_urls = locs(child_root)
            cache[child] = child_urls
            urls.extend(child_urls)
        # prune cache entries for sitemaps no longer listed (site reorganized/pruned old ones)
        for stale in set(cache) - set(child_sitemaps):
            del cache[stale]
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


ARTICLE_URL_RE = re.compile(rb'href="(https://dy365live\.com/[a-zA-Z-]+/[a-zA-Z0-9-]+-[0-9]{6,})"')
DY365_MAX_PAGES = 95  # verified reachable this session (pages 20/50/75/90/95 all
                       # returned fresh articles); page=100 itself 403s on every
                       # attempt including after backoff, looks like a WAF rule on
                       # that specific round number rather than genuine end of
                       # catalog, so this is a conservative floor, not a hard ceiling


def discover_paginated_html(base_url, max_pages):
    """Category-listing pagination discovery (?page=N), for sites whose
    sitemap only exposes a narrow recent-articles window. Same URL de-dupe
    happens naturally downstream (process_source filters against done_urls
    from whichever discovery method already saw a given URL)."""
    urls = set()
    for page in range(1, max_pages + 1):
        body = fetch(f"{base_url}?page={page}")
        if body is None:
            log(f"dy365: page {page} fetch failed, stopping pagination discovery here")
            break
        found = ARTICLE_URL_RE.findall(body)
        if not found:
            log(f"dy365: page {page} had no article links, stopping pagination discovery here")
            break
        urls.update(u.decode("utf-8") for u in found)
    return list(urls)


def discover_items(source, sitemap_cache=None):
    if source.get("discover") == "dy365_pagination":
        sitemap_urls = discover_urls(source["sitemap"], sitemap_cache) or []
        paginated_urls = discover_paginated_html("https://dy365live.com/assamese", DY365_MAX_PAGES)
        return list(set(sitemap_urls) | set(paginated_urls))
    return discover_urls(source["sitemap"], sitemap_cache)


def fetch_item_text(source, item):
    return scrape_article(item, source["content_selector"])


def process_source(source):
    name = source["name"]
    namespace = f"{NAMESPACE_PREFIX}_{name}"
    entry = state_io.load_namespace(
        STATE_FILE, namespace,
        {"done_urls": [], "total_words": 0, "dropped_lines": 0, "suspect_language": 0},
    )
    entry.setdefault("suspect_language", 0)  # back-compat for entries saved before this field existed
    entry.setdefault("sitemap_cache", {})
    done = set(entry["done_urls"])

    urls = None
    for attempt in range(3):
        urls = discover_items(source, entry["sitemap_cache"])
        if urls is not None:
            break
        log(f"{name}: discovery attempt {attempt + 1}/3 failed, retrying")
        time.sleep(REQUEST_DELAY_S * (attempt + 1))
    state_io.save_namespace(STATE_FILE, namespace, entry)  # persist sitemap_cache even if discovery then fails
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

    # Failed items are pushed to the tail of this queue and retried once
    # more before being given up on for this run: fetch() already retries a
    # single URL up to MAX_RETRIES times internally (with backoff on 429),
    # so a queue-level "failed" here means that whole internal retry budget
    # was exhausted once already. requeued tracks which URLs have already
    # had their one requeue -- a second failure after that is permanent for
    # this run (not marked done, so the next full script invocation retries
    # it fresh, but this run stops chasing it).
    work_queue = Queue()
    for u in new_urls:
        work_queue.put(u)
    requeued = set()
    permanently_failed = []

    with raw_path.open("a", encoding="utf-8") as raw_f, \
         clean_path.open("a", encoding="utf-8") as clean_f:

        def worker():
            nonlocal processed_this_run, hit_batch_cap
            while True:
                if hit_batch_cap:
                    return
                try:
                    url = work_queue.get_nowait()
                except Empty:
                    return

                text = fetch_item_text(source, url)
                if text is None:
                    with write_lock:
                        if url in requeued:
                            log(f"{name}: {url} failed again after requeue -- giving up for this run")
                            permanently_failed.append(url)
                        else:
                            log(f"{name}: {url} failed, pushed to end of queue, retrying after the rest")
                            requeued.add(url)
                            work_queue.put(url)
                    continue

                with write_lock:
                    if not text:
                        entry["done_urls"].append(url)  # confirmed non-article page -- permanent skip
                        state_io.save_namespace(STATE_FILE, namespace, entry)
                        continue

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
                    state_io.save_namespace(STATE_FILE, namespace, entry)

                    processed_this_run += 1
                    if processed_this_run % LOG_EVERY_N_ARTICLES == 0:
                        log(f"{name}: {processed_this_run}/{len(new_urls)} articles this run, "
                            f"~{entry['total_words']} words total so far, "
                            f"{entry['dropped_lines']} non-Assamese lines dropped so far")
                    if time.time() - batch_start >= BATCH_MAX_SECONDS:
                        hit_batch_cap = True

        threads = [threading.Thread(target=worker) for _ in range(WORKERS_PER_SOURCE)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    remaining = len(new_urls) - processed_this_run - len(permanently_failed)
    if hit_batch_cap:
        log(f"BATCH STOP {name}: {processed_this_run} new articles this batch "
            f"({time.time() - batch_start:.0f}s), {remaining} still remaining, "
            f"{len(permanently_failed)} gave up after requeue this run, "
            f"{len(entry['done_urls'])} total ever, ~{entry['total_words']} words total")
    elif permanently_failed:
        log(f"RUN DONE (with failures) {name}: {processed_this_run} new articles, "
            f"{len(permanently_failed)} gave up after requeue this run (will retry next invocation), "
            f"{len(entry['done_urls'])} total ever, ~{entry['total_words']} words total")
    else:
        log(f"COMPLETE {name}: all {len(new_urls)} discovered new items processed, "
            f"{processed_this_run} new articles, "
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

    # fetch() is blocking (subprocess curl calls), so plain OS threads give
    # real I/O concurrency here -- no asyncio/aiohttp needed. Each source now
    # loads/saves its own namespace independently (see NAMESPACE_PREFIX
    # comment above), so this is safe both within one process and across
    # separate processes/panes running different sources concurrently.
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        list(pool.map(lambda s: process_source(s), selected))

    log("Scraper run complete.")


if __name__ == "__main__":
    main()
