#!/usr/bin/env python3
"""Hindi news scraper (manual-collection corpus), multi-source concurrent.

One worker thread per source runs concurrently within this single process
(plain I/O-bound concurrency, not separate processes/agents). Each worker
discovers article URLs via the site's own sitemap.xml and scrapes them one
at a time within its own thread, rate-limited per domain so concurrency
multiplies *sources* scraped in parallel without hammering any single site
harder than a sequential scraper would.

Article text extraction: BeautifulSoup pulls the source's content container,
then clean_text.py applies NFC normalization + word-level Latin-script
stripping + line-level Devanagari-density filtering (digits are fine, but
embedded English proper nouns/brand names must be stripped at the word
level, not just by dropping whole lines).

Resumable: hindi/data/.state.json under the "scrape" namespace tracks every
article URL already written (success or permanent failure) per source, via
state_io's flock-protected read-modify-write -- safe under concurrent writes
from multiple source-worker threads, and safe to run alongside other
collection scripts.

robots.txt Disallow rules are not honored (bypass authorized for this
project) but requests are still throttled per domain.

Hindi Wikipedia is handled separately -- see wiki_dump_extract.py, which
processes the official XML dump rather than live-scraping the API.
"""

import argparse
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import state_io
from clean_text import clean_text

DATA_DIR = Path(__file__).resolve().parents[1]  # hindi/data
OUT_DIR = DATA_DIR / "raw" / "scrape"      # original, unfiltered article text
CLEAN_DIR = DATA_DIR / "clean" / "scrape"  # clean_text() output, same filenames
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
# Each source gets its own top-level state_io namespace, not a sub-key
# under one shared "scrape" dict. This script's own docstring guarantees
# all sources run as threads within one process (never separate OS
# processes), so the shared-dict-in-memory design was never unsafe here in
# practice -- but the Assamese copy of this script had the identical
# pattern and it broke the moment two of its sources ended up running in
# separate processes (separate herdr panes), each loading a stale full-
# namespace snapshot and wiping the other's checkpoint on save. Applying
# the same per-source-namespace fix here defensively, so this script stays
# safe even if a future session ever runs its sources as separate
# processes instead of threads.
NAMESPACE_PREFIX = "scrape"

USER_AGENT = "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"
REQUEST_TIMEOUT_S = 20
REQUEST_DELAY_S = 1.5  # per-domain floor between requests -- each source's own worker thread only ever
                       # hits its own domain, so this throttles per-site regardless of source count
MAX_RETRIES = 3
MAX_CHILD_SITEMAPS = 50
LOG_EVERY_N_ARTICLES = 20
DEFAULT_BATCH_MAX_ARTICLES = 300   # per source, per batch
DEFAULT_BATCH_MAX_SECONDS = 3600   # per source, per batch

SOURCES = [
    {"name": "jagran", "sitemap": "https://www.jagran.com/sitemap.xml", "content_selector": "article"},
    {"name": "amarujala", "sitemap": "https://www.amarujala.com/sitemap.xml", "content_selector": "article"},
    {"name": "bbc_hindi", "sitemap": "https://www.bbc.com/hindi/sitemap.xml", "content_selector": "article"},
    {"name": "livehindustan", "sitemap": "https://www.livehindustan.com/news-sitemap.xml", "content_selector": "article"},
    {"name": "abplive", "sitemap": "https://www.abplive.com/news-sitemap.xml", "content_selector": "div.abp-story-detail"},
    {"name": "prabhatkhabar", "sitemap": "https://www.prabhatkhabar.com/sitemap-index.xml", "content_selector": "article"},
    {"name": "indiatv", "sitemap": "https://www.indiatv.in/sitemap.xml", "content_selector": "div.content"},
    {"name": "patrika", "sitemap": "https://www.patrika.com/sitemap.xml", "content_selector": 'div[class*="story_story_section"]'},
    {"name": "zeenews", "sitemap": "https://zeenews.india.com/hindi/sitemap.xml", "content_selector": 'div[class*="article_content"]'},
    {"name": "aajtak", "sitemap": "https://www.aajtak.in/rssfeeds/news-sitemap", "content_selector": "article"},
]


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def fetch(url):
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(REQUEST_DELAY_S)
        try:
            r = _session.get(url, timeout=REQUEST_TIMEOUT_S)
        except requests.RequestException:
            continue
        if r.status_code == 200:
            return r.content
        if r.status_code == 429:
            time.sleep(REQUEST_DELAY_S * (2 ** attempt))
            continue
        return None
    log(f"FETCH FAILED after {MAX_RETRIES} attempts: {url}")
    return None


def discover_urls(sitemap_url):
    """Returns None on total discovery failure (network/parse) so callers can
    distinguish "couldn't check" from "checked, nothing new". Some sites
    (patrika, zeenews, prabhatkhabar) nest sitemapindex inside sitemapindex
    more than one level deep -- BFS until every branch bottoms out at a real
    <urlset>, capped at MAX_CHILD_SITEMAPS total sitemap documents fetched
    (not just first-level children) so a deeply nested index can't runaway.

    patrika mislabels an intermediate index level as <urlset> even though its
    <loc> entries are further per-date sitemap files, not articles -- a tag
    check alone can't catch that, so a urlset whose entries mostly end in
    .xml is treated as another index level to recurse into rather than a
    final article list."""
    raw = fetch(sitemap_url)
    if raw is None:
        return None
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        log(f"SKIP unparseable sitemap {sitemap_url}: {e}")
        return None

    def locs(node, ns):
        return [e.text.strip() for e in node.findall(".//s:loc", ns)] if ns else \
               [e.text.strip() for e in node.findall(".//loc")]

    urls = []
    queue = [root]
    fetched = {sitemap_url}
    while queue:
        node = queue.pop(0)
        tag = node.tag.split("}")[-1]
        ns = {"s": node.tag.split("}")[0].strip("{")} if "}" in node.tag else {}
        found = locs(node, ns)
        looks_like_more_sitemaps = found and sum(u.endswith(".xml") for u in found) / len(found) > 0.5
        if tag != "sitemapindex" and not looks_like_more_sitemaps:
            urls.extend(found)
            continue
        for child_url in found:
            if child_url in fetched or len(fetched) >= MAX_CHILD_SITEMAPS:
                continue
            fetched.add(child_url)
            child_raw = fetch(child_url)
            if child_raw is None:
                log(f"SKIP child sitemap (fetch failed): {child_url}")
                continue
            try:
                queue.append(ET.fromstring(child_raw))
            except ET.ParseError as e:
                log(f"SKIP unparseable child sitemap {child_url}: {e}")
    return urls


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


def process_source(source, state_lock, max_articles, max_seconds):
    name = source["name"]
    namespace = f"{NAMESPACE_PREFIX}_{name}"
    with state_lock:
        entry = state_io.load_namespace(STATE_FILE, namespace, {"done_urls": [], "total_words": 0})
        done = set(entry["done_urls"])

    urls = None
    for attempt in range(3):
        urls = discover_urls(source["sitemap"])
        if urls is not None:
            break
        log(f"{name}: discovery attempt {attempt + 1}/3 failed, retrying")
        time.sleep(REQUEST_DELAY_S * (attempt + 1))
    if urls is None:
        log(f"{name}: discovery failed 3/3 times -- skipping this run, "
            f"NOT marking as drained ({len(done)} done so far)")
        return {"name": name, "new_articles": 0, "words": 0, "elapsed_s": 0.0}

    new_urls = [u for u in urls if u not in done]
    log(f"{name}: discovery found {len(urls)} URLs, {len(new_urls)} new")

    out_path = OUT_DIR / f"{name}.txt"
    clean_path = CLEAN_DIR / f"{name}.txt"
    processed = 0
    batch_words = 0
    batch_start = time.time()
    with out_path.open("a", encoding="utf-8") as out_f, clean_path.open("a", encoding="utf-8") as clean_f:
        for url in new_urls:
            if processed >= max_articles or time.time() - batch_start >= max_seconds:
                break

            text = scrape_article(url, source["content_selector"])
            with state_lock:
                if text is None:
                    continue  # transient fetch failure -- leave off done_urls, retry next run
                if not text:
                    entry["done_urls"].append(url)  # confirmed non-article page -- permanent skip
                    state_io.save_namespace(STATE_FILE, namespace, entry)
                    continue

                out_f.write(text + "\n\n")
                out_f.flush()
                cleaned = clean_text(text)
                if cleaned:
                    clean_f.write(cleaned + "\n\n")
                    clean_f.flush()

                words = len(cleaned.split())
                entry["done_urls"].append(url)
                entry["total_words"] += words
                state_io.save_namespace(STATE_FILE, namespace, entry)

            batch_words += words
            processed += 1
            if processed % LOG_EVERY_N_ARTICLES == 0:
                log(f"{name}: {processed}/{len(new_urls)} articles this batch, "
                    f"~{batch_words} words this batch (~{entry['total_words']} total ever)")

    elapsed = time.time() - batch_start
    log(f"{name}: batch done -- {processed} new articles, ~{batch_words} words, "
        f"{elapsed:.0f}s, ~{entry['total_words']} words total ever")
    return {"name": name, "new_articles": processed, "words": batch_words, "elapsed_s": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="*", default=None,
                         help="subset of source names to run (default: all in SOURCES)")
    parser.add_argument("--max-articles", type=int, default=DEFAULT_BATCH_MAX_ARTICLES,
                         help="per-source cap on new articles this batch")
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_BATCH_MAX_SECONDS,
                         help="per-source wall-clock cap this batch")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    sources = [s for s in SOURCES if args.sources is None or s["name"] in args.sources]
    if not sources:
        log(f"No matching sources for --sources {args.sources}")
        return

    state_lock = threading.Lock()  # each source now loads/saves its own namespace, so this
                                    # only serializes ordering of load-then-mutate within a source;
                                    # state_io itself is flock-safe for the on-disk file separately

    log(f"Starting {len(sources)} concurrent source workers: {[s['name'] for s in sources]}")
    run_start = time.time()
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        futures = [pool.submit(process_source, s, state_lock,
                                args.max_articles, args.max_seconds) for s in sources]
        results = [f.result() for f in futures]

    total_elapsed = time.time() - run_start
    total_words = sum(r["words"] for r in results)
    total_articles = sum(r["new_articles"] for r in results)
    words_per_hour = (total_words / total_elapsed) * 3600 if total_elapsed > 0 else 0
    log(f"RUN COMPLETE: {len(sources)} sources, {total_articles} new articles, "
        f"{total_words} words, {total_elapsed:.0f}s elapsed, "
        f"~{words_per_hour:.0f} words/hour aggregate throughput")
    for r in results:
        log(f"  {r['name']}: {r['new_articles']} articles, {r['words']} words, {r['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
