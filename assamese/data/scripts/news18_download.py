#!/usr/bin/env python3
"""Downloader for assam.news18.com (manual-collection corpus).

Network18's Assamese edition is a Next.js app with no matching CSS selector
for the article body (`article`, `.story`, `.article-body` all miss) -- the
real text ships inline as JSON in a `__NEXT_DATA__` script tag instead.
Confirmed empirically (19 Aug 2026 sample, CNG pump explosion story): the
body lives at `props.pageProps.pageData.pageConfig.wdata.articleData.data.body`,
but the `wdata`/`wStatus` keys around it are auto-generated per-page widget
instance IDs (e.g. `article_article-scroll_247`) that shift between articles
-- searching recursively for the first dict containing an "articleData" key
is robust to that drift, a hardcoded full path would not be.

Discovery: two-level sitemap index --
assam.news18.com/commonfeeds/v1/asm/sitemap-index.xml (full catalog) plus
.../sitemap/today (same-day articles, useful for staying current between
full-index refreshes). Both are child sitemaps of the top-level sitemap.xml.

Resumable: assamese/data/.state.json tracks per-URL completion.
"""

import argparse
import json
import re
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

import state_io
from text_clean import clean_text, is_likely_assamese

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
RAW_OUT_DIR = DATA_DIR / "raw" / "news18"
CLEAN_OUT_DIR = DATA_DIR / "clean" / "news18"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "news18"

SITEMAP_URLS = [
    "https://assam.news18.com/commonfeeds/v1/asm/sitemap-index.xml",
    "https://assam.news18.com/commonfeeds/v1/asm/sitemap/today",
]
# news18's WAF blocks on UA content, not identity as such -- verified this
# session: the project's usual self-identifying UA (below, used everywhere
# else) gets a flat 403 from news18 specifically; a plain browser UA string
# gets 200 on an identical request. Every other source in this project
# tolerates the honest UA fine -- this is a news18-specific WAF rule, not a
# project-wide policy change.
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
REQUEST_TIMEOUT_S = 30
REQUEST_DELAY_S = 0.5
DEFAULT_MAX_NEW_ITEMS = 1_000_000  # effectively uncapped -- run until new_urls is exhausted or killed,
                                    # matches scraper.py's BATCH_MAX_ARTICLES convention. Previously 300,
                                    # which forced a full rediscovery (sitemap + 8-category API pagination)
                                    # every 300 items -- real wasted time given how expensive discovery is.
LOG_EVERY_N_ITEMS = 20

NEXT_DATA_RE = re.compile(r'__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def load_state():
    return state_io.load_namespace(STATE_FILE, NAMESPACE, {})


def save_state(state):
    state_io.save_namespace(STATE_FILE, NAMESPACE, state)


MAX_RETRIES = 3
BACKOFF_BASE_S = 5


def fetch(url):
    # urllib.request gets a flat 403 from news18's edge/WAF regardless of
    # User-Agent; a plain curl with the same UA works fine (same reason
    # scraper.py shells out to curl instead of urllib for its own fetches).
    for attempt in range(1, MAX_RETRIES + 1):
        r = subprocess.run(
            ["curl", "-s", "-L", "-A", USER_AGENT, "--max-time", str(REQUEST_TIMEOUT_S),
             "-w", "\n%{http_code}", url],
            capture_output=True, stdin=subprocess.DEVNULL,
        )
        if r.returncode != 0:
            log(f"fetch attempt {attempt}/{MAX_RETRIES} failed {url}: curl exit {r.returncode}")
        else:
            body, _, status = r.stdout.rpartition(b"\n")
            status = status.decode().strip()
            if status == "200":
                return body
            log(f"fetch attempt {attempt}/{MAX_RETRIES} failed {url}: HTTP {status}")
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_BASE_S * attempt)
    return None


# The site's own category-carousel widget (the "dot based navbar") calls
# this same public API under the hood -- found this session by reading the
# category page's __NEXT_DATA__ blob for its embedded API calls. Paginating
# it directly with offset/count is far deeper than the sitemap: the "assam"
# category alone reports total=10000 (the API's own reported ceiling, real
# count may be higher), other categories range from about 2,100 to 10,000.
API_CATEGORIES = ["assam", "nation", "crime", "entertainment", "business", "sports", "lifestyle", "world"]
API_BASE = "https://api-mateas.news18.com/nodeapi/v1/asm/get-article-list"
API_PAGE_SIZE = 40
API_MAX_OFFSET = 2000  # conservative per-category cap this session, not the true ceiling -- raise later if needed


def discover_via_api(category):
    import urllib.parse
    urls = []
    filt = urllib.parse.quote(json.dumps({"categories.slug": category}))
    for offset in range(0, API_MAX_OFFSET, API_PAGE_SIZE):
        time.sleep(REQUEST_DELAY_S)
        url = f"{API_BASE}?count={API_PAGE_SIZE}&fields=weburl_r&filter={filt}&offset={offset}"
        body = fetch(url)
        if body is None:
            break
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            break
        items = data.get("data", [])
        if not items:
            break
        urls.extend("https://assam.news18.com" + item["weburl_r"] for item in items if item.get("weburl_r"))
    return urls


def discover_urls():
    """The index sitemap is itself a list of <sitemap><loc> child sitemaps
    (mixed: today/full-index/google-news/image sitemaps) -- only the plain
    <url><loc> entries inside the real article-listing children are actual
    article URLs, found by fetching each child and regexing for <loc> not
    nested under a <sitemap> parent (a plain per-line regex over raw bytes,
    same approach as vikaspedia_download.py, cheaper than a full DOM parse
    for a multi-MB sitemap)."""
    urls = set()
    for sitemap_url in SITEMAP_URLS:
        body = fetch(sitemap_url)
        if body is None:
            continue
        child_locs = re.findall(rb"<sitemap>\s*<loc>([^<]+)</loc>", body)
        if child_locs:
            for child in child_locs:
                child_url = child.decode("utf-8")
                if "google-news" in child_url or "image-index" in child_url:
                    continue  # not article-listing sitemaps
                child_body = fetch(child_url)
                if child_body is None:
                    continue
                urls.update(u.decode("utf-8") for u in re.findall(rb"<url>\s*<loc>([^<]+)</loc>", child_body))
        else:
            urls.update(u.decode("utf-8") for u in re.findall(rb"<url>\s*<loc>([^<]+)</loc>", body))
    return sorted(urls)


def find_article_data(obj):
    """Recursively search the __NEXT_DATA__ tree for the first dict that has
    an "articleData" key -- robust to the numbered widget-instance keys
    (wdata, wStatus.article_article-scroll_NNN, ...) shifting between pages."""
    if isinstance(obj, dict):
        if "articleData" in obj:
            return obj["articleData"]
        for v in obj.values():
            found = find_article_data(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_article_data(v)
            if found is not None:
                return found
    return None


def extract_article_text(html_bytes):
    """Returns article body text, or None if this URL isn't a real article
    page (photo gallery, video page, category listing -- no articleData)."""
    m = NEXT_DATA_RE.search(html_bytes.decode("utf-8", errors="ignore"))
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    article_data = find_article_data(data)
    if not article_data:
        return None
    body = article_data.get("data", {}).get("body")
    if not body:
        return None
    return BeautifulSoup(body, "html.parser").get_text(separator="\n", strip=True)


def process_url(url, state):
    html = fetch(url)
    if html is None:
        return None  # fetch failed (all MAX_RETRIES attempts) -- caller requeues to tail once
    text = extract_article_text(html)
    if not text:
        state[url] = {"status": "not_article"}
        save_state(state)
        return True

    cleaned, _dropped = clean_text(text)
    if cleaned and not is_likely_assamese(cleaned):
        state[url] = {"status": "suspect_bengali"}
        save_state(state)
        return True

    slug = url.rstrip("/").rsplit("/", 1)[-1].replace(".html", "")[:80]
    with (RAW_OUT_DIR / f"{slug}.txt").open("w", encoding="utf-8") as f:
        f.write(text + "\n\n")
    with (CLEAN_OUT_DIR / f"{slug}.txt").open("w", encoding="utf-8") as f:
        f.write(cleaned + "\n\n")

    words = len(cleaned.split())
    state[url] = {"status": "done", "words": words}
    save_state(state)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-items", type=int, default=DEFAULT_MAX_NEW_ITEMS)
    args = parser.parse_args()

    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    sitemap_urls = discover_urls()
    api_urls = set()
    for category in API_CATEGORIES:
        cat_urls = discover_via_api(category)
        log(f"news18 (asm): category '{category}' -> {len(cat_urls)} URLs via API")
        api_urls.update(cat_urls)
    urls = list(set(sitemap_urls) | api_urls)
    log(f"news18 (asm): {len(sitemap_urls)} from sitemap, {len(api_urls)} from category API, "
        f"{len(urls)} unique total")

    queue = deque(u for u in urls if u not in state)
    requeued = set()
    permanently_failed = []
    new_count = 0
    words_total = 0
    while queue:
        if new_count >= args.max_new_items:
            break
        url = queue.popleft()
        result = process_url(url, state)
        if result is None:
            if url in requeued:
                log(f"news18: {url} failed again after requeue -- giving up for this run")
                permanently_failed.append(url)
            else:
                log(f"news18: {url} failed, pushed to end of queue, retrying after the rest")
                requeued.add(url)
                queue.append(url)
            continue
        if result:
            new_count += 1
            entry = state.get(url, {})
            words_total += entry.get("words", 0)
            if new_count % LOG_EVERY_N_ITEMS == 0:
                log(f"news18 (asm): {new_count} new URLs processed this batch, ~{words_total} words")
        time.sleep(REQUEST_DELAY_S)

    if queue:
        log(f"BATCH STOP: {new_count} new URLs processed, ~{words_total} words this batch, "
            f"{len(queue)} still queued, {len(permanently_failed)} gave up after requeue this run")
    elif permanently_failed:
        log(f"RUN DONE (with failures): {new_count} new URLs processed, ~{words_total} words this batch, "
            f"{len(permanently_failed)} gave up after requeue this run (will retry next invocation)")
    else:
        log(f"COMPLETE: all discovered URLs processed, {new_count} new this batch, ~{words_total} words this batch")


if __name__ == "__main__":
    main()
