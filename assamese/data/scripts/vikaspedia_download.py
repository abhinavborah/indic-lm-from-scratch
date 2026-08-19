#!/usr/bin/env python3
"""Downloader for as.vikaspedia.in (manual-collection corpus).

Vikaspedia (C-DAC / MeitY, Government of India) is a multilingual public-
policy knowledge portal. Its article pages are a Next.js app that appears
JS-rendered (a plain HTML fetch of the page shell shows no body text), but
the real article body ships inline as JSON in a `__NEXT_DATA__` script tag
regardless -- `props.pageProps.ssrPageContent.content` holds the actual
HTML-formatted article body for real leaf articles. `ssrPageData` (rather
than `ssrPageContent`) or a non-null `ssrContentStatus` (e.g. 301) means the
URL is a folder/category page or stale/moved -- skipped, not an article.
Confirmed empirically via direct curl before writing this script; no
browser automation needed.

Discovery: as.vikaspedia.in/sitemap.xml lists every article URL directly
(no crawling needed).

Resumable: assamese/data/.state.json tracks per-URL completion.
"""

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

import state_io
from text_clean import clean_text, is_likely_assamese

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
RAW_OUT_DIR = DATA_DIR / "raw" / "vikaspedia"
CLEAN_OUT_DIR = DATA_DIR / "clean" / "vikaspedia"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "vikaspedia"

SITEMAP_URL = "https://as.vikaspedia.in/sitemap.xml"
USER_AGENT = "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"
REQUEST_TIMEOUT_S = 30
REQUEST_DELAY_S = 0.5
DEFAULT_MAX_NEW_ITEMS = 300
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


def fetch(url):
    # sitemap <loc> entries and the article slugs derived from them are raw
    # UTF-8 (Assamese script), not percent-encoded -- urllib.request refuses
    # a non-ASCII URL outright, so encode the path/query safely first.
    safe_url = urllib.parse.quote(url, safe=":/?=&")
    req = urllib.request.Request(safe_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return resp.read()
    except Exception as e:
        log(f"fetch failed {url}: {e}")
        return None


def discover_urls():
    """sitemap.xml is huge (MBs) -- regex over raw bytes rather than a full
    XML DOM parse, streaming-friendly and avoids pulling in an XML lib."""
    body = fetch(SITEMAP_URL)
    if body is None:
        return []
    return re.findall(rb"<loc>([^<]+)</loc>", body)


def extract_article_text(html_bytes):
    """Returns article body text, or None if this URL isn't a real leaf
    article (folder page / stale redirect per ssrContentStatus)."""
    m = NEXT_DATA_RE.search(html_bytes.decode("utf-8", errors="ignore"))
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    props = data.get("props", {}).get("pageProps", {})
    if props.get("ssrContentStatus"):
        return None  # redirect/moved
    page_content = props.get("ssrPageContent")
    if not page_content or not page_content.get("content"):
        return None  # folder/category page, no article body
    html_body = page_content["content"]
    return BeautifulSoup(html_body, "html.parser").get_text(separator="\n", strip=True)


def process_url(url_bytes, state):
    url = url_bytes.decode("utf-8")
    if url in state:
        return False

    html = fetch(url)
    if html is None:
        return False  # transient -- leave off state, retry next run
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

    # non-ASCII (Assamese-script) slugs collapse to near-identical ASCII
    # residue after stripping -- append a URL hash so distinct articles
    # never collide on the same filename.
    url_hash = hashlib.blake2b(url.encode(), digest_size=4).hexdigest()
    ascii_slug = re.sub(r"[^a-zA-Z0-9]+", "_", url.split("vikaspedia.in")[-1]).strip("_").lower()[:60]
    slug = f"{ascii_slug}_{url_hash}"
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
    urls = discover_urls()
    log(f"vikaspedia (as): {len(urls)} URLs in sitemap")

    new_count = 0
    words_total = 0
    for url_bytes in urls:
        if url_bytes.decode("utf-8") in state:
            continue
        if new_count >= args.max_new_items:
            break
        attempted = process_url(url_bytes, state)
        if attempted:
            new_count += 1
            entry = state.get(url_bytes.decode("utf-8"), {})
            words_total += entry.get("words", 0)
            if new_count % LOG_EVERY_N_ITEMS == 0:
                log(f"vikaspedia (as): {new_count} new URLs processed this batch, ~{words_total} words")
        time.sleep(REQUEST_DELAY_S)

    log(f"BATCH COMPLETE: {new_count} new URLs processed, ~{words_total} words this batch")


if __name__ == "__main__":
    main()
