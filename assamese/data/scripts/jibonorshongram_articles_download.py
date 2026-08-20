#!/usr/bin/env python3
"""Downloader for jibonorshongram.in's general articles via Wayback Machine (manual-collection corpus).

Separate from `jibonorshongram_download.py` (the ~78-book ebook listing).
The Wayback CDX index holds 2,275+ other canonical URLs from this site --
it was a general Assamese lifestyle/blog/news-aggregator, not just an ebook
page. Manual spot-check found mixed quality: job-alert posts are thin
boilerplate (the site's own footer admits job listings are aggregated from
elsewhere -- same third-party-republish caveat as the ebook page), but
story-category ("কাহিনী") posts are real,
original, full-length prose. No per-category quality gate is applied here;
clean_text's purity filter plus a minimum-word floor does the filtering.

Discovery: web.archive.org/cdx/search/cdx, filtered to 200-status HTML pages,
excluding known non-article paths (feed/embed/share-links/wp-admin/tag/
category/author/pagination/the ebook listing itself).

Resumable: assamese/data/.state.json tracks per-URL completion.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

import state_io
from text_clean import clean_text

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
RAW_OUT_DIR = DATA_DIR / "raw" / "jibonorshongram_articles"
CLEAN_OUT_DIR = DATA_DIR / "clean" / "jibonorshongram_articles"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "jibonorshongram_articles"

CDX_URL = "http://web.archive.org/cdx/search/cdx"
USER_AGENT = "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"
REQUEST_TIMEOUT_S = 30
MIN_WORDS = 30  # drop stub/boilerplate-only pages (thin job-alert posts etc.)
DEFAULT_MAX_NEW_ITEMS = 100

SKIP_PATTERN = re.compile(
    r"(/feed/?$|/embed/?$|\?share=|/JibonorShongram\.in$|/wp-|/tag/|/category/|/author/|/page/[0-9]|assamese-ebook)"
)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def load_state():
    return state_io.load_namespace(STATE_FILE, NAMESPACE, {})


def save_state(state):
    state_io.save_namespace(STATE_FILE, NAMESPACE, state)


def correct_ra(text):
    return text.replace("র", "ৰ")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return resp.read()
    except Exception as e:
        log(f"fetch failed {url}: {e}")
        return None


def discover_urls():
    params = {
        "url": "jibonorshongram.in*",
        "output": "json",
        "collapse": "urlkey",
        "limit": "20000",
        "fl": "original,timestamp,statuscode,mimetype",
    }
    body = fetch(CDX_URL + "?" + urllib.parse.urlencode(params))
    if body is None:
        return {}
    rows = json.loads(body)[1:]
    canon = {}
    for orig, ts, code, mime in rows:
        if code != "200" or "html" not in (mime or ""):
            continue
        if SKIP_PATTERN.search(orig):
            continue
        key = orig.split("?")[0].rstrip("/")
        canon[key] = ts  # later (sorted) capture wins, closer to final content
    return canon


def slugify(url):
    tail = urllib.parse.unquote(url).rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", tail).strip("_").lower()[:50]
    url_hash = __import__("hashlib").blake2b(url.encode(), digest_size=4).hexdigest()
    return f"{slug}_{url_hash}" if slug else url_hash


def process_url(url, ts, state):
    if url in state:
        return False

    wayback_url = f"https://web.archive.org/web/{ts}/{url}"
    html = fetch(wayback_url)
    if html is None:
        return False  # transient -- leave off state, retry next run

    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("article")
    text = container.get_text(separator="\n", strip=True) if container else ""

    if not text or len(text.split()) < MIN_WORDS:
        state[url] = {"status": "too_short_or_no_article"}
        save_state(state)
        return True

    raw_text = correct_ra(text)
    cleaned, _dropped = clean_text(raw_text)
    if not cleaned or len(cleaned.split()) < MIN_WORDS:
        state[url] = {"status": "filtered_below_min_words"}
        save_state(state)
        return True

    slug = slugify(url)
    with (RAW_OUT_DIR / f"{slug}.txt").open("w", encoding="utf-8") as f:
        f.write(raw_text + "\n\n")
    with (CLEAN_OUT_DIR / f"{slug}.txt").open("w", encoding="utf-8") as f:
        f.write(cleaned + "\n\n")

    words = len(cleaned.split())
    state[url] = {"status": "done", "words": words}
    save_state(state)
    log(f"jibonorshongram_articles {slug}: ~{words} words")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-items", type=int, default=DEFAULT_MAX_NEW_ITEMS)
    args = parser.parse_args()

    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    urls = discover_urls()
    log(f"jibonorshongram_articles: {len(urls)} candidate URLs from Wayback CDX")

    new_count = 0
    words_total = 0
    for url, ts in urls.items():
        if new_count >= args.max_new_items:
            break
        if url in state:
            continue
        if process_url(url, ts, state):
            new_count += 1
            words_total += state.get(url, {}).get("words", 0)
        time.sleep(0.5)

    log(f"jibonorshongram_articles BATCH COMPLETE: {new_count} new URLs processed, ~{words_total} words this batch")


if __name__ == "__main__":
    main()
