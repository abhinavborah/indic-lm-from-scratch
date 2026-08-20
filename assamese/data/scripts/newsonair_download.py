#!/usr/bin/env python3
"""Downloader for newsonair.gov.in's Assamese regional bulletin PDFs (manual-collection corpus).

News On AIR (Akashvani/All India Radio, Prasar Bharati, Govt of India):
official news-script PDFs for its Guwahati Assamese regional unit, published
daily. Same underlying source as AIKosh's "Assamese News Bulletin - Akashvani"
dataset, but live and open (no account registration needed), unlike AIKosh's
copy which is marked Restricted.

Discovery: bulletins-city/assamese/ lists the most recent bulletins with
direct PDF download links. Real pagination exists via ?page=N (confirmed by
direct check: page=29 reaches mid-May 2026, page=100 reaches Sep 2025, each
page distinct from the last); an earlier version of this script missed
it (tried /page/N/ as a path segment, got 404, wrongly concluded there was
no pagination). Paginates until two consecutive pages add nothing new.

Resumable: assamese/data/.state.json tracks per-PDF-URL completion.
"""

import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

import state_io
from text_clean import clean_text

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
RAW_OUT_DIR = DATA_DIR / "raw" / "newsonair"
CLEAN_OUT_DIR = DATA_DIR / "clean" / "newsonair"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "newsonair"

LISTING_URL = "https://newsonair.gov.in/bulletins-city/assamese/"
USER_AGENT = "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"
REQUEST_TIMEOUT_S = 30
MAX_PAGES = 1000  # safety cap; real depth unknown, confirmed real content past page 100
EMPTY_PAGE_STREAK_TO_STOP = 2
# These site-wide static PDFs sit in the page footer and appear on EVERY
# page regardless of content, discovered when discovery ran to page 500
# with the URL count stuck flat since ~page 440: the stop condition never
# fired because these kept every page looking "non-empty". Filtered by
# filename since they're not scoped to the results table any other way.
STATIC_FOOTER_PDF_NAMES = {
    "citizens-charter.pdf", "citizens-charter-hindi.pdf", "media-coverage.pdf",
    "icc_28_08_2023.pdf", "ptc-guidelines-2024_compressed.pdf", "cep_2021.pdf",
}

BENGALI_LO, BENGALI_HI = 0x0980, 0x09FF
MIN_BENGALI_RATIO = 0.3
DEFAULT_MAX_NEW_ITEMS = 1000


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def load_state():
    return state_io.load_namespace(STATE_FILE, NAMESPACE, {})


def save_state(state):
    state_io.save_namespace(STATE_FILE, NAMESPACE, state)


def bengali_ratio(text):
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    beng = sum(1 for c in alpha if BENGALI_LO <= ord(c) <= BENGALI_HI)
    return beng / len(alpha)


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


def discover_pdf_urls_on_page(page_num, retries=2):
    """Returns None on a fetch failure (after retries), distinct from a
    real empty list, so the caller doesn't mistake a transient timeout for
    genuinely reaching the end of the archive."""
    url = LISTING_URL if page_num == 1 else f"{LISTING_URL}?page={page_num}"
    for attempt in range(retries + 1):
        html = fetch(url)
        if html is not None:
            break
    else:
        return None
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    seen = set()
    for a in soup.select("a[href$='.pdf']"):
        href = a["href"]
        if href.rsplit("/", 1)[-1].lower() in STATIC_FOOTER_PDF_NAMES:
            continue
        if href not in seen:
            seen.add(href)
            urls.append(href)
    return urls


def discover_pdf_urls(state):
    """Paginate via ?page=N until EMPTY_PAGE_STREAK_TO_STOP consecutive pages
    successfully load and have zero PDF links (structurally past the end of
    the archive), or MAX_PAGES is hit. A fetch failure does NOT count toward
    that streak; it's retried within discover_pdf_urls_on_page, and if
    still unreachable the page is just skipped and pagination continues, so
    a transient timeout can never be mistaken for end-of-archive. Also does
    NOT stop just because a page's items are already in state; on a resume
    run, early pages are fully synced but later pages still hold new
    content, and stopping on "nothing new this page" would cut the crawl
    short before reaching it."""
    all_urls = []
    seen = set()
    empty_streak = 0
    for page_num in range(1, MAX_PAGES + 1):
        page_urls = discover_pdf_urls_on_page(page_num)
        if page_urls is None:
            log(f"newsonair: page {page_num} unreachable after retries, skipping")
        elif not page_urls:
            empty_streak += 1
        else:
            empty_streak = 0
            for u in page_urls:
                if u not in seen:
                    seen.add(u)
                    all_urls.append(u)
        if page_num % 10 == 0:
            log(f"newsonair: discovery at page {page_num}, {len(all_urls)} URLs found so far")
        if empty_streak >= EMPTY_PAGE_STREAK_TO_STOP:
            log(f"newsonair: pagination stopped at page {page_num}, {empty_streak} consecutive genuinely-empty pages")
            break
        time.sleep(0.3)
    return all_urls


def extract_pdftotext(pdf_path):
    import subprocess
    r = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def extract_ocr(pdf_path, tmp_dir):
    import subprocess
    out = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    total = int(m.group(1)) if m else 0
    pages = []
    for page in range(1, total + 1):
        prefix = tmp_dir / f"pg{page}"
        subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", "200",
                         "-png", str(pdf_path), str(prefix)], capture_output=True)
        pngs = sorted(tmp_dir.glob(f"pg{page}-*.png")) or sorted(tmp_dir.glob(f"pg{page}.png"))
        if not pngs:
            continue
        r = subprocess.run(["tesseract", str(pngs[0]), "-", "-l", "ben"], capture_output=True, text=True)
        pages.append(r.stdout if r.returncode == 0 else "")
        for p in pngs:
            p.unlink(missing_ok=True)
    return "\n\n".join(pages)


def slugify(url):
    name = url.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()[:60]


def process_pdf(pdf_url, state, tmp_dir):
    if pdf_url in state:
        return False

    body = fetch(pdf_url)
    if body is None or not body.startswith(b"%PDF"):
        state[pdf_url] = {"status": "download_failed"}
        save_state(state)
        return True

    slug = slugify(pdf_url)
    pdf_path = tmp_dir / f"{slug}.pdf"
    pdf_path.write_bytes(body)

    text = extract_pdftotext(pdf_path)
    used_ocr = bengali_ratio(text) < MIN_BENGALI_RATIO
    if used_ocr:
        text = extract_ocr(pdf_path, tmp_dir)
    raw_text = correct_ra(text)
    cleaned, _dropped = clean_text(raw_text)
    pdf_path.unlink(missing_ok=True)

    with (RAW_OUT_DIR / f"{slug}.txt").open("w", encoding="utf-8") as f:
        f.write(raw_text + "\n\n")
    with (CLEAN_OUT_DIR / f"{slug}.txt").open("w", encoding="utf-8") as f:
        f.write(cleaned + "\n\n")

    words = len(cleaned.split())
    state[pdf_url] = {"status": "done", "used_ocr": used_ocr, "words": words}
    save_state(state)
    log(f"newsonair {slug}: {'OCR' if used_ocr else 'pdftotext'}, ~{words} words")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-items", type=int, default=DEFAULT_MAX_NEW_ITEMS)
    args = parser.parse_args()

    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = RAW_OUT_DIR / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    state = load_state()
    urls = discover_pdf_urls(state)
    log(f"newsonair: {len(urls)} PDF URLs found across all paginated listing pages")

    new_count = 0
    for url in urls:
        if new_count >= args.max_new_items:
            break
        if process_pdf(url, state, tmp_dir):
            new_count += 1
        time.sleep(0.5)

    log(f"newsonair BATCH COMPLETE: {new_count} new PDFs processed this run")


if __name__ == "__main__":
    main()
