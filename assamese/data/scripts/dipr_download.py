#!/usr/bin/env python3
"""Downloader for DIPR Assam's Assamese-language press releases (manual-collection corpus).

Directorate of Information and Public Relations, Govt. of Assam
(dipr.assam.gov.in), official state government communications, a subset
explicitly titled "...Press Release Assamese No N...". Each release is a
small born-digital PDF, not scrapeable HTML.

Two-step discovery: paginate the press-release listing (~161 pages, 10
items/page), filter titles containing "Assamese" client-side (avoids
fetching every detail page), then visit each matching detail page to pull
its actual PDF URL.

Resumable: assamese/data/.state.json tracks per-document-slug completion.
Safe to interrupt (Ctrl-C) and rerun.
"""

import re
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

import state_io
from text_clean import clean_text

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
RAW_OUT_DIR = DATA_DIR / "raw" / "dipr"
CLEAN_OUT_DIR = DATA_DIR / "clean" / "dipr"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "dipr"

BASE = "https://dipr.assam.gov.in"
LISTING_URL = BASE + "/documents/press-release"
TOTAL_PAGES = 161
USER_AGENT = "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"
REQUEST_TIMEOUT_S = 30

BENGALI_LO, BENGALI_HI = 0x0980, 0x09FF
MIN_BENGALI_RATIO = 0.3
OCR_DPI = 200
DEFAULT_MAX_NEW_ITEMS = 200


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


def discover_assamese_items(page):
    url = LISTING_URL if page == 0 else f"{LISTING_URL}?page={page}"
    html = fetch(url)
    if html is None:
        return []
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for a in soup.select("a[href*='/documents-detail/']"):
        title = a.get_text(strip=True)
        if "assamese" in title.lower():
            items.append((title, BASE + a["href"] if a["href"].startswith("/") else a["href"]))
    return items


def find_pdf_url(detail_url):
    html = fetch(detail_url)
    if html is None:
        return None
    soup = BeautifulSoup(html, "html.parser")
    a = soup.select_one("a[href$='.pdf']")
    return a["href"] if a else None


def slugify(title):
    return re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()[:60]


def page_count(pdf_path):
    out = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    return int(m.group(1)) if m else 0


def extract_pdftotext(pdf_path):
    r = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def extract_ocr(pdf_path, tmp_dir):
    total = page_count(pdf_path)
    pages = []
    for page in range(1, total + 1):
        prefix = tmp_dir / f"pg{page}"
        subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(OCR_DPI),
                         "-png", str(pdf_path), str(prefix)], capture_output=True)
        pngs = sorted(tmp_dir.glob(f"pg{page}-*.png")) or sorted(tmp_dir.glob(f"pg{page}.png"))
        if not pngs:
            continue
        r = subprocess.run(["tesseract", str(pngs[0]), "-", "-l", "ben"], capture_output=True, text=True)
        pages.append(r.stdout if r.returncode == 0 else "")
        for p in pngs:
            p.unlink(missing_ok=True)
    return "\n\n".join(pages)


def process_item(title, detail_url, state, tmp_dir):
    slug = slugify(title)
    if slug in state:
        return

    pdf_url = find_pdf_url(detail_url)
    if pdf_url is None:
        state[slug] = {"status": "no_pdf", "title": title}
        save_state(state)
        return

    pdf_path = tmp_dir / f"{slug}.pdf"
    body = fetch(pdf_url)
    if body is None or not body.startswith(b"%PDF"):
        state[slug] = {"status": "download_failed", "title": title}
        save_state(state)
        return
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
    state[slug] = {"status": "done", "title": title, "used_ocr": used_ocr, "words": words}
    save_state(state)
    log(f"{title}: {'OCR' if used_ocr else 'pdftotext'}, ~{words} words")


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
    last_page = state.get("_last_listing_page", -1) + 1
    new_count = 0

    for page in range(last_page, TOTAL_PAGES):
        items = discover_assamese_items(page)
        for title, detail_url in items:
            if new_count >= args.max_new_items:
                break
            slug = slugify(title)
            if slug in state:
                continue
            process_item(title, detail_url, state, tmp_dir)
            new_count += 1
            time.sleep(0.5)
        state["_last_listing_page"] = page
        save_state(state)
        if new_count >= args.max_new_items:
            log(f"BATCH COMPLETE: {new_count} new items processed, stopped at listing page {page}")
            return
        time.sleep(0.5)

    log(f"dipr: reached end of listing ({TOTAL_PAGES} pages), {new_count} new items this run")


if __name__ == "__main__":
    main()
