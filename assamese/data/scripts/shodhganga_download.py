#!/usr/bin/env python3
"""Downloader for Shodhganga's Assamese-language PhD theses (manual-collection corpus).

Shodhganga (shodhganga.inflibnet.ac.in), INFLIBNET's national repository of
Indian PhD theses. 610 items are tagged `language=Assamese` (real academic
theses in Assamese, mostly literature/linguistics/culture, from Gauhati
University and other Assam institutions). Each thesis is split into 10-14
chapter/section PDFs. Confirmed via a direct check: files download without
login, and Shodhganga states its items are licensed CC BY-NC 4.0, an
explicit open license, cleaner than any other manual source in this
project. PDFs are consistently modest in size (1.3-1.7MB for ~250 page
theses), consistent with born-digital text rather than scanned images, but
the pdftotext-then-OCR-fallback pattern is kept anyway since this hasn't
been verified for every thesis.

Discovery: DSpace's simple-search with a language filter, paginated.
Resumable: assamese/data/.state.json tracks per-PDF-URL completion.
"""

import re
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

import state_io
from text_clean import clean_text

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
RAW_OUT_DIR = DATA_DIR / "raw" / "shodhganga"
CLEAN_OUT_DIR = DATA_DIR / "clean" / "shodhganga"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "shodhganga"

BASE = "https://shodhganga.inflibnet.ac.in"
SEARCH_URL = BASE + "/simple-search"
USER_AGENT = "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"
REQUEST_TIMEOUT_S = 30
RESULTS_PER_PAGE = 50

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


def discover_thesis_handles(start):
    params = {
        "query": "",
        "filter_field_1": "language",
        "filter_type_1": "equals",
        "filter_value_1": "Assamese",
        "rpp": RESULTS_PER_PAGE,
        "start": start,
    }
    url = SEARCH_URL + "?" + urllib.parse.urlencode(params)
    body = fetch(url)
    if body is None:
        return []
    soup = BeautifulSoup(body, "html.parser")
    handles = []
    for a in soup.select("a[href*='/handle/10603/']"):
        href = a["href"]
        if re.search(r"/handle/10603/\d+$", href):
            handles.append(BASE + href if href.startswith("/") else href)
    # de-dup while preserving order (title cell links to the same handle as row links)
    seen = set()
    out = []
    for h in handles:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def discover_pdf_urls(handle_url):
    body = fetch(handle_url)
    if body is None:
        return []
    soup = BeautifulSoup(body, "html.parser")
    return [BASE + a["href"] if a["href"].startswith("/") else a["href"]
            for a in soup.select("a[href*='/bitstream/']") if a["href"].lower().endswith(".pdf")]


def extract_pdftotext(pdf_path):
    r = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def extract_ocr(pdf_path, tmp_dir):
    out = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    total = int(m.group(1)) if m else 0
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


def slugify(pdf_url):
    tail = urllib.parse.unquote(pdf_url).rsplit("/", 3)
    thesis_id = tail[-3] if len(tail) >= 3 else "unknown"
    fname = re.sub(r"[^a-zA-Z0-9]+", "_", tail[-1]).strip("_").lower()[:40]
    return f"{thesis_id}_{fname}"


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
    log(f"shodhganga {slug}: {'OCR' if used_ocr else 'pdftotext'}, ~{words} words")
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
    start = state.get("_last_search_start", 0)
    new_count = 0
    words_total = 0

    while new_count < args.max_new_items:
        handles = discover_thesis_handles(start)
        if not handles:
            log("shodhganga: reached end of search results")
            break
        for handle_url in handles:
            if new_count >= args.max_new_items:
                break
            pdf_urls = discover_pdf_urls(handle_url)
            for pdf_url in pdf_urls:
                if new_count >= args.max_new_items:
                    break
                attempted = process_pdf(pdf_url, state, tmp_dir)
                if attempted:
                    new_count += 1
                    words_total += state.get(pdf_url, {}).get("words", 0)
                time.sleep(0.3)
        start += RESULTS_PER_PAGE
        state["_last_search_start"] = start
        save_state(state)

    log(f"shodhganga BATCH COMPLETE: {new_count} new PDFs processed, ~{words_total} words this batch")


if __name__ == "__main__":
    main()
