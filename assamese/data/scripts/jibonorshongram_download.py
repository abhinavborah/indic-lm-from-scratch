#!/usr/bin/env python3
"""Downloader for jibonorshongram.in's Assamese e-book listing (manual-collection corpus).

Live site is unreachable; the book listing is pulled from a Wayback Machine
snapshot instead (~100 titles: novels and poetry by named Assamese authors,
each linked as a Google Drive-hosted PDF). This source is a third-party
e-book aggregator, not a licensed publisher: the listing page's own
disclaimer states these are republished without confirmed author/publisher
permission. Included per explicit user decision after the copyright
exposure was flagged; report/SOURCES.md must describe this source
accurately, not as a licensed release.

Per book: download the PDF from its Google Drive link (handling Drive's
large-file "can't scan for viruses" interstitial), then per-page try
pdftotext first, falling back to Tesseract (`ben` model) when the Bengali-
script density is too low, the same validation/fallback shape as
ocr_pipeline.py, reused here since these PDFs may mix born-digital and
scanned-image pages.

Resumable: assamese/data/.state.json tracks per-book last completed page.
Safe to interrupt (Ctrl-C) and rerun; already-finished pages are skipped.
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
RAW_OUT_DIR = DATA_DIR / "raw" / "jibonorshongram"
CLEAN_OUT_DIR = DATA_DIR / "clean" / "jibonorshongram"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "jibonorshongram"

LISTING_URL = "https://web.archive.org/web/20240718035739/https://jibonorshongram.in/assamese-ebook/"
USER_AGENT = "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"

BENGALI_LO, BENGALI_HI = 0x0980, 0x09FF
MIN_BENGALI_RATIO = 0.3
OCR_DPI = 200
DOWNLOAD_TIMEOUT_S = 60


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


def slugify(title):
    return re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()[:60]


def discover_books():
    """Parse the Wayback-archived listing page for (title, drive_file_id) pairs.
    Google Drive share links repeat across sections (same book cross-listed) --
    dedup by file_id, first title seen wins."""
    req = urllib.request.Request(LISTING_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read()
    soup = BeautifulSoup(html, "html.parser")

    seen_ids = set()
    books = []
    for a in soup.select("table a[href*='drive.google.com/file/d/']"):
        m = re.search(r"/file/d/([^/]+)/", a["href"])
        if not m:
            continue
        file_id = m.group(1)
        if file_id in seen_ids:
            continue
        seen_ids.add(file_id)
        title = a.get_text(strip=True).lstrip("📖").strip()
        books.append((title, file_id))
    return books


def download_drive_pdf(file_id, out_path):
    """Google Drive shows an HTML interstitial ("can't scan this file for
    viruses") instead of the file for anything past a small size threshold --
    detected by checking the response content-type, then retried with the
    confirm token scraped from that interstitial page."""
    base = f"https://drive.google.com/uc?export=download&id={file_id}"
    req = urllib.request.Request(base, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp:
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read()
    except Exception as e:
        log(f"download failed for {file_id}: {e}")
        return False

    if "text/html" in content_type:
        m = re.search(r'confirm=([0-9A-Za-z_-]+)', body.decode("utf-8", errors="ignore"))
        if not m:
            log(f"SKIP {file_id}: Drive interstitial with no confirm token (quota-limited or private)")
            return False
        confirm_url = f"{base}&confirm={m.group(1)}"
        req = urllib.request.Request(confirm_url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp:
                body = resp.read()
        except Exception as e:
            log(f"confirmed download failed for {file_id}: {e}")
            return False

    if not body.startswith(b"%PDF"):
        log(f"SKIP {file_id}: response isn't a PDF (deleted/private file)")
        return False
    out_path.write_bytes(body)
    return True


def page_count(pdf_path):
    out = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    return int(m.group(1)) if m else 0


def extract_page_pdftotext(pdf_path, page_num):
    r = subprocess.run(
        ["pdftotext", "-f", str(page_num), "-l", str(page_num), "-layout", str(pdf_path), "-"],
        capture_output=True, text=True,
    )
    return r.stdout if r.returncode == 0 else ""


def extract_page_ocr(pdf_path, page_num, tmp_dir):
    prefix = tmp_dir / f"p{page_num}"
    subprocess.run(
        ["pdftoppm", "-f", str(page_num), "-l", str(page_num), "-r", str(OCR_DPI),
         "-png", str(pdf_path), str(prefix)],
        capture_output=True,
    )
    pngs = sorted(tmp_dir.glob(f"p{page_num}-*.png")) or sorted(tmp_dir.glob(f"p{page_num}.png"))
    if not pngs:
        return ""
    png = pngs[0]
    r = subprocess.run(["tesseract", str(png), "-", "-l", "ben"], capture_output=True, text=True)
    for p in pngs:
        p.unlink(missing_ok=True)
    return r.stdout if r.returncode == 0 else ""


def process_book(title, file_id, state, tmp_dir):
    slug = slugify(title) or file_id[:12]
    entry = state.setdefault(file_id, {
        "title": title, "slug": slug, "downloaded": False,
        "last_page": 0, "total_pages": None, "ocr_pages": 0, "text_pages": 0,
    })
    pdf_path = tmp_dir / f"{file_id}.pdf"

    if not entry["downloaded"]:
        if not download_drive_pdf(file_id, pdf_path):
            entry["downloaded"] = "failed"
            save_state(state)
            return
        entry["downloaded"] = True
        save_state(state)
    elif entry["downloaded"] == "failed" or entry["total_pages"] == 0:
        return
    else:
        if not pdf_path.exists():
            if not download_drive_pdf(file_id, pdf_path):
                return

    raw_path = RAW_OUT_DIR / f"{slug}.txt"
    clean_path = CLEAN_OUT_DIR / f"{slug}.txt"
    total = entry["total_pages"] or page_count(pdf_path)
    entry["total_pages"] = total
    if total == 0:
        log(f"SKIP {title}: pdfinfo reported 0 pages")
        pdf_path.unlink(missing_ok=True)
        save_state(state)
        return
    if entry["last_page"] >= total:
        pdf_path.unlink(missing_ok=True)
        return

    mode = "a" if entry["last_page"] > 0 else "w"
    with raw_path.open(mode, encoding="utf-8") as raw_f, \
         clean_path.open(mode, encoding="utf-8") as clean_f:
        for page in range(entry["last_page"] + 1, total + 1):
            text = extract_page_pdftotext(pdf_path, page)
            used_ocr = bengali_ratio(text) < MIN_BENGALI_RATIO
            if used_ocr:
                text = extract_page_ocr(pdf_path, page, tmp_dir)
            raw_text = correct_ra(text)
            cleaned, _dropped = clean_text(raw_text)

            raw_f.write(raw_text + "\n\n")
            raw_f.flush()
            clean_f.write(cleaned + "\n\n")
            clean_f.flush()

            entry["last_page"] = page
            entry["ocr_pages" if used_ocr else "text_pages"] += 1
            save_state(state)

    log(f"DONE {title}: {total} pages ({entry['ocr_pages']} OCR, {entry['text_pages']} text-layer)")
    pdf_path.unlink(missing_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-items", type=int, default=1_000_000)
    args = parser.parse_args()

    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = RAW_OUT_DIR / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    state = load_state()
    books = discover_books()
    log(f"jibonorshongram: {len(books)} books discovered in listing")

    new_count = 0
    for title, file_id in books:
        if new_count >= args.max_new_items:
            break
        entry = state.get(file_id)
        if entry and entry.get("downloaded") == "failed":
            continue
        if entry and entry.get("total_pages") and entry["last_page"] >= entry["total_pages"]:
            continue
        process_book(title, file_id, state, tmp_dir)
        new_count += 1
        time.sleep(1)  # be polite to Drive between books

    log(f"jibonorshongram run complete ({new_count} new books this run).")


if __name__ == "__main__":
    main()
