#!/usr/bin/env python3
"""OCR pipeline for Assamese manual-collection corpus (SCERT/NCERT textbook PDFs).

Per page: try pdftotext first (fast path for born-digital PDFs). Many of
these government PDFs embed a non-Unicode glyph font, so pdftotext succeeds
but returns garbage (zero real Bengali/Assamese codepoints) -- validated by
checking the Bengali Unicode block density of the extracted text. Below
threshold, falls back to rendering the page via pdftoppm and running
Tesseract with the `ben` (Bengali) model (no usable `asm` pack exists;
Assamese and Bengali share nearly the whole glyph inventory).

Known OCR limitation: Assamese has no letter corresponding to Bengali "র" --
every "র" the ben model outputs is actually misrecognized Assamese "ৰ", so
that substitution is applied unconditionally (100% safe). Bengali "ব" is
NOT touched: Assamese uses both "ব" and "ৱ" natively, so ben-model output of
"ব" is genuinely ambiguous and is left uncorrected -- spot-check if precision
here matters.

Resumable: assamese/data/.state.json tracks per-PDF last completed page.
Safe to interrupt (Ctrl-C) and rerun; already-finished pages are skipped.
"""

import re
import subprocess
from datetime import datetime
from pathlib import Path

import state_io
from text_clean import clean_text

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
BOOKS_DIR = DATA_DIR.parents[2] / "docs" / "assamese" / "books"  # .../ind_proj/docs/...
RAW_OUT_DIR = DATA_DIR / "raw" / "ocr"
CLEAN_OUT_DIR = DATA_DIR / "clean" / "ocr"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "ocr"

BENGALI_LO, BENGALI_HI = 0x0980, 0x09FF
MIN_BENGALI_RATIO = 0.3
OCR_DPI = 200
LOG_EVERY_N_PAGES = 10


def load_state():
    return state_io.load_namespace(STATE_FILE, NAMESPACE, {})


def save_state(ocr_state):
    state_io.save_namespace(STATE_FILE, NAMESPACE, ocr_state)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def bengali_ratio(text):
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    beng = sum(1 for c in alpha if BENGALI_LO <= ord(c) <= BENGALI_HI)
    return beng / len(alpha)


def correct_ra(text):
    return text.replace("র", "ৰ")


def slugify(pdf_path):
    return re.sub(r"[^a-zA-Z0-9]+", "_", pdf_path.stem).strip("_").lower()


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


def process_pdf(pdf_path, ocr_state, tmp_dir):
    slug = slugify(pdf_path)
    raw_path = RAW_OUT_DIR / f"{slug}.txt"
    clean_path = CLEAN_OUT_DIR / f"{slug}.txt"
    entry = ocr_state.setdefault(
        slug, {"last_page": 0, "total_pages": None, "ocr_pages": 0, "text_pages": 0}
    )
    total = entry["total_pages"] or page_count(pdf_path)
    entry["total_pages"] = total
    if total == 0:
        log(f"SKIP {pdf_path.name}: pdfinfo reported 0 pages")
        return
    if entry["last_page"] >= total:
        return

    mode = "a" if entry["last_page"] > 0 else "w"
    with raw_path.open(mode, encoding="utf-8") as raw_f, \
         clean_path.open(mode, encoding="utf-8") as clean_f:
        for page in range(entry["last_page"] + 1, total + 1):
            text = extract_page_pdftotext(pdf_path, page)
            used_ocr = bengali_ratio(text) < MIN_BENGALI_RATIO
            if used_ocr:
                text = extract_page_ocr(pdf_path, page, tmp_dir)
            # correct_ra is an OCR-fidelity fix (Assamese "ৰ" is always
            # misrecognized as Bengali "র" by the ben model), not corpus
            # cleaning -- applied to both raw and clean output.
            raw_text = correct_ra(text)
            clean_text_out, _dropped_lines = clean_text(raw_text)

            raw_f.write(raw_text + "\n\n")
            raw_f.flush()
            clean_f.write(clean_text_out + "\n\n")
            clean_f.flush()

            entry["last_page"] = page
            entry["ocr_pages" if used_ocr else "text_pages"] += 1
            save_state(ocr_state)

            if page % LOG_EVERY_N_PAGES == 0 or page == total:
                words = len(clean_text_out.split())
                log(f"{pdf_path.name}: page {page}/{total} "
                    f"({'OCR' if used_ocr else 'pdftotext'}), ~{words} clean words this page")

    log(f"DONE {pdf_path.name}: {total} pages total "
        f"({entry['ocr_pages']} OCR, {entry['text_pages']} text-layer)")


def main():
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = RAW_OUT_DIR / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ocr_state = load_state()
    pdfs = sorted(BOOKS_DIR.glob("*.pdf"))
    log(f"Found {len(pdfs)} PDFs in {BOOKS_DIR}")

    for pdf_path in pdfs:
        entry = ocr_state.get(slugify(pdf_path))
        if entry and entry.get("total_pages") and entry["last_page"] >= entry["total_pages"]:
            continue
        process_pdf(pdf_path, ocr_state, tmp_dir)

    log("OCR pipeline run complete (all PDFs at or past their last checkpoint).")


if __name__ == "__main__":
    main()
