#!/usr/bin/env python3
"""OCR pipeline for Hindi manual-collection corpus (NCERT Hindi-medium textbooks).

Per chapter code (see ncert_catalog.py): download the PDF from
ncert.nic.in/textbook/pdf/, then try pdftotext first (many NCERT PDFs are
born-digital). Some NCERT PDFs embed a non-Unicode glyph font -- pdftotext
"succeeds" but returns garbage/mojibake with near-zero real Devanagari
codepoints -- so the extracted text is validated by Devanagari-block density
before being trusted; below threshold, falls back to rendering the page via
pdftoppm and running Tesseract with the `hin` model. Output is passed through
clean_text.py (NFC normalize + drop non-Devanagari lines) before being
appended to the per-book output file, so raw/ocr/*.txt is already
script-filtered.

Resumable: hindi/data/.state.json tracks per-chapter-code status ("done" /
"no_pdf"). Safe to interrupt (Ctrl-C) and rerun; already-finished codes are
skipped and never re-downloaded.
"""

import argparse
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import state_io
from clean_text import clean_text, devanagari_ratio
from ncert_catalog import chapter_codes

DATA_DIR = Path(__file__).resolve().parents[1]  # hindi/data
OUT_DIR = DATA_DIR / "raw" / "ocr"      # original, unfiltered extracted text
CLEAN_DIR = DATA_DIR / "clean" / "ocr"  # clean_text() output, same filenames
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "ocr"

PDF_URL = "https://ncert.nic.in/textbook/pdf/{code}.pdf"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
MIN_DEVANAGARI_RATIO = 0.3  # below this, pdftotext output is treated as garbage -> OCR fallback
OCR_DPI = 200
CURL_ATTEMPT_TIMEOUT_S = 30  # single curl invocation's own -m timeout
DOWNLOAD_DEADLINE_S = 45 * 60  # retry a single file's download for up to this long before giving up
DOWNLOAD_RETRY_BACKOFF_S = 5   # pause between retry attempts on the same file
DOWNLOAD_WORKERS = 6  # concurrent PDF downloads -- a slow/flaky file must not block the others
SUBPROCESS_TIMEOUT_S = 60  # pdftoppm/tesseract can hang on a malformed page; never block forever
LOG_EVERY_N_ITEMS = 10
DEFAULT_MAX_NEW_ITEMS = 200  # batch pacing: stop after N *newly* attempted items...
DEFAULT_MAX_SECONDS = 3600   # ...or M wall-clock seconds, whichever comes first


def load_state():
    return state_io.load_namespace(STATE_FILE, NAMESPACE, {})


def save_state(state):
    state_io.save_namespace(STATE_FILE, NAMESPACE, state)


def log(msg, log_file=LOG_FILE):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def _looks_complete(pdf_path):
    with pdf_path.open("rb") as f:
        if f.read(4) != b"%PDF":
            return False
        f.seek(-1024, 2)
        return b"%%EOF" in f.read()


def download_pdf(code, tmp_dir, stop_event=None):
    """ncert.nic.in resets the connection mid-transfer on larger PDFs fairly
    often; retry with -C - (range resume) rather than restarting from zero.
    Retries against a wall-clock deadline (not a fixed attempt count) so a
    file that's merely slow -- rather than genuinely dead -- gets a generous
    chance before being given up on as a real 404/failure. `stop_event`, when
    given, lets an in-progress retry loop abandon early once the batch has
    already decided to stop -- otherwise a worker thread's up-to-45min retry
    budget would keep the whole process alive well past the batch boundary
    (ThreadPoolExecutor workers are not daemon threads).
    """
    out_path = tmp_dir / f"{code}.pdf"
    out_path.unlink(missing_ok=True)
    url = PDF_URL.format(code=code)

    deadline = time.monotonic() + DOWNLOAD_DEADLINE_S
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            out_path.unlink(missing_ok=True)
            return None
        try:
            r = subprocess.run(
                ["curl", "-s", "-m", str(CURL_ATTEMPT_TIMEOUT_S), "-A", USER_AGENT, "-C", "-",
                 "-o", str(out_path), "-w", "%{http_code}", url],
                capture_output=True, text=True, timeout=CURL_ATTEMPT_TIMEOUT_S + 10,
            )
        except subprocess.TimeoutExpired:
            continue
        http_code = r.stdout.strip()
        if http_code not in ("200", "206"):
            out_path.unlink(missing_ok=True)
            return None
        if out_path.exists() and _looks_complete(out_path):
            return out_path
        time.sleep(DOWNLOAD_RETRY_BACKOFF_S)
    out_path.unlink(missing_ok=True)
    return None


def _run(cmd, **kw):
    """subprocess.run with a hard timeout -- pdftoppm/tesseract can hang on a
    malformed page, and an unbounded call would stall the whole pipeline."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S, **kw)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="timeout")


def page_count(pdf_path):
    out = _run(["pdfinfo", str(pdf_path)]).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":")[1].strip())
    return 0


def extract_pdftotext(pdf_path):
    r = _run(["pdftotext", "-layout", str(pdf_path), "-"])
    return r.stdout if r.returncode == 0 else ""


def extract_ocr(pdf_path, tmp_dir):
    total = page_count(pdf_path)
    pages = []
    for page in range(1, total + 1):
        prefix = tmp_dir / f"pg{page}"
        _run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(OCR_DPI),
              "-png", str(pdf_path), str(prefix)])
        pngs = sorted(tmp_dir.glob(f"pg{page}-*.png")) or sorted(tmp_dir.glob(f"pg{page}.png"))
        if not pngs:
            continue
        png = pngs[0]
        r = _run(["tesseract", str(png), "-", "-l", "hin"])
        pages.append(r.stdout if r.returncode == 0 else "")
        for p in pngs:
            p.unlink(missing_ok=True)
    return "\n\n".join(pages)


def process_download_result(prefix, code, pdf_path, state, tmp_dir, log_file=LOG_FILE):
    """Extract + clean + checkpoint a downloaded PDF (or record no_pdf if the
    download failed). Runs in the main thread even when downloads themselves
    are parallelized, since it also does the CPU-bound OCR fallback and
    writes to the shared state file. Writes BOTH the original unfiltered
    extracted text (raw/ocr/) and clean_text()'s output (clean/ocr/) -- the
    spec requires retaining both, not just the cleaned version."""
    if pdf_path is None:
        state[code] = {"status": "no_pdf"}
        save_state(state)
        return False, 0

    text = extract_pdftotext(pdf_path)
    used_ocr = devanagari_ratio(text) < MIN_DEVANAGARI_RATIO
    if used_ocr:
        text = extract_ocr(pdf_path, tmp_dir)
    cleaned = clean_text(text)
    pdf_path.unlink(missing_ok=True)

    with (OUT_DIR / f"{prefix}.txt").open("a", encoding="utf-8") as f:
        f.write(text + "\n\n")
    with (CLEAN_DIR / f"{prefix}.txt").open("a", encoding="utf-8") as f:
        f.write(cleaned + "\n\n")

    words = len(cleaned.split())
    state[code] = {"status": "done", "used_ocr": used_ocr, "words": words}
    save_state(state)
    log(f"{code}: {'OCR' if used_ocr else 'pdftotext'}, ~{words} words", log_file)
    return True, words


def process_item(prefix, code, state, tmp_dir, log_file=LOG_FILE):
    """Serial download+extract for a single item (used by the resume self-test
    and any one-off/ad hoc call). The batch runner in main() parallelizes the
    download step across items instead of calling this directly."""
    if code in state:  # "done" or "no_pdf" -- both are resolved, skip on resume/re-batch
        return
    pdf_path = download_pdf(code, tmp_dir)
    process_download_result(prefix, code, pdf_path, state, tmp_dir, log_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-items", type=int, default=DEFAULT_MAX_NEW_ITEMS,
                         help="stop after attempting this many previously-unresolved codes")
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS,
                         help="stop after this many wall-clock seconds, whichever comes first")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    state = load_state()
    items = list(chapter_codes())
    already_resolved = sum(1 for _, code in items if code in state)
    log(f"NCERT Hindi catalog: {len(items)} chapter/prelim codes across "
        f"{len({p for p, _ in items})} books ({already_resolved} already resolved from prior batches)")

    pending = iter((prefix, code) for prefix, code in items if code not in state)
    batch_start = time.monotonic()
    batch_new = 0     # codes newly attempted this batch (skips don't count)
    batch_done = 0    # of those, how many extracted successfully
    batch_words = 0

    # Downloads run in a thread pool (I/O-bound, so a slow/flaky file doesn't
    # block the others behind it); extraction (CPU-bound OCR + shared-state
    # writes) stays serial in the main thread as each download completes.
    # cancel_futures=True on shutdown only drops *unstarted* work -- a
    # download already mid-retry (up to DOWNLOAD_DEADLINE_S) keeps running in
    # its thread even after we stop consuming results, so at-most-N-workers
    # of overrun past a batch's stop point is expected, not a bug.
    stop_event = threading.Event()
    pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
    try:
        in_flight = {}  # future -> (prefix, code)

        def _top_up():
            while len(in_flight) < DOWNLOAD_WORKERS:
                item = next(pending, None)
                if item is None:
                    break
                prefix, code = item
                fut = pool.submit(download_pdf, code, tmp_dir, stop_event)
                in_flight[fut] = (prefix, code)

        _top_up()
        stop = False
        while in_flight and not stop:
            fut = next(as_completed(in_flight))
            prefix, code = in_flight.pop(fut)
            pdf_path = fut.result()
            extracted, words = process_download_result(prefix, code, pdf_path, state, tmp_dir)
            batch_new += 1
            if extracted:
                batch_done += 1
                batch_words += words

            if batch_new % LOG_EVERY_N_ITEMS == 0:
                log(f"batch progress: {batch_new} new codes attempted "
                    f"({batch_done} extracted, {batch_words} words) this batch")

            elapsed = time.monotonic() - batch_start
            stop = batch_new >= args.max_new_items or elapsed >= args.max_seconds
            if not stop:
                _top_up()
    finally:
        stop_event.set()  # tell any in-flight retry loops to abandon now, not in up to 45min
        pool.shutdown(wait=True, cancel_futures=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    elapsed = time.monotonic() - batch_start
    total_attempted = sum(1 for _, code in items if code in state)
    log(f"BATCH COMPLETE: {batch_new} new codes attempted this batch "
        f"({batch_done} extracted, {batch_words} words), {elapsed:.0f}s elapsed. "
        f"Overall: {total_attempted}/{len(items)} codes resolved so far.")


if __name__ == "__main__":
    main()
