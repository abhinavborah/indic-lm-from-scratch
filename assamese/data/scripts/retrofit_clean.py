#!/usr/bin/env python3
"""One-off: re-clean already-written raw/ocr and raw/scrape .txt files with
the word-level purity filter added after they were first produced (course
clarification: strip individual Latin-script non-digit words, not just whole
lines). Rewrites each file in place; safe to rerun (idempotent -- re-cleaning
already-clean text is a no-op).
"""

from pathlib import Path

from text_clean import clean_text

DATA_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"


def log(msg):
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def retrofit_dir(dir_path):
    for path in sorted(dir_path.glob("*.txt")):
        before = path.read_text(encoding="utf-8")
        after, dropped_lines = clean_text(before)
        path.write_text(after + "\n", encoding="utf-8")
        before_words, after_words = len(before.split()), len(after.split())
        log(f"RETROFIT {path.relative_to(DATA_DIR)}: {before_words} -> {after_words} words "
            f"({dropped_lines} lines dropped this pass)")


if __name__ == "__main__":
    retrofit_dir(DATA_DIR / "raw" / "ocr")
    retrofit_dir(DATA_DIR / "raw" / "scrape")
    log("Retrofit word-level purity pass complete.")
