#!/usr/bin/env python3
"""Cleaning pass applied before feeding text to the sentencepiece trainer.

Reuses hindi/data/scripts/clean_text.py (NFC-normalize, strip embedded
Latin-script words, drop non-Devanagari-majority lines) so the tokenizer is
trained on the same notion of "clean Hindi" as the corpus-progress reports
use -- one purity definition per language, not a second copy that could
drift from it.
"""

import sys
from pathlib import Path

DATA_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "data" / "scripts"
sys.path.insert(0, str(DATA_SCRIPTS_DIR))
from clean_text import clean_text  # noqa: E402


def preprocess_lines(lines):
    """Clean an iterable of raw lines/segments, dropping ones that clean to nothing."""
    for line in lines:
        cleaned = clean_text(line.strip())
        if cleaned.strip():
            yield cleaned


def preprocess_file(raw_path, out_path):
    """Clean a raw text file into a plain-text file, one cleaned line per line
    -- the format sentencepiece's trainer expects as `input=`."""
    raw_path, out_path = Path(raw_path), Path(out_path)
    with raw_path.open("r", encoding="utf-8", errors="replace") as src, \
         out_path.open("w", encoding="utf-8") as dst:
        for cleaned in preprocess_lines(src):
            dst.write(cleaned + "\n")
    return out_path
