"""Script-purity filter + Unicode normalization shared by hindi/data collection scripts.

Spec requires collected text be "in the target language's script and natural
phrasing" -- NFC normalization alone doesn't catch English bylines, ads, or
code-switched Hinglish lines that a scrape/OCR pass commonly picks up. Course
Q&A clarification: digits are fine, but embedded English proper nouns/brand
names (e.g. "Tata", "IRCTC" inside an otherwise-Devanagari sentence) must be
stripped at the word level, not just by dropping the whole line. So cleaning
is two passes: (1) word-level -- drop any whitespace-separated token
containing a Latin letter, keeping digit/punctuation-only tokens and the
surrounding Devanagari words; (2) line-level -- on what's left, drop any line
that still isn't predominantly Devanagari (catches lines that were mostly/
entirely English before word-stripping, rather than leaving stray fragments).
"""

import re
import unicodedata

DEVANAGARI_LO, DEVANAGARI_HI = 0x0900, 0x097F
MIN_DEVANAGARI_RATIO = 0.5
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")


def devanagari_ratio(text):
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    deva = sum(1 for c in alpha if DEVANAGARI_LO <= ord(c) <= DEVANAGARI_HI)
    return deva / len(alpha)


def strip_latin_words(line):
    """Drop whitespace-separated tokens containing a Latin letter (English
    words, proper nouns, code-switched fragments); digit-only/punctuation
    tokens pass through untouched."""
    tokens = [t for t in line.split(" ") if not LATIN_LETTER_RE.search(t)]
    return " ".join(tokens)


def clean_text(text):
    """NFC-normalize, strip Latin-script words line by line, then drop any
    resulting line that isn't predominantly Devanagari."""
    text = unicodedata.normalize("NFC", text)
    lines = (strip_latin_words(line) for line in text.split("\n"))
    kept = [line for line in lines if devanagari_ratio(line) >= MIN_DEVANAGARI_RATIO]
    return "\n".join(kept)
