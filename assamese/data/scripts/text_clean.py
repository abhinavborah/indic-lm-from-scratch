"""Shared Assamese text-purity helpers, used by both ocr_pipeline.py and
scraper.py so cleaning behaves identically regardless of source.

Two passes, per course guidance ("digits are fine but do remove non
language words"):
  - line-level: drop a whole line if it isn't majority Bengali-Assamese-block
    script (catches boilerplate lines: bylines, dates, "Advertisment", etc).
  - word-level: within a kept line, drop individual whitespace-separated
    tokens that are pure Latin script with no Bengali characters (catches
    English proper nouns/brand names embedded mid-sentence). Digit-only and
    punctuation-only tokens are never dropped.
"""

import re

BENGALI_LO, BENGALI_HI = 0x0980, 0x09FF
MIN_LINE_BENGALI_RATIO = 0.5
_LATIN_RE = re.compile(r"[A-Za-z]")

# Assamese/Bengali share the same Unicode block, so script-range checks alone
# can't tell them apart. ৰ (ra) and ৱ (va/wa) are letters Assamese uses
# constantly (e.g. the -ৰ genitive case marker) and standard Bengali never
# uses at all; their near-total absence in an otherwise in-script segment
# is a signal the text is actually Bengali, not Assamese. This is a
# heuristic, not a language-ID classifier: short segments are exempted
# (too little signal), and it's only meaningful on text that hasn't gone
# through an Assamese-specific correction pass already (see ocr_pipeline.py's
# correct_ra, which force-converts all Bengali "র"->"ৰ" and would make this
# check trivially pass on anything, so it's not applied to OCR output for that
# reason, only to scraped text).
RA_VA_CHARS = {"ৰ", "ৱ"}
MIN_ASSAMESE_MARKER_RATIO = 0.005
MIN_CHARS_FOR_MARKER_CHECK = 30


def is_bengali_char(c):
    return BENGALI_LO <= ord(c) <= BENGALI_HI


def is_bengali_line(line):
    alpha = [c for c in line if c.isalpha()]
    if not alpha:
        return False
    beng = sum(1 for c in alpha if is_bengali_char(c))
    return (beng / len(alpha)) >= MIN_LINE_BENGALI_RATIO


def strip_latin_words(line):
    kept = []
    for word in line.split(" "):
        has_bengali = any(is_bengali_char(c) for c in word)
        has_latin = bool(_LATIN_RE.search(word))
        if has_latin and not has_bengali:
            continue  # pure-Latin token (English word/brand name), drop
        kept.append(word)
    return " ".join(kept)


def is_likely_assamese(text):
    """Whole-segment heuristic (not per-line: a single pure-Bengali-range
    line inside a genuinely Assamese article is normal). Returns True when
    there's too little Bengali-range text to judge, so callers should apply
    this only as a soft/logged signal, not a hard gate on short segments."""
    bengali_chars = [c for c in text if is_bengali_char(c)]
    if len(bengali_chars) < MIN_CHARS_FOR_MARKER_CHECK:
        return True
    markers = sum(1 for c in bengali_chars if c in RA_VA_CHARS)
    return (markers / len(bengali_chars)) >= MIN_ASSAMESE_MARKER_RATIO


def clean_text(raw_text):
    """Line-level filter, then word-level filter on survivors. Returns
    (cleaned_text, dropped_line_count)."""
    lines = [ln.strip() for ln in raw_text.splitlines()]
    kept_lines = []
    dropped = 0
    for ln in lines:
        if not ln:
            continue
        if not is_bengali_line(ln):
            dropped += 1
            continue
        kept_lines.append(strip_latin_words(ln))
    return "\n".join(kept_lines), dropped
