"""Script-purity filter + Unicode normalization shared by hindi/data collection scripts.

Spec requires collected text be "in the target language's script and natural
phrasing." NFC normalization alone doesn't catch English bylines, ads, or
fully non-Devanagari lines that a scrape/OCR pass commonly picks up, so
whole lines that aren't predominantly Devanagari are dropped.

Word-level stripping of embedded Latin tokens was removed after a later
course Q&A clarification: an earlier answer ("digits are fine but do remove
non language words") was ambiguous and had been read as "strip embedded
English words/brand names like Tata, IRCTC." A follow-up question asked
directly whether that meant only gibberish (e.g. "asdfg") should be
removed while keeping meaningful English words and names like Tata, IRCTC,
and Google, and the answer was yes. Real proper nouns and loanwords
embedded in otherwise-Devanagari news prose are natural phrasing, not
noise, so they are kept; only whole lines that fail the majority-Devanagari
check are dropped.
"""

import unicodedata

DEVANAGARI_LO, DEVANAGARI_HI = 0x0900, 0x097F
MIN_DEVANAGARI_RATIO = 0.5


def devanagari_ratio(text):
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    deva = sum(1 for c in alpha if DEVANAGARI_LO <= ord(c) <= DEVANAGARI_HI)
    return deva / len(alpha)


def clean_text(text):
    """NFC-normalize, then drop any line that isn't predominantly Devanagari."""
    text = unicodedata.normalize("NFC", text)
    lines = text.split("\n")
    kept = [line for line in lines if devanagari_ratio(line) >= MIN_DEVANAGARI_RATIO]
    return "\n".join(kept)
