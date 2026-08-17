#!/usr/bin/env python3
"""Assamese Wikipedia manual-collection source: dump-based, not live scraping.

Course clarification: a Wikipedia XML dump counts as manual collection
*if you preprocess it yourself* (unlike a ready-made corpus like Sangraha,
which is already tokenized prose). This script streams
aswiki-latest-pages-articles.xml.bz2, strips wikitext markup down to prose
by hand (no mwparserfromhell/wikiextractor dependency -- regex-based, same
spirit as the hand-rolled tokenizer elsewhere in this project), and runs the
result through the same text_clean purity filters used for OCR/scrape output.

Dump source: https://dumps.wikimedia.org/aswiki/latest/
             aswiki-latest-pages-articles.xml.bz2 (~67.5MiB compressed)

Resumable: assamese/data/.state.json under the "wikidump" namespace tracks
the last fully-processed page index (dump order is stable across identical
files), so a rerun skips already-extracted pages. Output: unfiltered prose
to assamese/data/raw/scrape/as_wikipedia.txt, purity-filtered text to
assamese/data/clean/scrape/as_wikipedia.txt (same shape as the news-scrape
output files, one article's text per block, blank-line separated).
"""

import bz2
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import state_io
from text_clean import clean_text, is_likely_assamese

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
DUMP_PATH = DATA_DIR / "raw" / "wikipedia_dump" / "aswiki-latest-pages-articles.xml.bz2"
RAW_OUT_PATH = DATA_DIR / "raw" / "scrape" / "as_wikipedia.txt"
CLEAN_OUT_PATH = DATA_DIR / "clean" / "scrape" / "as_wikipedia.txt"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "wikidump"
LOG_EVERY_N_PAGES = 200


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


# --- wikitext -> prose -------------------------------------------------

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_RE = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.DOTALL | re.IGNORECASE)
_TABLE_INNERMOST_RE = re.compile(r"\{\|(?:(?!\{\||\|\}).)*?\|\}", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TEMPLATE_INNERMOST_RE = re.compile(r"\{\{[^{}]*\}\}")
_EXTERNAL_LINK_WITH_TEXT_RE = re.compile(r"\[https?://\S+\s+([^\]]+)\]")
_EXTERNAL_LINK_BARE_RE = re.compile(r"\[https?://\S+\]")
_WIKILINK_PIPED_RE = re.compile(r"\[\[[^\]|]*\|([^\]]+)\]\]")
_WIKILINK_PLAIN_RE = re.compile(r"\[\[([^\]|]+)\]\]")
_BOLD_ITALIC_RE = re.compile(r"'{2,5}")
_HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+$", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^[*#:;]+\s*", re.MULTILINE)
# Namespaces that are never prose (files/images/categories: in whichever
# language the wiki localizes them, plus the English fallback names).
_NON_PROSE_NS_RE = re.compile(
    r"\[\[\s*(File|Image|Category|চিত্ৰ|শ্ৰেণী):"
    r"(?:[^\[\]]|\[\[[^\[\]]*\]\])*\]\]",  # tolerates one nested [[link]] in the caption
    re.IGNORECASE,
)


def wikitext_to_prose(wikitext):
    text = _COMMENT_RE.sub("", wikitext)
    text = _REF_RE.sub("", text)
    text = _NON_PROSE_NS_RE.sub("", text)

    # {| tables |} and {{templates}} can both nest; repeatedly collapse the
    # innermost match (one with no further nesting inside) until none remain.
    for _ in range(10):
        new_text = _TABLE_INNERMOST_RE.sub("", text)
        new_text = _TEMPLATE_INNERMOST_RE.sub("", new_text)
        if new_text == text:
            break
        text = new_text

    text = _EXTERNAL_LINK_WITH_TEXT_RE.sub(r"\1", text)
    text = _EXTERNAL_LINK_BARE_RE.sub("", text)
    text = _WIKILINK_PIPED_RE.sub(r"\1", text)
    text = _WIKILINK_PLAIN_RE.sub(r"\1", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = _HEADING_RE.sub(r"\1", text)
    text = _LIST_MARKER_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = html.unescape(text)
    # Safety net: malformed source wikitext (mismatched single/double
    # brackets, truncated templates) occasionally survives the structured
    # passes above as a stray bracket -- real Assamese prose never uses
    # square/curly brackets, so any left at this point are markup residue.
    text = re.sub(r"[\[\]{}]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- dump streaming -----------------------------------------------------

def iter_dump_pages(dump_path):
    """Yields (title, wikitext) for namespace-0, non-redirect pages, in
    stable dump order. Streams the bz2 file directly -- never fully
    decompresses to disk (the decompressed dump is several hundred MB)."""
    with bz2.open(dump_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag != "page":
                continue
            ns_text = elem.findtext("./{*}ns")
            redirect = elem.find("./{*}redirect")
            title = elem.findtext("./{*}title")
            text = elem.findtext("./{*}revision/{*}text")
            if ns_text == "0" and redirect is None and title and text:
                yield title, text
            elem.clear()


def main():
    if not DUMP_PATH.exists():
        log(f"MISSING dump file: {DUMP_PATH} -- download it first from "
            f"https://dumps.wikimedia.org/aswiki/latest/aswiki-latest-pages-articles.xml.bz2")
        return

    state = state_io.load_namespace(
        STATE_FILE, NAMESPACE,
        {"last_page_index": 0, "pages_written": 0, "total_words": 0,
         "dropped_lines": 0, "suspect_language": 0},
    )
    start_index = state["last_page_index"]
    log(f"Resuming wikidump extraction from page index {start_index}")

    mode = "a" if start_index > 0 else "w"
    RAW_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_OUT_PATH.open(mode, encoding="utf-8") as raw_f, \
         CLEAN_OUT_PATH.open(mode, encoding="utf-8") as clean_f:
        for i, (title, wikitext) in enumerate(iter_dump_pages(DUMP_PATH)):
            if i < start_index:
                continue

            prose = wikitext_to_prose(wikitext)
            if prose:
                raw_f.write(prose + "\n\n")
                raw_f.flush()

            cleaned, dropped = clean_text(prose)
            if cleaned and not is_likely_assamese(cleaned):
                log(f"SUSPECT-LANGUAGE (low ৰ/ৱ frequency): {title}")
                state["suspect_language"] += 1
                cleaned = ""

            if cleaned:
                clean_f.write(cleaned + "\n\n")
                clean_f.flush()
                state["pages_written"] += 1
                state["total_words"] += len(cleaned.split())
            state["dropped_lines"] += dropped
            state["last_page_index"] = i + 1
            state_io.save_namespace(STATE_FILE, NAMESPACE, state)

            if (i + 1) % LOG_EVERY_N_PAGES == 0:
                log(f"as_wikipedia (dump): {i + 1} pages scanned, "
                    f"{state['pages_written']} written, ~{state['total_words']} words so far")

    log(f"DONE as_wikipedia (dump): {state['pages_written']} pages written, "
        f"~{state['total_words']} words total, {state['suspect_language']} suspect-language pages excluded")


if __name__ == "__main__":
    main()
