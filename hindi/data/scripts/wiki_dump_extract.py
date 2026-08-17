#!/usr/bin/env python3
"""Hindi Wikipedia dump extraction (manual-collection corpus).

Downloaded from dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-
articles.xml.bz2 (MediaWiki XML export, bz2-compressed, ~239MB). This script
is the "own preprocessing" that makes the dump count toward manual
collection rather than a ready-to-use downloaded corpus: it streams and
parses the raw MediaWiki XML/wikitext itself (hand-rolled, WikiExtractor-
style) and produces plain prose, applying the same purity-filter cleaning
(NFC normalize, word-level Latin-script strip, line-level Devanagari-density
filter) as every other source in this pipeline.

Streams the bz2 file page-by-page (never holds the whole decompressed dump
in memory) using stdlib xml.etree.ElementTree's iterparse. Skips redirects,
disambiguation stubs, and non-article namespaces. Wikitext markup (templates,
refs, tables, links, headings) is stripped with regex-based rules -- not a
full MediaWiki parser, but clean_text.py's Devanagari/Latin filters catch
most of what leaks through as non-prose.

Single-process/CPU-bound: a multiprocessing.Pool was tried for the
strip_wikitext()+clean_text() step, but the actual bottleneck turned out to
be the bz2 decompress + XML iterparse itself (both inherently sequential,
single stream, stay on the main process no matter how many downstream
workers exist) -- the pool sat near-idle while the main process stayed
pegged. A real fix means splitting the *decompressed* XML by page boundaries
into byte-range chunks and running independent decompress+parse+strip+clean
per chunk (with per-chunk state/output that gets merged) -- meaningfully
more complex, left as a follow-up rather than rushed in.

Resumable: hindi/data/.state.json under the "wiki_dump" namespace tracks the
page title of the last fully-written article, so a rerun skips already-
processed titles rather than re-parsing/re-writing them. Since iterparse must
walk the file from the start every time (bz2 streams aren't seekable to an
arbitrary page), resume still costs re-scanning time but not re-writing
output or re-cleaning text for already-done pages.
"""

import argparse
import bz2
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import state_io
from clean_text import clean_text

DATA_DIR = Path(__file__).resolve().parents[1]  # hindi/data
DUMP_PATH = DATA_DIR / "raw" / "scrape" / ".wikidump" / "hiwiki-latest-pages-articles.xml.bz2"
OUT_PATH = DATA_DIR / "raw" / "scrape" / "hi_wikipedia_dump.txt"        # stripped-wikitext prose, unfiltered
CLEAN_PATH = DATA_DIR / "clean" / "scrape" / "hi_wikipedia_dump.txt"    # clean_text() output
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "wiki_dump"

LOG_EVERY_N_PAGES = 200
DEFAULT_MAX_PAGES = 20000
DEFAULT_MAX_SECONDS = 3600

REDIRECT_RE = re.compile(r"^#REDIRECT", re.IGNORECASE)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def strip_wikitext(wikitext):
    """Minimal wikitext -> plain text: drop templates/refs/tables/markup
    noise so what's left is prose. Not a full MediaWiki parser -- good
    enough given clean_text.py's Devanagari-density filter mops up anything
    that slips through as a non-Hindi line/word anyway."""
    text = wikitext
    text = re.sub(r"<ref[^>]*/?>.*?</ref>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", " ", text)
    # Nested {{templates}} -- strip innermost-first, a few passes handles
    # realistic nesting depth without a real balanced-brace parser.
    for _ in range(5):
        new_text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
        if new_text == text:
            break
        text = new_text
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.DOTALL)  # tables
    text = re.sub(r"\[\[(?:[^|\]]*:)?(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)  # [[ns:link|text]] -> text
    text = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", text)  # external link w/ label
    text = re.sub(r"\[https?://\S+\]", " ", text)
    text = re.sub(r"'''?", "", text)  # bold/italic markers
    text = re.sub(r"==+\s*([^=]+?)\s*==+", r"\1", text)  # headings -> plain line
    text = re.sub(r"<[^>]+>", " ", text)  # stray HTML tags
    text = re.sub(r"^\s*[*#:;]+\s*", "", text, flags=re.MULTILINE)  # list/indent markers
    return text


def iter_pages(dump_path):
    """Yields (title, wikitext) for each main-namespace, non-redirect page in
    the dump. Streams via bz2 + iterparse -- never loads the full XML tree."""
    with bz2.open(dump_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        ns_prefix = None
        for _, elem in context:
            tag = elem.tag
            if ns_prefix is None and "}" in tag:
                ns_prefix = tag.split("}")[0] + "}"
            local = tag.split("}")[-1]
            if local != "page":
                continue

            def find(name):
                return elem.find(f"{ns_prefix}{name}" if ns_prefix else name)

            ns_el = find("ns")
            if ns_el is None or ns_el.text != "0":  # main article namespace only
                elem.clear()
                continue
            title_el = find("title")
            revision_el = find("revision")
            text_el = revision_el.find(f"{ns_prefix}text" if ns_prefix else "text") if revision_el is not None else None
            title = title_el.text if title_el is not None else None
            wikitext = text_el.text if text_el is not None else None
            elem.clear()

            if not title or not wikitext or REDIRECT_RE.match(wikitext.strip()):
                continue
            yield title, wikitext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                         help="stop after extracting this many new pages")
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS,
                         help="stop after this many wall-clock seconds")
    args = parser.parse_args()

    if not DUMP_PATH.exists():
        log(f"ERROR: dump not found at {DUMP_PATH}")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    state = state_io.load_namespace(STATE_FILE, NAMESPACE, {"done_titles": [], "total_words": 0})
    done = set(state["done_titles"])
    log(f"wiki_dump: {len(done)} pages already extracted from prior runs")

    batch_start = time.monotonic()
    batch_new = 0
    batch_words = 0
    scanned = 0
    with OUT_PATH.open("a", encoding="utf-8") as out_f, CLEAN_PATH.open("a", encoding="utf-8") as clean_f:
        for title, wikitext in iter_pages(DUMP_PATH):
            scanned += 1
            if title in done:
                continue

            stripped = strip_wikitext(wikitext)
            cleaned = clean_text(stripped)
            words = len(cleaned.split())
            out_f.write(stripped + "\n\n")
            out_f.flush()
            if cleaned:
                clean_f.write(cleaned + "\n\n")
                clean_f.flush()

            done.add(title)
            state["done_titles"] = list(done)
            state["total_words"] = state.get("total_words", 0) + words
            state_io.save_namespace(STATE_FILE, NAMESPACE, state)

            batch_new += 1
            batch_words += words
            if batch_new % LOG_EVERY_N_PAGES == 0:
                elapsed = time.monotonic() - batch_start
                log(f"wiki_dump: {batch_new} new pages this batch ({scanned} scanned), "
                    f"~{batch_words} words this batch, {elapsed:.0f}s elapsed")

            elapsed = time.monotonic() - batch_start
            if batch_new >= args.max_pages or elapsed >= args.max_seconds:
                break

    elapsed = time.monotonic() - batch_start
    log(f"BATCH COMPLETE: {batch_new} new pages extracted ({scanned} scanned this run), "
        f"~{batch_words} words this batch, {elapsed:.0f}s elapsed. "
        f"Overall: {len(done)} pages, ~{state['total_words']} words total ever.")


if __name__ == "__main__":
    main()
