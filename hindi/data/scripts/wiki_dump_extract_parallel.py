#!/usr/bin/env python3
"""Hindi Wikipedia dump extraction, parallel (multistream dump).

Replaces wiki_dump_extract.py's single-stream approach. The plain
hiwiki-latest-pages-articles.xml.bz2 is one continuous bz2 stream that can
only be decompressed sequentially from byte zero -- exactly why an earlier
attempt at multiprocessing.Pool over strip_wikitext()/clean_text() did
nothing (the real bottleneck, bz2 decompress + XML parse, stayed on the main
process no matter how many downstream workers existed).

hiwiki-latest-pages-articles-multistream.xml.bz2 is Wikimedia's dump built
for exactly this: it's a sequence of *independent* bz2 streams, each holding
~100 <page> elements, with no cross-stream dependency. Its companion index
(hiwiki-latest-pages-articles-multistream-index.txt.bz2, format
"byte_offset:page_id:title", grouped by the offset where each stream
starts) lets any process seek directly to a stream's start and decompress
just that slice -- independent of every other stream. That's what makes
real parallelism possible: N worker processes each own a disjoint set of
byte ranges and decompress/parse/strip/clean them concurrently, same
approach WikiExtractor and other dump-parallelization tools use.

Checkpointing is per-chunk (one multistream block, ~100 pages) rather than
per-page: chunks are the natural unit of independent work here, there are
thousands of them, and per-chunk granularity keeps the state file from
needing per-page locking across worker processes. A rerun skips whole chunks
already marked done in hindi/data/.state.json under the "wiki_dump_parallel"
namespace (separate from the older single-stream script's "wiki_dump"
namespace, since they track different units).
"""

import argparse
import bz2
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import state_io
from clean_text import clean_text

DATA_DIR = Path(__file__).resolve().parents[1]  # hindi/data
DUMP_DIR = DATA_DIR / "raw" / "scrape" / ".wikidump"
DUMP_PATH = DUMP_DIR / "hiwiki-latest-pages-articles-multistream.xml.bz2"
INDEX_PATH = DUMP_DIR / "hiwiki-latest-pages-articles-multistream-index.txt.bz2"
OUT_PATH = DATA_DIR / "raw" / "scrape" / "hi_wikipedia_dump.txt"
CLEAN_PATH = DATA_DIR / "clean" / "scrape" / "hi_wikipedia_dump.txt"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "wiki_dump_parallel"

DEFAULT_WORKERS = None  # None -> ProcessPoolExecutor default (os.cpu_count())
DEFAULT_MAX_CHUNKS = 100000
DEFAULT_MAX_SECONDS = 3600
LOG_EVERY_N_CHUNKS = 20

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
    for _ in range(5):
        new_text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
        if new_text == text:
            break
        text = new_text
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[\[(?:[^|\]]*:)?(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", " ", text)
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"==+\s*([^=]+?)\s*==+", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s*[*#:;]+\s*", "", text, flags=re.MULTILINE)
    return text


def build_chunk_offsets(index_path):
    """Parses the multistream index (bz2-compressed, "offset:page_id:title"
    per line, many consecutive lines sharing one offset -- one per stream)
    into a sorted list of unique byte offsets. Each offset is the start of
    one independent bz2 stream in the dump; paired with the next offset (or
    EOF for the last one) it defines that stream's exact byte range."""
    offsets = set()
    with bz2.open(index_path, "rt", encoding="utf-8") as f:
        for line in f:
            offset_str = line.split(":", 1)[0]
            offsets.add(int(offset_str))
    return sorted(offsets)


def _process_chunk(args_tuple):
    """Runs in a worker process: seek to this chunk's byte range in the
    multistream dump, decompress just that slice (an independent bz2 stream
    -- no dependency on any other chunk), parse the <page> elements inside,
    strip+clean each one. Returns a list of (title, stripped, cleaned).

    A malformed chunk (bad XML) is logged and skipped rather than raised --
    one bad ~100-page stream out of thousands shouldn't kill the whole pool
    run and lose every already-completed chunk's in-flight work."""
    dump_path, start, end = args_tuple
    with open(dump_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start) if end is not None else f.read()
    xml_fragment = bz2.decompress(raw)
    # Each stream is a bare sequence of <page>...</page> elements with no
    # single root -- wrap in one so ElementTree can parse it as a document.
    try:
        root = ET.fromstring(b"<root>" + xml_fragment + b"</root>")
    except ET.ParseError as e:
        print(f"wiki_dump_parallel: SKIP malformed chunk at offset {start}: {e}", flush=True)
        return []

    results = []
    for page in root.findall("page"):
        ns_el = page.find("ns")
        if ns_el is None or ns_el.text != "0":
            continue
        title_el = page.find("title")
        revision_el = page.find("revision")
        text_el = revision_el.find("text") if revision_el is not None else None
        title = title_el.text if title_el is not None else None
        wikitext = text_el.text if text_el is not None else None
        if not title or not wikitext or REDIRECT_RE.match(wikitext.strip()):
            continue
        stripped = strip_wikitext(wikitext)
        cleaned = clean_text(stripped)
        results.append((title, stripped, cleaned))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS,
                         help="stop after processing this many new chunks")
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS,
                         help="stop after this many wall-clock seconds")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help="parallel worker processes (default: os.cpu_count())")
    args = parser.parse_args()

    if not DUMP_PATH.exists() or not INDEX_PATH.exists():
        log(f"ERROR: multistream dump/index not found at {DUMP_PATH} / {INDEX_PATH}")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    log("wiki_dump_parallel: building chunk offset table from index...")
    offsets = build_chunk_offsets(INDEX_PATH)
    dump_size = DUMP_PATH.stat().st_size
    chunks = [(offsets[i], offsets[i + 1] if i + 1 < len(offsets) else dump_size)
              for i in range(len(offsets))]
    log(f"wiki_dump_parallel: {len(chunks)} independent chunks in multistream dump")

    state = state_io.load_namespace(STATE_FILE, NAMESPACE, {"done_offsets": [], "total_words": 0})
    done_offsets = set(state["done_offsets"])
    pending = [(start, end) for start, end in chunks if start not in done_offsets]
    log(f"wiki_dump_parallel: {len(done_offsets)} chunks already done, {len(pending)} pending")

    workers = args.workers or (os.cpu_count() or 4)
    in_flight_limit = workers * 2  # bounded window -- NOT "submit everything upfront": that would make
                                    # an early time-cap stop meaningless, since shutdown(wait=True) blocks
                                    # until every already-dispatched chunk finishes regardless
    todo = iter(pending[:args.max_chunks])

    batch_start = time.monotonic()
    batch_chunks = 0
    batch_pages = 0
    batch_words = 0
    with OUT_PATH.open("a", encoding="utf-8") as out_f, CLEAN_PATH.open("a", encoding="utf-8") as clean_f, \
         ProcessPoolExecutor(max_workers=workers) as pool:
        in_flight = {}

        def _top_up():
            while len(in_flight) < in_flight_limit:
                item = next(todo, None)
                if item is None:
                    break
                start, end = item
                fut = pool.submit(_process_chunk, (str(DUMP_PATH), start, end))
                in_flight[fut] = start

        _top_up()
        stop = False
        while in_flight and not stop:
            future = next(as_completed(in_flight))
            start = in_flight.pop(future)
            pages = future.result()
            for title, stripped, cleaned in pages:
                words = len(cleaned.split())
                out_f.write(stripped + "\n\n")
                if cleaned:
                    clean_f.write(cleaned + "\n\n")
                batch_pages += 1
                batch_words += words
            out_f.flush()
            clean_f.flush()

            done_offsets.add(start)
            state["done_offsets"] = list(done_offsets)
            state["total_words"] = state.get("total_words", 0) + sum(
                len(c.split()) for _, _, c in pages)
            state_io.save_namespace(STATE_FILE, NAMESPACE, state)

            batch_chunks += 1
            if batch_chunks % LOG_EVERY_N_CHUNKS == 0:
                elapsed = time.monotonic() - batch_start
                log(f"wiki_dump_parallel: {batch_chunks} chunks / {batch_pages} pages this batch, "
                    f"~{batch_words} words this batch, {elapsed:.0f}s elapsed")

            elapsed = time.monotonic() - batch_start
            stop = batch_chunks >= args.max_chunks or elapsed >= args.max_seconds
            if not stop:
                _top_up()

    elapsed = time.monotonic() - batch_start
    log(f"BATCH COMPLETE: {batch_chunks} chunks / {batch_pages} pages extracted this batch, "
        f"~{batch_words} words this batch, {elapsed:.0f}s elapsed. "
        f"Overall: {len(done_offsets)}/{len(chunks)} chunks done, "
        f"~{state['total_words']} words total ever.")


if __name__ == "__main__":
    main()
