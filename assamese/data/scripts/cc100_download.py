#!/usr/bin/env python3
"""One-shot downloader for CC-100's Assamese slice (data.statmt.org).

CC-100 (Conneau et al. 2020, recreation of the XLM-R training corpus) is a
CommonCrawl-derived monolingual web corpus -- a different source lineage
from Sangraha/MWirelabs (both trace back through Samanantar) and from the
Wikimedia dumps, so it's not expected to substantially overlap. Assamese's
slice is tiny (7.6MB compressed) compared to higher-resource languages in
the same corpus -- reflects how little Assamese web text CommonCrawl
actually indexed, not a bug in this script.

Downloaded corpus, not manual collection -- same "downloaded" bucket as
Sangraha/MWirelabs. Writes true raw text; cleaning happens read-only at
count time in token_tracker.py (registered there as ("raw/cc100",
"downloaded")), matching every other downloaded-bucket source in this tree.

A single static file, not a crawl -- no resumability/checkpoint state
needed. Safe to just rerun; skips the download if the raw file already
exists.
"""

import lzma
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
OUT_PATH = DATA_DIR / "raw" / "cc100" / "as.txt"
DUMP_URL = "https://data.statmt.org/cc-100/as.txt.xz"
USER_AGENT = "Mozilla/5.0 (compatible; lma-individual-project-research-bot)"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        print(f"already downloaded: {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
        return

    print(f"downloading {DUMP_URL}")
    req = urllib.request.Request(DUMP_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        compressed = resp.read()
    print(f"downloaded {len(compressed)} compressed bytes, decompressing")

    text = lzma.decompress(compressed).decode("utf-8")
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_PATH}: {len(text)} chars")


if __name__ == "__main__":
    main()
