#!/usr/bin/env python3
"""Final preprocessing pipeline for Hindi: dedup + purity filter + split.

Reuses exactly the same source list, segment iteration, and dedup logic as
token_tracker.py so the split's totals match what's been reported all
session -- this isn't a second, independently-tuned cleaning pass, it's the
same filter, now actually writing the surviving text out instead of just
counting it.

Split method: each blank-line-delimited segment (the same unit dedup
operates on -- one article, OCR page, dictionary entry, etc.) is assigned
to train/val/test by hashing its own cleaned content mod 100. This is
single-pass (no need to know the corpus size in advance, no second scan)
and fully deterministic and reproducible from the corpus alone -- rerunning
this script on the same corpus always produces the same split, with no
separate random seed to track. ~98% train / 1% val / 1% test: this is
current standard LM-pretraining practice (see nanoGPT's ~99.95/0.05
split on OpenWebText), not the 80/10/10 convention from classification
tasks -- pretraining corpora are large enough that a held-out set doesn't
need to scale proportionally with corpus size for a statistically stable
loss estimate.
"""

import hashlib
from pathlib import Path

from clean_text import clean_text
from token_tracker import DATA_DIR, SOURCES, iter_segments, rough_token_count

SPLITS_DIR = DATA_DIR / "splits"

TRAIN_CUTOFF = 98  # [0, 98) -> train
VAL_CUTOFF = 99    # [98, 99) -> val, [99, 100) -> test


def bucket_for(cleaned_text):
    h = hashlib.blake2b(cleaned_text.encode("utf-8"), digest_size=8).digest()
    bucket_id = int.from_bytes(h, "big") % 100
    if bucket_id < TRAIN_CUTOFF:
        return "train"
    if bucket_id < VAL_CUTOFF:
        return "val"
    return "test"


def build():
    seen_hashes = set()
    counts = {"train": 0, "val": 0, "test": 0}
    tokens = {"train": 0, "val": 0, "test": 0}
    segments_scanned = 0
    segments_impure = 0
    segments_duplicate = 0

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    files = {name: (SPLITS_DIR / f"{name}.txt").open("w", encoding="utf-8")
             for name in ("train", "val", "test")}
    try:
        for rel_path, _bucket in SOURCES:
            src_dir = DATA_DIR / rel_path
            if not src_dir.is_dir():
                continue
            for path in sorted(src_dir.rglob("*.txt")):
                for seg in iter_segments(path):
                    stripped = seg.strip()
                    if not stripped:
                        continue
                    segments_scanned += 1

                    cleaned = clean_text(stripped)
                    if not cleaned.strip():
                        segments_impure += 1
                        continue

                    h = hashlib.blake2b(cleaned.encode("utf-8"), digest_size=16).digest()
                    if h in seen_hashes:
                        segments_duplicate += 1
                        continue
                    seen_hashes.add(h)

                    split = bucket_for(cleaned)
                    files[split].write(cleaned + "\n\n")
                    counts[split] += 1
                    tokens[split] += rough_token_count(cleaned)
    finally:
        for f in files.values():
            f.close()

    return {
        "counts": counts, "tokens": tokens,
        "segments_scanned": segments_scanned,
        "segments_impure": segments_impure,
        "segments_duplicate": segments_duplicate,
    }


def main():
    print("building hindi splits...", flush=True)
    stats = build()
    total_tokens = sum(stats["tokens"].values())
    for split in ("train", "val", "test"):
        pct = stats["tokens"][split] / total_tokens * 100 if total_tokens else 0
        print(f"  {split}: {stats['counts'][split]:,} documents, "
              f"{stats['tokens'][split]:,} rough tokens ({pct:.2f}%)", flush=True)
    print(f"  scanned {stats['segments_scanned']:,} segments, "
          f"dropped {stats['segments_impure']:,} impure, "
          f"{stats['segments_duplicate']:,} duplicate", flush=True)
    print(f"written to {SPLITS_DIR}", flush=True)


if __name__ == "__main__":
    main()
