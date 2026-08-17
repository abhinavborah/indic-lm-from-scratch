#!/usr/bin/env python3
"""Streaming downloader for MWirelabs/assamese-monolingual-corpus (HuggingFace).

This is a pre-cleaned, pre-deduplicated third-party corpus (IITB-IndicMonoDoc
+ Samanantar Assamese side + Assamese poetry/civic texts), not something we
scrape or preprocess ourselves -- same "downloaded" bucket as Sangraha, not
"manual". Added specifically to help close the 500M *total* token target
after the 100M *manual* floor was found to be unreachable with current
manual sources (see report/token_progress.md).

License: CC BY-SA 4.0 -- attribution required, must be cited in
report/SOURCES.md (already done). Some overlap with Sangraha is expected
since both draw on Samanantar; token_tracker.py's exact-duplicate dedup
handles the overlap, so the net gain will be less than this dataset's raw
77.4M-token size.

Resumable: checkpoints into assamese/data/.state.json under the
"mwirelabs" namespace. Writes plain-text shard files into
assamese/data/raw/mwirelabs/, one file per shard index, rotated every
SHARD_TOKEN_LIMIT rough tokens -- token_tracker.py needs a
("raw/mwirelabs", "downloaded") entry in its SOURCES list to pick this up
(already added).
"""

import re
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

import state_io

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
OUT_DIR = DATA_DIR / "raw" / "mwirelabs"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
NAMESPACE = "mwirelabs"

DATASET_ID = "MWirelabs/assamese-monolingual-corpus"
SHARD_TOKEN_LIMIT = 5_000_000
LOG_EVERY_N_DOCS = 2000

TOKEN_RE = re.compile(r"\S+")


def rough_token_count(text):
    return len(TOKEN_RE.findall(text))


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def load_state():
    return state_io.load_namespace(
        STATE_FILE, NAMESPACE,
        {"rows_done": 0, "tokens_done": 0, "shard_idx": 0, "shard_tokens": 0, "done": False},
    )


def save_state(state):
    state_io.save_namespace(STATE_FILE, NAMESPACE, state)


def shard_path(shard_idx):
    return OUT_DIR / f"mwirelabs_shard{shard_idx:05d}.txt"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if state["done"]:
        log("already marked done, skipping")
        return

    skip_n = state["rows_done"]
    log(f"resuming at row {skip_n} ({state['tokens_done']:,} tokens so far, shard {state['shard_idx']})")

    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    if skip_n:
        ds = ds.skip(skip_n)

    out_path = shard_path(state["shard_idx"])
    out_f = out_path.open("a" if out_path.exists() else "w", encoding="utf-8")

    docs_since_log = 0
    try:
        for row in ds:
            text = (row.get("text") or "").strip()
            state["rows_done"] += 1
            if not text:
                continue

            out_f.write(text + "\n\n")
            out_f.flush()

            n_tok = rough_token_count(text)
            state["tokens_done"] += n_tok
            state["shard_tokens"] += n_tok
            save_state(state)

            docs_since_log += 1
            if docs_since_log >= LOG_EVERY_N_DOCS:
                log(f"row {state['rows_done']}, {state['tokens_done']:,} tokens so far")
                docs_since_log = 0

            if state["shard_tokens"] >= SHARD_TOKEN_LIMIT:
                out_f.close()
                state["shard_idx"] += 1
                state["shard_tokens"] = 0
                out_f = shard_path(state["shard_idx"]).open("w", encoding="utf-8")
        else:
            state["done"] = True
            log(f"stream exhausted -- {state['tokens_done']:,} tokens, {state['rows_done']} rows")
    finally:
        out_f.close()
        save_state(state)

    log(f"pass complete: {state['tokens_done']:,} rough tokens across {state['rows_done']} rows")


if __name__ == "__main__":
    main()
