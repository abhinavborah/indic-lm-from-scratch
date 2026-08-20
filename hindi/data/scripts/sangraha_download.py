#!/usr/bin/env python3
"""Streaming downloader for ai4bharat/sangraha: Hindi only.

Configs pulled: "verified" and "unverified". Never "synthetic" (that split
is machine-translated + romanized text, not organic monolingual prose, so
it's excluded from the public/downloaded corpus entirely).

Sampling strategy: stream until a rough whitespace-token cap is hit, rather
than pulling the full multi-billion-token split, leaving headroom for the
manual-collection quota on top of the downloaded portion.

Resumable: checkpoints into hindi/data/.state.json under the "sangraha"
namespace (same file ocr_pipeline.py and scraper.py checkpoint into, under
their own "ocr"/"scrape" namespaces: state_io.py's flock-protected
save/load keeps concurrent scripts from clobbering each other). Safe to
interrupt and rerun: already-consumed docs are skipped via
IterableDataset.skip(n).

Writes plain-text shard files into hindi/data/raw/sangraha/, one file per
<config, shard index>, rotated every SHARD_TOKEN_LIMIT rough tokens.
"""

import re
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

import state_io

LANG = "hin"
DATA_DIR = Path(__file__).resolve().parents[1]  # hindi/data
OUT_DIR = DATA_DIR / "raw" / "sangraha"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "SANGRAHA_LOG.md"
NAMESPACE = "sangraha"

CONFIGS = ["verified", "unverified"]  # never "synthetic"
TARGET_TOKENS = 450_000_000

SHARD_TOKEN_LIMIT = 5_000_000
LOG_EVERY_N_DOCS = 500

TOKEN_RE = re.compile(r"\S+")


def rough_token_count(text):
    return len(TOKEN_RE.findall(text))


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def default_state():
    return {cfg: {"docs_done": 0, "tokens_done": 0, "shard_idx": 0,
                  "shard_tokens": 0, "done": False} for cfg in CONFIGS}


def load_state():
    return state_io.load_namespace(STATE_FILE, NAMESPACE, default_state())


def save_state(state):
    state_io.save_namespace(STATE_FILE, NAMESPACE, state)


def total_tokens(state):
    return sum(c["tokens_done"] for c in state.values())


def shard_path(config, shard_idx):
    return OUT_DIR / f"{config}_shard{shard_idx:05d}.txt"


def process_config(config, state):
    entry = state[config]
    skip_n = entry["docs_done"]
    log(f"[{config}] resuming at doc {skip_n} "
        f"({entry['tokens_done']:,} tokens so far, shard {entry['shard_idx']})")

    ds = load_dataset("ai4bharat/sangraha", data_dir=f"{config}/{LANG}",
                       split="train", streaming=True)
    if skip_n:
        # ponytail: IterableDataset.skip() re-reads and discards every prior
        # doc from the start of the stream, so resume cost grows with
        # docs_done. Fine at current scale (network is the bottleneck
        # either way); upgrade to tracking parquet-file+row-group offsets
        # if resume overhead becomes noticeable.
        ds = ds.skip(skip_n)

    out_path = shard_path(config, entry["shard_idx"])
    out_f = out_path.open("a" if out_path.exists() else "w", encoding="utf-8")

    docs_since_log = 0
    hit_target = False
    try:
        for row in ds:
            text = (row.get("text") or "").strip()
            entry["docs_done"] += 1
            if not text:
                continue

            out_f.write(text + "\n\n")
            out_f.flush()

            n_tok = rough_token_count(text)
            entry["tokens_done"] += n_tok
            entry["shard_tokens"] += n_tok
            save_state(state)

            docs_since_log += 1
            if docs_since_log >= LOG_EVERY_N_DOCS:
                log(f"[{config}] doc {entry['docs_done']}, "
                    f"{entry['tokens_done']:,} tokens this config, "
                    f"{total_tokens(state):,} tokens total")
                docs_since_log = 0

            if entry["shard_tokens"] >= SHARD_TOKEN_LIMIT:
                out_f.close()
                entry["shard_idx"] += 1
                entry["shard_tokens"] = 0
                out_f = shard_path(config, entry["shard_idx"]).open("w", encoding="utf-8")

            if total_tokens(state) >= TARGET_TOKENS:
                log(f"reached target {TARGET_TOKENS:,} rough tokens; stopping {config}")
                entry["done"] = True
                hit_target = True
                break
        else:
            entry["done"] = True
            log(f"[{config}] stream exhausted: {entry['tokens_done']:,} tokens, "
                f"{entry['docs_done']} docs")
    finally:
        out_f.close()
        save_state(state)

    return hit_target


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()

    for config in CONFIGS:
        if state[config].get("done"):
            log(f"[{config}] already marked done, skipping")
            continue
        if process_config(config, state):
            break

    log(f"pass complete: {total_tokens(state):,} rough tokens across "
        f"{sum(c['docs_done'] for c in state.values())} docs")


if __name__ == "__main__":
    main()
