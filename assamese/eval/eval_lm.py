#!/usr/bin/env python3
"""Intrinsic language-modeling evaluation: cross-entropy, perplexity, and
bits-per-byte over a full held-out split, per spec section 2.3.

Forward-only, no gradients. Splits are walked as non-overlapping
context_length windows (same windowing convention as get_batch/
get_batch_memmap elsewhere in this project), covering every full window in
the split, no sampling, per project decision 2026-08-30.

Run directly: python3 eval_lm.py --checkpoint-path /path/to/assamese_checkpoint.pt
"""

import argparse
import json
import math
import sys
from pathlib import Path

import sentencepiece as spm
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "train"))

from model import DecoderLM  # noqa: E402
from train import load_checkpoint  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "model_config.json"
TOKENIZER_PATH = ROOT / "tokenizer" / "assamese_bpe_8000.model"
DEFAULT_DATA_DIR = ROOT / "data" / "splits"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def tokenize_text(text_path, tokenizer_path):
    """Encode a full text split to a 1D token tensor, and return the raw
    UTF-8 byte length of the source text alongside it (needed for the BPB
    denominator). val/test splits are at most a few tens of MB, small
    enough to read and encode whole, unlike train.txt, which needs the
    chunked streaming tokenizer used elsewhere in this project."""
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    ids = sp.encode(text, out_type=int)
    byte_length = len(text.encode("utf-8"))
    return torch.tensor(ids, dtype=torch.long), byte_length


def evaluate_split(model, data, batch_size, context_length, device):
    """Sum-reduced cross-entropy over every non-overlapping context_length
    window in data, batched for throughput. Returns (mean_loss_nats,
    num_tokens_evaluated); summing losses and dividing once at the end
    (rather than averaging per-batch means) keeps the result exact
    regardless of the final, possibly smaller, batch."""
    model.eval()
    num_windows = (len(data) - 1) // context_length
    total_loss_sum = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, num_windows, batch_size):
            window_ids = range(start, min(start + batch_size, num_windows))
            xs, ys = [], []
            for w in window_ids:
                s = w * context_length
                xs.append(data[s : s + context_length])
                ys.append(data[s + 1 : s + context_length + 1])
            x = torch.stack(xs).to(device)
            y = torch.stack(ys).to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum",
            )
            total_loss_sum += loss.item()
            total_tokens += y.numel()
    model.train()
    return total_loss_sum / total_tokens, total_tokens


def compute_metrics(mean_loss_nats, num_tokens, byte_length):
    """PPL = e^CrossEntropyLoss, per spec. BPB: convert the per-token nats
    loss to bits (times log2(e)), scale by tokens evaluated, divide by the
    source text's raw UTF-8 byte length. A remainder of fewer than
    context_length tokens at the end of the split is dropped by the
    non-overlapping windowing and not reflected in byte_length,
    negligible against splits of this size (a few million tokens),
    not worth tracking exact per-window byte offsets for."""
    ppl = math.exp(mean_loss_nats)
    total_bits = mean_loss_nats * math.log2(math.e) * num_tokens
    bpb = total_bits / byte_length
    return ppl, bpb


def run(checkpoint_path, data_dir=None, batch_size=32, splits=("val", "test")):
    full_config = load_model_config()
    model_config = full_config["model"]
    context_length = model_config["context_length"]

    device = pick_device()
    print(f"device={device}")

    model = DecoderLM(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.1)
    step = load_checkpoint(checkpoint_path, model, optimizer, map_location=device)
    model.to(device)
    print(f"loaded checkpoint at step {step:,}")

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    results = {}
    for split in splits:
        text_path = data_dir / f"{split}.txt"
        data, byte_length = tokenize_text(text_path, TOKENIZER_PATH)
        mean_loss_nats, num_tokens = evaluate_split(model, data, batch_size, context_length, device)
        ppl, bpb = compute_metrics(mean_loss_nats, num_tokens, byte_length)
        results[split] = {
            "cross_entropy_nats": round(mean_loss_nats, 4),
            "perplexity": round(ppl, 2),
            "bits_per_byte": round(bpb, 4),
            "tokens_evaluated": num_tokens,
            "byte_length": byte_length,
        }
        print(f"{split}: loss={mean_loss_nats:.4f} ppl={ppl:.2f} bpb={bpb:.4f} "
              f"({num_tokens:,} tokens)")

    return {"language": "assamese", "checkpoint_step": step, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default=str(ROOT / "eval" / "lm_metrics.json"))
    args = parser.parse_args()

    output = run(args.checkpoint_path, data_dir=args.data_dir, batch_size=args.batch_size)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
