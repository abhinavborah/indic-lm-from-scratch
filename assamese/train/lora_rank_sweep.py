#!/usr/bin/env python3
"""Empirical LoRA rank/alpha sweep on real data, since no literature exists
at this model's scale (24.3M params, d_model=384) to borrow a default from
-- see docs-phase-3/finetune_method_choice.md's "open, not yet decided"
section. Same pattern as Phase 2's own LR/batch-size sweeps
(smoke_test.py): pick the smallest rank that gets within a small tolerance
of the best reasoning-val loss, favoring more regularization (per this
project's actual reason for choosing LoRA: avoiding overfitting on a tiny
6000-example set), while checking pretrain-val PPL doesn't blow up.

Uses the real Phase 2 pretrained checkpoint and a subset of the real
generated reasoning data -- this is a short, cheap sweep to pick one
hyperparameter, not the final training run.

Run directly: python3 lora_rank_sweep.py --pretrained-checkpoint PATH
    --pretrain-val-bin PATH
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from finetune import (
    PRETRAIN_EVAL_SEED, finetune, freeze_non_lora_params, inject_lora, load_pretrained_weights,
    load_reasoning_examples, pretrain_val_ppl,
)
from model import DecoderLM

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "assamese" / "configs" / "model_config.json"
TOKENIZER_PATH = REPO_ROOT / "assamese" / "tokenizer" / "assamese_bpe_8000.model"
REASONING_DIR = REPO_ROOT / "assamese" / "data" / "reasoning"

RANK_CANDIDATES = [2, 4, 8]
ALPHA_MULTIPLIER = 2  # alpha = ALPHA_MULTIPLIER * rank, a common default absent scale-specific guidance
SWEEP_EPOCHS = 5  # short and cheap, just to rank candidates against each other, not a final run
TRAIN_SUBSET = 1200  # subset of the real 4500 train examples, for sweep speed
PEAK_LR = 1e-4  # within the CeADAR guide's cited 1e-4 to 2e-4 range (sft_ultimate_guide.md)
SEED = 20260914


def load_pretrain_val_tensor(path):
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    return torch.from_numpy(arr.astype(np.int64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--pretrain-val-bin", required=True)
    args = parser.parse_args()
    pretrained_ckpt = Path(args.pretrained_checkpoint)
    pretrain_val_bin = Path(args.pretrain_val_bin)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        full_config = json.load(f)
    model_config = full_config["model"]
    opt_config = full_config["optimizer"]

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))

    train_examples = load_reasoning_examples(REASONING_DIR / "train.jsonl")[:TRAIN_SUBSET]
    val_examples = load_reasoning_examples(REASONING_DIR / "val.jsonl")
    pretrain_val_data = load_pretrain_val_tensor(pretrain_val_bin)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}, train subset: {len(train_examples)}, val: {len(val_examples)}, "
          f"pretrain val tokens: {len(pretrain_val_data):,}")

    base_model = DecoderLM(model_config)
    load_pretrained_weights(pretrained_ckpt, base_model)
    base_model.to(device)
    baseline_loss, baseline_ppl = pretrain_val_ppl(
        base_model, pretrain_val_data, model_config["context_length"], device, seed=PRETRAIN_EVAL_SEED
    )
    print(f"baseline (no finetune) pretrain-val PPL: {baseline_ppl:.2f}")

    results = []
    for rank in RANK_CANDIDATES:
        alpha = ALPHA_MULTIPLIER * rank
        torch.manual_seed(SEED)
        model = copy.deepcopy(base_model).to("cpu")
        inject_lora(model, rank=rank, alpha=alpha)
        freeze_non_lora_params(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            betas=(opt_config["beta1"], opt_config["beta2"]), weight_decay=opt_config["weight_decay"],
        )

        log = finetune(
            model, optimizer, sp, train_examples, val_examples, pretrain_val_data,
            num_epochs=SWEEP_EPOCHS, batch_size=32, context_length=model_config["context_length"],
            peak_lr=PEAK_LR, warmup_steps=10, grad_clip_norm=opt_config["grad_clip_norm"],
            seed=SEED, device=device,
        )
        final = log[-1]
        print(f"rank={rank} alpha={alpha} trainable_params={trainable:,} "
              f"reasoning_val_loss={final['reasoning_val_loss']:.4f} "
              f"pretrain_val_ppl={final['pretrain_val_ppl']:.2f}")
        results.append({
            "rank": rank, "alpha": alpha, "trainable_params": trainable,
            "reasoning_val_loss": final["reasoning_val_loss"],
            "pretrain_val_ppl": final["pretrain_val_ppl"],
            "log": log,
        })

    best = min(r["reasoning_val_loss"] for r in results)
    tolerance = 0.02  # absolute loss tolerance for preferring a smaller (more regularized) rank
    within_tolerance = [r for r in results if r["reasoning_val_loss"] <= best + tolerance]
    winner = min(within_tolerance, key=lambda r: r["rank"])

    print("\n--- sweep result ---")
    for r in results:
        marker = " <-- picked" if r is winner else ""
        print(f"rank={r['rank']:2d} alpha={r['alpha']:2d} params={r['trainable_params']:,} "
              f"reasoning_val_loss={r['reasoning_val_loss']:.4f} "
              f"pretrain_val_ppl={r['pretrain_val_ppl']:.2f} (baseline {baseline_ppl:.2f}){marker}")

    out_path = REASONING_DIR.parent / "lora_rank_sweep_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_pretrain_val_ppl": baseline_ppl,
            "candidates": [{k: v for k, v in r.items() if k != "log"} | {"log": r["log"]} for r in results],
            "picked_rank": winner["rank"],
            "picked_alpha": winner["alpha"],
            "tolerance": tolerance,
            "sweep_epochs": SWEEP_EPOCHS,
            "train_subset": TRAIN_SUBSET,
            "peak_lr": PEAK_LR,
            "seed": SEED,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
