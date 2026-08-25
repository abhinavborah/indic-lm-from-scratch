#!/usr/bin/env python3
"""Local MPS smoke test: run the real pipeline end to end on real data
before touching Colab. Does three things a synthetic unit test can't:

1. Picks peak_lr/min_lr from a real small sweep (deliberately, per
   docs-phase-2/training_diagnostics.md -- "peak LR is the single most
   important hyperparameter", not guessed) and writes the result into
   model_config.json, replacing the placeholder nulls.
2. Confirms the whole pipeline (real tokenizer, real train/val splits,
   real model) runs on the M4 Pro's MPS backend and produces a falling
   loss curve with a val loss that stays close to train loss -- the
   "healthy loss curve" diagnostic from training_diagnostics.md.
3. Measures real MPS tokens/sec throughput, the number
   docs-phase-2/implementation_plan.md said not to guess before committing
   to a Colab training-duration schedule.

Run directly: python3 smoke_test.py
"""

import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model import DecoderLM  # noqa: E402

from train import get_batch, lr_at_step, save_checkpoint, load_checkpoint, tokenize_corpus  # noqa: E402

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "model_config.json"
TOKENIZER_PATH = Path(__file__).parent.parent / "tokenizer" / "assamese_bpe_8000.model"
TRAIN_TEXT_PATH = Path(__file__).parent.parent / "data" / "splits" / "train.txt"
VAL_TEXT_PATH = Path(__file__).parent.parent / "data" / "splits" / "val.txt"

TRAIN_SAMPLE_CHARS = 2_000_000
VAL_SAMPLE_CHARS = 500_000

BATCH_SIZE = 8
SWEEP_STEPS = 50
SWEEP_CANDIDATES = [1e-4, 3e-4, 6e-4, 1e-3]
SMOKE_STEPS = 300
SMOKE_WARMUP_STEPS = 30
VAL_EVERY = 50
GRAD_CLIP = 1.0


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_model_config(full_config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=2, ensure_ascii=False)
        f.write("\n")


def train_steps(model, data, optimizer, num_steps, batch_size, context_length,
                 warmup_steps, peak_lr, min_lr, grad_clip_norm, device, total_steps=None):
    """Minimal re-implementation of train.py's loop body, kept local so this
    script can log per-step loss without threading a callback through
    train.py's reusable function."""
    if total_steps is None:
        total_steps = num_steps
    model.to(device)
    model.train()
    losses = []
    for step in range(num_steps):
        lr = lr_at_step(step, warmup_steps, peak_lr, min_lr, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        torch.manual_seed(step)
        x, y = get_batch(data, batch_size, context_length, step)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        losses.append(loss.item())
    return losses


def sweep_peak_lr(model_config, data, device):
    """Try a few candidate peak_lr values for a short run each, from a
    freshly initialized model every time so the comparison is fair. Picks
    the lowest last-10-step average loss among candidates that actually
    decreased (rules out the 'LR too high, bounces/plateaus' failure mode
    from training_diagnostics.md)."""
    results = []
    for candidate in SWEEP_CANDIDATES:
        torch.manual_seed(0)
        model = DecoderLM(copy.deepcopy(model_config))
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.1)
        losses = train_steps(
            model, data, optimizer, SWEEP_STEPS, BATCH_SIZE, model_config["context_length"],
            warmup_steps=5, peak_lr=candidate, min_lr=candidate / 10, grad_clip_norm=GRAD_CLIP,
            device=device,
        )
        first_avg = sum(losses[:10]) / 10
        last_avg = sum(losses[-10:]) / 10
        decreased = last_avg < first_avg
        finite = all(l == l and abs(l) != float("inf") for l in losses)  # not NaN, not inf
        results.append({
            "peak_lr": candidate, "first_avg": first_avg, "last_avg": last_avg,
            "decreased": decreased, "finite": finite,
        })
        print(f"  peak_lr={candidate:.0e}: first10_avg={first_avg:.4f} "
              f"last10_avg={last_avg:.4f} decreased={decreased} finite={finite}")

    viable = [r for r in results if r["decreased"] and r["finite"]]
    assert viable, "no candidate LR produced a decreasing, finite loss -- sweep range is wrong"
    best = min(viable, key=lambda r: r["last_avg"])
    return best["peak_lr"], results


def demo():
    device = pick_device()
    print(f"device: {device}")

    full_config = load_model_config()
    model_config = full_config["model"]

    train_data = tokenize_corpus(TRAIN_TEXT_PATH, TOKENIZER_PATH, max_chars=TRAIN_SAMPLE_CHARS)
    val_data = tokenize_corpus(VAL_TEXT_PATH, TOKENIZER_PATH, max_chars=VAL_SAMPLE_CHARS)
    print(f"train sample: {len(train_data):,} tokens, val sample: {len(val_data):,} tokens")

    print("peak_lr sweep:")
    best_peak_lr, sweep_results = sweep_peak_lr(model_config, train_data, device)
    best_min_lr = best_peak_lr / 10
    print(f"picked peak_lr={best_peak_lr:.0e}, min_lr={best_min_lr:.0e}")

    torch.manual_seed(0)
    model = DecoderLM(copy.deepcopy(model_config))
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.1)

    print(f"running {SMOKE_STEPS}-step smoke run on {device}...")
    start = time.time()
    train_losses = []
    val_losses = {}
    model.to(device)
    for step in range(SMOKE_STEPS):
        lr = lr_at_step(step, SMOKE_WARMUP_STEPS, best_peak_lr, best_min_lr, SMOKE_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = lr

        torch.manual_seed(step)
        x, y = get_batch(train_data, BATCH_SIZE, model_config["context_length"], step)
        x, y = x.to(device), y.to(device)

        model.train()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_losses.append(loss.item())

        if (step + 1) % VAL_EVERY == 0 or step == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, BATCH_SIZE, model_config["context_length"], step)
                vx, vy = vx.to(device), vy.to(device)
                vlogits = model(vx)
                vloss = torch.nn.functional.cross_entropy(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
            val_losses[step] = vloss.item()
            print(f"  step {step + 1}/{SMOKE_STEPS}: train_loss={loss.item():.4f} val_loss={vloss.item():.4f} lr={lr:.2e}")

    elapsed = time.time() - start
    tokens_processed = SMOKE_STEPS * BATCH_SIZE * model_config["context_length"]
    tokens_per_sec = tokens_processed / elapsed
    print(f"elapsed: {elapsed:.1f}s, throughput: {tokens_per_sec:,.0f} tokens/sec on {device}")

    expected_l0 = torch.log(torch.tensor(float(model_config["vocab_size"]))).item()
    assert abs(train_losses[0] - expected_l0) < 1.0, \
        f"first real-data loss {train_losses[0]:.4f} far from L0={expected_l0:.4f} -- pipeline is likely broken"

    first_avg = sum(train_losses[:10]) / 10
    last_avg = sum(train_losses[-10:]) / 10
    assert last_avg < first_avg, "smoke run's loss did not decrease overall"
    assert all(l == l and abs(l) != float("inf") for l in train_losses), \
        "NaN or Inf appeared in the training loss -- training diverged"

    final_val = val_losses[max(val_losses)]
    assert abs(last_avg - final_val) < 3.0, \
        f"train/val loss gap too large (train={last_avg:.4f}, val={final_val:.4f}) -- possible overfitting even at this small scale"

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "smoke_checkpoint.pt"
        save_checkpoint(ckpt_path, model, optimizer, SMOKE_STEPS, model_config)
        reloaded_model = DecoderLM(copy.deepcopy(model_config))
        reloaded_optimizer = torch.optim.AdamW(reloaded_model.parameters(), betas=(0.9, 0.95), weight_decay=0.1)
        resumed_step = load_checkpoint(ckpt_path, reloaded_model, reloaded_optimizer, map_location=device)
        assert resumed_step == SMOKE_STEPS
        reloaded_model.to(device)
        reloaded_model.eval()
        model.eval()
        x, _ = get_batch(train_data, BATCH_SIZE, model_config["context_length"], 0)
        x = x.to(device)
        with torch.no_grad():
            out_original = model(x)
            out_reloaded = reloaded_model(x)
        assert torch.allclose(out_original, out_reloaded, atol=1e-5), \
            "reloaded checkpoint produces different output than the in-memory model on real data"
    print("checkpoint reload on real data: OK")

    full_config["schedule"]["peak_lr"] = round(best_peak_lr, 8)
    full_config["schedule"]["min_lr"] = round(best_min_lr, 8)
    full_config["schedule"]["notes"] = (
        f"picked via smoke_test.py's LR sweep on real data ({device}); "
        f"candidates {SWEEP_CANDIDATES} over {SWEEP_STEPS} steps each, "
        f"min_lr = peak_lr / 10 (standard nanoGPT convention). "
        f"See viva-lma.md for the sweep results."
    )
    full_config["smoke_test_throughput"] = {
        "device": device, "tokens_per_sec": round(tokens_per_sec, 1),
        "batch_size": BATCH_SIZE, "context_length": model_config["context_length"],
    }
    save_model_config(full_config)
    print(f"model_config.json updated with peak_lr={best_peak_lr:.0e}, min_lr={best_min_lr:.0e}")

    print("assamese local MPS smoke test: OK")


if __name__ == "__main__":
    demo()
