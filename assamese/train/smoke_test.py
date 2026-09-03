#!/usr/bin/env python3
"""Local MPS smoke test: run the real pipeline end to end on real data
before touching Colab. Does four things a synthetic unit test can't:

1. Picks peak_lr/min_lr from a real small sweep (deliberately, per
   docs-phase-2/training_diagnostics.md, "peak LR is the single most
   important hyperparameter", not guessed) and writes the result into
   model_config.json, replacing the placeholder nulls.
2. Picks batch_size from a real sweep too, not a guessed default. Cheap
   proxy for the critical-batch-size methodology (Merrill et al. 2025,
   "Critical Batch Size Revisited"; McCandlish et al. 2018, "An Empirical
   Model of Large-Batch Training"): rather than full training runs per
   candidate, measures throughput and a fixed-token-budget loss for each
   candidate batch size, and picks the fastest one that doesn't degrade
   loss-per-token meaningfully versus the baseline.
3. Confirms the whole pipeline (real tokenizer, real train/val splits,
   real model) runs on the M4 Pro's MPS backend and produces a falling
   loss curve with a val loss that stays close to train loss, the
   "healthy loss curve" diagnostic from training_diagnostics.md.
4. Measures real MPS tokens/sec throughput at the winning batch size, the
   number docs-phase-2/implementation_plan.md said not to guess before
   committing to a Colab training-duration schedule.

Run directly: python3 smoke_test.py
"""

import copy
import gc
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import torch


def peak_rss_gb():
    """This process's peak resident memory so far, in GB. ru_maxrss is
    bytes on macOS (Darwin) but KB on Linux; this project only runs
    locally on the M4 Pro, but the unit split is a classic gotcha worth
    getting right rather than silently wrong on a different machine."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 ** 3) if sys.platform == "darwin" else raw / (1024 ** 2)


def release_mps_memory():
    """Force a GC pass and release MPS's cached buffers back to the system.
    Must be called AFTER the caller has already deleted its own references
    to the short-lived model/optimizer (del inside this function would only
    drop this function's own local binding, not the caller's; the object
    would still be alive). Without this, a sweep loop that creates a fresh
    model+optimizer per candidate (the LR sweep does 4, the batch-size
    sweep does 8, a re-sweep does 4 more) accumulates memory across
    candidates instead of releasing it between them; on a real run this
    drove physical memory to 26.6GB (near the M4 Pro's full 24GB unified
    memory plus swap), stalling the whole process on memory pressure
    rather than actually computing."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model import DecoderLM  # noqa: E402

from train import get_batch, lr_at_step, save_checkpoint, load_checkpoint, tokenize_corpus  # noqa: E402

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "model_config.json"
TOKENIZER_PATH = Path(__file__).parent.parent / "tokenizer" / "assamese_bpe_8000.model"
TRAIN_TEXT_PATH = Path(__file__).parent.parent / "data" / "splits" / "train.txt"
VAL_TEXT_PATH = Path(__file__).parent.parent / "data" / "splits" / "val.txt"

TRAIN_SAMPLE_CHARS = 2_000_000
VAL_SAMPLE_CHARS = 500_000

SWEEP_BATCH_SIZE = 8  # anchor batch size for the initial LR sweep and linear-scaling reference
SWEEP_STEPS = 50
SWEEP_CANDIDATES = [1e-4, 3e-4, 6e-4, 1e-3]
SMOKE_STEPS = 300
SMOKE_WARMUP_STEPS = 30
VAL_EVERY = 50
GRAD_CLIP = 1.0

# Batch-size sweep: candidates to try, a fixed token budget per candidate
# (not fixed steps: bigger batches take fewer, larger steps to cover the
# same tokens, so token budget is the fair comparison unit), a throughput
# probe length, and how much worse (fractionally) a bigger batch's
# fixed-token-budget loss is allowed to be before it's rejected in favor
# of a smaller, slower-but-not-degraded candidate. Kept to 128 as the
# starting ceiling, not 256; activation memory scales roughly linearly
# with batch size, and a real run hit 26.6GB physical footprint (near the
# M4 Pro's full 24GB unified memory) partly from candidates this large,
# not just from the cross-candidate leak that's also fixed below.
BATCH_CANDIDATES = [8, 16, 32, 64]  # 8/16 added to test below the baseline too, not just
# above it (the original sweep only ever tested going bigger than 32, never justified
# treating 32 as a floor). 128 dropped: already hit an unbounded MPS shader-compile
# stall on this hardware in an earlier run (hardware-level, not data-dependent), no new
# information from re-triggering the same 12+ minute stall here
BATCH_THROUGHPUT_STEPS = 50
BATCH_TOKEN_BUDGET = 2_000_000
BATCH_QUALITY_TOLERANCE = 0.05
BASELINE_BATCH_SIZE = 32  # current default, the comparison anchor for the tolerance check
MAX_PEAK_RSS_GB = 12  # conservative ceiling given other processes share this machine's memory


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


def sweep_peak_lr(model_config, data, device, batch_size=SWEEP_BATCH_SIZE):
    """Try a few candidate peak_lr values for a short run each, from a
    freshly initialized model every time so the comparison is fair. Picks
    the lowest last-10-step average loss among candidates that actually
    decreased (rules out the 'LR too high, bounces/plateaus' failure mode
    from training_diagnostics.md). Parameterized on batch_size so it can be
    re-run at whichever batch size the batch-size sweep picks, not just
    the fixed anchor batch size."""
    results = []
    for candidate in SWEEP_CANDIDATES:
        print(f"  [starting] peak_lr={candidate:.0e} at batch_size={batch_size}...", flush=True)
        torch.manual_seed(0)
        model = DecoderLM(copy.deepcopy(model_config))
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.1)
        losses = train_steps(
            model, data, optimizer, SWEEP_STEPS, batch_size, model_config["context_length"],
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
              f"last10_avg={last_avg:.4f} decreased={decreased} finite={finite}", flush=True)
        del model, optimizer, losses
        release_mps_memory()

    viable = [r for r in results if r["decreased"] and r["finite"]]
    assert viable, "no candidate LR produced a decreasing, finite loss; sweep range is wrong"
    best = min(viable, key=lambda r: r["last_avg"])
    return best["peak_lr"], results


def measure_batch_candidate(model_config, data, device, batch_size, base_lr, base_lr_batch_size):
    """Measure throughput and fixed-token-budget loss for one batch size
    candidate. base_lr (swept at base_lr_batch_size) is linearly scaled to
    this candidate's batch size (standard linear scaling rule) as a fair
    starting point for the loss comparison, not a final LR choice, just
    enough to rank candidates consistently against each other. The real
    LR gets properly re-swept at whichever batch size wins, via
    sweep_peak_lr, not read off this scaled estimate."""
    scaled_lr = base_lr * (batch_size / base_lr_batch_size)
    context_length = model_config["context_length"]

    print(f"  [starting] batch_size={batch_size} throughput probe "
          f"({BATCH_THROUGHPUT_STEPS} steps)...", flush=True)
    torch.manual_seed(0)
    model = DecoderLM(copy.deepcopy(model_config))
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.1)
    model.to(device)
    model.train()
    t0 = time.time()
    for step in range(BATCH_THROUGHPUT_STEPS):
        torch.manual_seed(step)
        x, y = get_batch(data, batch_size, context_length, step)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
    elapsed = time.time() - t0
    tokens_per_sec = (BATCH_THROUGHPUT_STEPS * batch_size * context_length) / elapsed
    del model, optimizer, x, y, logits, loss
    release_mps_memory()

    num_steps = max(10, BATCH_TOKEN_BUDGET // (batch_size * context_length))
    print(f"  [starting] batch_size={batch_size} loss comparison "
          f"({num_steps} steps, {num_steps * batch_size * context_length:,} tokens)...", flush=True)
    torch.manual_seed(0)
    model2 = DecoderLM(copy.deepcopy(model_config))
    optimizer2 = torch.optim.AdamW(model2.parameters(), betas=(0.9, 0.95), weight_decay=0.1)
    losses = train_steps(
        model2, data, optimizer2, num_steps, batch_size, context_length,
        warmup_steps=max(1, num_steps // 10), peak_lr=scaled_lr, min_lr=scaled_lr / 10,
        grad_clip_norm=GRAD_CLIP, device=device,
    )
    window = min(10, len(losses))
    final_loss = sum(losses[-window:]) / window
    del model2, optimizer2, losses
    release_mps_memory()

    return {
        "batch_size": batch_size, "scaled_lr": scaled_lr, "tokens_per_sec": round(tokens_per_sec, 1),
        "num_steps": num_steps, "tokens_processed": num_steps * batch_size * context_length,
        "final_loss": final_loss,
    }


def sweep_batch_size(model_config, data, device, base_lr):
    """Pick a batch size without needing full training runs per candidate.
    For each candidate: measure throughput (BATCH_THROUGHPUT_STEPS) and a
    fixed-token-budget (BATCH_TOKEN_BUDGET) final loss, using an LR
    linearly scaled from base_lr for a fair per-candidate comparison.
    Picks the fastest candidate whose final loss doesn't exceed the
    BASELINE_BATCH_SIZE candidate's loss by more than BATCH_QUALITY_TOLERANCE:
    if a bigger batch trains just as well per token and faster, it's
    strictly better; if it trains meaningfully worse per token, the extra
    speed isn't worth it."""
    results = []
    for candidate in BATCH_CANDIDATES:
        r = measure_batch_candidate(model_config, data, device, candidate, base_lr, SWEEP_BATCH_SIZE)
        results.append(r)
        rss = peak_rss_gb()
        print(f"  batch_size={candidate}: {r['tokens_per_sec']:,.0f} tok/s, "
              f"final_loss={r['final_loss']:.4f} over {r['num_steps']} steps "
              f"({r['tokens_processed']:,} tokens, scaled_lr={r['scaled_lr']:.2e}), "
              f"peak_rss={rss:.1f}GB", flush=True)
        if rss >= MAX_PEAK_RSS_GB:
            print(f"  peak RSS {rss:.1f}GB reached the {MAX_PEAK_RSS_GB}GB safety ceiling, "
                  f"stopping before trying a larger batch size", flush=True)
            break

    baseline = next((r for r in results if r["batch_size"] == BASELINE_BATCH_SIZE), results[0])
    max_acceptable_loss = baseline["final_loss"] * (1 + BATCH_QUALITY_TOLERANCE)
    acceptable = [r for r in results if r["final_loss"] <= max_acceptable_loss]
    winner = max(acceptable, key=lambda r: r["tokens_per_sec"])
    print(f"  baseline batch_size={baseline['batch_size']} final_loss={baseline['final_loss']:.4f}, "
          f"max acceptable loss at tolerance {BATCH_QUALITY_TOLERANCE:.0%}: {max_acceptable_loss:.4f}")
    return winner, results


def demo():
    device = pick_device()
    print(f"device: {device}")

    full_config = load_model_config()
    model_config = full_config["model"]

    train_data = tokenize_corpus(TRAIN_TEXT_PATH, TOKENIZER_PATH, max_chars=TRAIN_SAMPLE_CHARS)
    val_data = tokenize_corpus(VAL_TEXT_PATH, TOKENIZER_PATH, max_chars=VAL_SAMPLE_CHARS)
    print(f"train sample: {len(train_data):,} tokens, val sample: {len(val_data):,} tokens")

    print(f"peak_lr sweep (batch_size={SWEEP_BATCH_SIZE}, anchor):")
    anchor_peak_lr, anchor_sweep_results = sweep_peak_lr(model_config, train_data, device, batch_size=SWEEP_BATCH_SIZE)
    print(f"anchor peak_lr={anchor_peak_lr:.0e}")

    print("batch_size sweep:")
    batch_winner, batch_results = sweep_batch_size(model_config, train_data, device, base_lr=anchor_peak_lr)
    best_batch_size = batch_winner["batch_size"]
    print(f"picked batch_size={best_batch_size} "
          f"({batch_winner['tokens_per_sec']:,.0f} tok/s, final_loss={batch_winner['final_loss']:.4f})")

    if best_batch_size == SWEEP_BATCH_SIZE:
        best_peak_lr, sweep_results = anchor_peak_lr, anchor_sweep_results
    else:
        print(f"peak_lr re-sweep (batch_size={best_batch_size}, winning batch size):")
        best_peak_lr, sweep_results = sweep_peak_lr(model_config, train_data, device, batch_size=best_batch_size)
    best_min_lr = best_peak_lr / 10
    print(f"picked peak_lr={best_peak_lr:.0e}, min_lr={best_min_lr:.0e} at batch_size={best_batch_size}")

    torch.manual_seed(0)
    model = DecoderLM(copy.deepcopy(model_config))
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.1)

    print(f"running {SMOKE_STEPS}-step smoke run on {device} at batch_size={best_batch_size}...")
    start = time.time()
    train_losses = []
    val_losses = {}
    model.to(device)
    for step in range(SMOKE_STEPS):
        lr = lr_at_step(step, SMOKE_WARMUP_STEPS, best_peak_lr, best_min_lr, SMOKE_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = lr

        torch.manual_seed(step)
        x, y = get_batch(train_data, best_batch_size, model_config["context_length"], step)
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
                vx, vy = get_batch(val_data, best_batch_size, model_config["context_length"], step)
                vx, vy = vx.to(device), vy.to(device)
                vlogits = model(vx)
                vloss = torch.nn.functional.cross_entropy(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
            val_losses[step] = vloss.item()
            print(f"  step {step + 1}/{SMOKE_STEPS}: train_loss={loss.item():.4f} val_loss={vloss.item():.4f} lr={lr:.2e}")

    elapsed = time.time() - start
    tokens_processed = SMOKE_STEPS * best_batch_size * model_config["context_length"]
    tokens_per_sec = tokens_processed / elapsed
    print(f"elapsed: {elapsed:.1f}s, throughput: {tokens_per_sec:,.0f} tokens/sec on {device}")

    expected_l0 = torch.log(torch.tensor(float(model_config["vocab_size"]))).item()
    assert abs(train_losses[0] - expected_l0) < 1.0, \
        f"first real-data loss {train_losses[0]:.4f} far from L0={expected_l0:.4f}; pipeline is likely broken"

    first_avg = sum(train_losses[:10]) / 10
    last_avg = sum(train_losses[-10:]) / 10
    assert last_avg < first_avg, "smoke run's loss did not decrease overall"
    assert all(l == l and abs(l) != float("inf") for l in train_losses), \
        "NaN or Inf appeared in the training loss; training diverged"

    final_val = val_losses[max(val_losses)]
    assert abs(last_avg - final_val) < 3.0, \
        f"train/val loss gap too large (train={last_avg:.4f}, val={final_val:.4f}); possible overfitting even at this small scale"

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
        x, _ = get_batch(train_data, best_batch_size, model_config["context_length"], 0)
        x = x.to(device)
        with torch.no_grad():
            out_original = model(x)
            out_reloaded = reloaded_model(x)
        assert torch.allclose(out_original, out_reloaded, atol=1e-5), \
            "reloaded checkpoint produces different output than the in-memory model on real data"
    print("checkpoint reload on real data: OK")

    full_config["batch_size"] = best_batch_size
    full_config["batch_size_sweep"] = {
        "candidates": BATCH_CANDIDATES,
        "results": [
            {"batch_size": r["batch_size"], "tokens_per_sec": r["tokens_per_sec"],
             "final_loss": round(r["final_loss"], 4), "num_steps": r["num_steps"]}
            for r in batch_results
        ],
        "baseline_batch_size": BASELINE_BATCH_SIZE,
        "quality_tolerance": BATCH_QUALITY_TOLERANCE,
        "picked": best_batch_size,
        "notes": (
            f"picked via smoke_test.py's batch-size sweep on real data ({device}); "
            f"fastest candidate among {BATCH_CANDIDATES} whose fixed-{BATCH_TOKEN_BUDGET:,}-token "
            f"loss did not exceed batch_size={BASELINE_BATCH_SIZE}'s loss by more than "
            f"{BATCH_QUALITY_TOLERANCE:.0%}. See viva-lma.md for full reasoning."
        ),
    }
    full_config["schedule"]["peak_lr"] = round(best_peak_lr, 8)
    full_config["schedule"]["min_lr"] = round(best_min_lr, 8)
    full_config["schedule"]["notes"] = (
        f"picked via smoke_test.py's LR sweep on real data ({device}) at batch_size={best_batch_size}; "
        f"candidates {SWEEP_CANDIDATES} over {SWEEP_STEPS} steps each, "
        f"min_lr = peak_lr / 10 (standard nanoGPT convention). "
        f"See viva-lma.md for the sweep results."
    )
    full_config["smoke_test_throughput"] = {
        "device": device, "tokens_per_sec": round(tokens_per_sec, 1),
        "batch_size": best_batch_size, "context_length": model_config["context_length"],
    }
    save_model_config(full_config)
    print(f"model_config.json updated with batch_size={best_batch_size}, "
          f"peak_lr={best_peak_lr:.0e}, min_lr={best_min_lr:.0e}")

    print("assamese local MPS smoke test: OK")


if __name__ == "__main__":
    demo()
