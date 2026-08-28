"""Training loop for the Hindi decoder-only LM: AdamW, warmup-then-cosine
LR schedule, gradient clipping, and a full-state checkpoint (weights,
optimizer state, training step, config) so a Colab session dying mid-run
never loses progress.

Every training step reseeds torch's global RNG to a value keyed only on
the step index, before the forward pass. This makes dropout masks (and
any other RNG-consuming op) a deterministic function of step number, so a
resumed run reproduces the exact same trajectory as an uninterrupted one
-- the property the checkpoint-resume self-check in test_train.py proves.
"""

import json
import math
import os
from pathlib import Path

import sentencepiece as spm
import torch


def load_config(path):
    """Load a model_config.json file (model + optimizer + schedule + tokenizer sections)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tokenize_corpus(text_path, tokenizer_path, max_chars=None):
    """Tokenize a text file into a flat 1D tensor of token ids. Reads only the
    first max_chars characters when given -- these corpus files run into the
    gigabytes, so reading the whole thing just to slice it is wasteful (and
    can be slow enough to look like a hang)."""
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    with open(text_path, encoding="utf-8") as f:
        text = f.read(max_chars) if max_chars else f.read()
    ids = sp.encode(text, out_type=int)
    return torch.tensor(ids, dtype=torch.long)


def get_batch(data, batch_size, context_length, step):
    """Deterministic, non-random contiguous batch selection keyed on step --
    lets a resumed run reproduce the exact same batches as an uninterrupted
    run, without needing to snapshot a data-loader's RNG state."""
    span = batch_size * context_length
    max_start = len(data) - context_length - 1
    start = (step * span) % max_start
    xs, ys = [], []
    for b in range(batch_size):
        s = (start + b * context_length) % max_start
        xs.append(data[s : s + context_length])
        ys.append(data[s + 1 : s + context_length + 1])
    return torch.stack(xs), torch.stack(ys)


def lr_at_step(step, warmup_steps, peak_lr, min_lr, total_steps):
    """Linear warmup to peak_lr over warmup_steps, then cosine decay to min_lr."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (peak_lr - min_lr)


def save_checkpoint(path, model, optimizer, step, config):
    """Save weights, optimizer state, training step, and config -- the four
    things the project's checkpoint-resume requirement mandates.

    Writes to a temp file first, then renames onto the real path. path holds
    the only copy of the checkpoint (each call overwrites it), so a session
    dying mid torch.save would otherwise corrupt the one checkpoint that
    exists, losing all prior progress rather than just the latest interval."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def load_checkpoint(path, model, optimizer, map_location="cpu"):
    """Restore weights, optimizer state, and step from a checkpoint. Returns the resumed step."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"]


def train(
    model,
    data,
    optimizer,
    num_steps,
    batch_size,
    context_length,
    warmup_steps,
    peak_lr,
    min_lr,
    grad_clip_norm,
    start_step=0,
    total_steps=None,
    checkpoint_path=None,
    checkpoint_every=None,
    config=None,
    device="cpu",
):
    """Run training from start_step up to (not including) num_steps. Returns the
    per-step loss history.

    total_steps is the LR schedule's cosine-decay horizon, independent of
    num_steps (this call's own stopping point) -- required so that a resumed
    run's later, shorter train() call still decays against the *original*
    training horizon instead of treating its own early stop as the end of
    the schedule. Defaults to num_steps for a plain, non-resumed run."""
    if total_steps is None:
        total_steps = num_steps

    model.to(device)
    model.train()
    losses = []

    for step in range(start_step, num_steps):
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

        if checkpoint_path and checkpoint_every and (step + 1) % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, model, optimizer, step + 1, config)

    return losses
