"""Reasoning finetuning for the Hindi decoder-only LM.

Starts from the Phase 2 pretrained checkpoint, keeps the tokenizer and
vocabulary fixed, full-parameter finetune (not PEFT -- see
docs-phase-3/finetune_method_choice.md for why), on the synthetic
reasoning dataset in data/reasoning/.

Sequence format and loss masking, per docs-phase-3/finetune_protocol.md:
    <question tokens> <s> <answer tokens> </s>
Loss is masked to answer positions only: predicting the first answer
token through the terminal </s>. Predicting the transition into <s>
itself is excluded from the loss, since <s> is always inserted
deterministically right after the prompt at both train and inference
time -- it is a structural delimiter we control at data-prep time, not
something the model needs to learn to emit on its own.

Right-padding needs no attention-mask change to model.py: causal masking
already prevents any real token from attending to a later, padded
position, and padded target positions are excluded from the loss, so
their (meaningless) output is simply never used.

Reuses save_checkpoint/load_config/lr_at_step/get_batch from train.py
unchanged -- same resume-capable checkpoint format as pretraining, per
spec's Phase 3 protocol requirement.
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import get_batch, load_config, lr_at_step, save_checkpoint  # noqa: E402

BOS_ID = 1  # <s>, reused here as the prompt/answer boundary marker
EOS_ID = 2  # </s>, terminal end of sequence
PAD_ID = 0
MAX_SEQ_LEN = 128  # comfortably above the generator's measured worst case (~67 tokens,
# see data/reasoning/test_generate_reasoning_data.py's token-budget self-check)


def load_pretrained_weights(path, model, map_location="cpu"):
    """Load only the model weights from a Phase 2 pretrained checkpoint --
    deliberately not its optimizer state or step counter. Finetuning starts a
    fresh optimizer (no stale momentum from the pretraining trajectory) and
    its own step counter, tracked separately from the pretrain run's."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("step")  # returned for logging only, never resumed from


def load_reasoning_examples(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode_example(sp, prompt, answer, max_seq_len=MAX_SEQ_LEN):
    """Tokenize one (prompt, answer) pair into a fixed-length, right-padded
    (input_ids, target_ids, loss_mask) triple, each of length max_seq_len - 1.

    loss_mask[i] = 1 iff target position i is an answer token or the
    terminal </s> (i.e. i falls at or after the <s> boundary's index in the
    unshifted sequence); 0 for prompt positions, the <s> boundary's own
    prediction target, and padding."""
    prompt_ids = sp.encode(prompt, out_type=int)
    answer_ids = sp.encode(answer, out_type=int)
    full = prompt_ids + [BOS_ID] + answer_ids + [EOS_ID]
    boundary_idx = len(prompt_ids)  # index of <s> within `full`

    if len(full) > max_seq_len:
        raise ValueError(f"example too long ({len(full)} tokens) for max_seq_len={max_seq_len}")

    padded = full + [PAD_ID] * (max_seq_len - len(full))

    x = torch.tensor(padded[:-1], dtype=torch.long)
    y = torch.tensor(padded[1:], dtype=torch.long)

    mask = torch.zeros(max_seq_len - 1, dtype=torch.float)
    mask[boundary_idx : len(full) - 1] = 1.0
    return x, y, mask


def make_batches(examples, sp, batch_size, max_seq_len, rng):
    """One shuffled pass over examples, yielding (x, y, mask) batches."""
    order = list(range(len(examples)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        xs, ys, masks = [], [], []
        for i in idxs:
            x, y, mask = encode_example(sp, examples[i]["prompt"], examples[i]["answer"], max_seq_len)
            xs.append(x)
            ys.append(y)
            masks.append(mask)
        yield torch.stack(xs), torch.stack(ys), torch.stack(masks)


def masked_lm_loss(logits, targets, mask):
    """L_SFT = -(1/|A|) * sum_{t in A} log p(x_t | x_<t), per
    docs-phase-3/finetune_protocol.md. Padded/prompt positions are excluded
    via mask; averaged (not summed) over the answer-token count."""
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view(targets.shape)
    masked = per_token_loss * mask
    return masked.sum() / mask.sum().clamp(min=1.0)


def pretrain_val_ppl(model, val_data, context_length, device, num_batches=10, batch_size=8, seed=0):
    """Perplexity on a sample of the ORIGINAL Phase 2 pretrain validation
    split (plain next-token prediction, no masking) -- the forgetting
    signal from docs-phase-3/sample_size_and_stopping_criterion.md, tracked
    alongside (never conflated with) the reasoning-task loss above."""
    model.eval()
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for b in range(num_batches):
            x, y = get_batch(val_data, batch_size, context_length, step=seed * 10_000 + b)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * y.numel()
            total_count += y.numel()
    model.train()
    avg_loss = total_loss / total_count
    return avg_loss, math.exp(avg_loss)


def finetune(
    model,
    optimizer,
    sp,
    train_examples,
    reasoning_val_examples,
    pretrain_val_data,
    num_epochs,
    batch_size,
    context_length,
    peak_lr,
    min_lr,
    warmup_steps,
    grad_clip_norm,
    seed,
    checkpoint_path=None,
    config=None,
    device="cpu",
    max_seq_len=MAX_SEQ_LEN,
):
    """Runs num_epochs full passes over train_examples. Checkpoints and logs
    both the reasoning-task validation loss and the pretrain-val PPL after
    every epoch, so the actual stopping point (per the TA's samples-vs-PPL
    guidance in docs-phase-3/sample_size_and_stopping_criterion.md) can be
    picked by inspecting the log afterward, rather than an automatic
    in-loop early stop.

    Returns a list of per-epoch dicts: {epoch, train_loss, reasoning_val_loss,
    pretrain_val_loss, pretrain_val_ppl}.
    """
    model.to(device)
    model.train()
    rng = __import__("random").Random(seed)

    steps_per_epoch = math.ceil(len(train_examples) / batch_size)
    total_steps = steps_per_epoch * num_epochs

    log = []
    step = 0
    for epoch in range(num_epochs):
        epoch_losses = []
        for x, y, mask in make_batches(train_examples, sp, batch_size, max_seq_len, rng):
            lr = lr_at_step(step, warmup_steps, peak_lr, min_lr, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x)
            loss = masked_lm_loss(logits, y, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            epoch_losses.append(loss.item())
            step += 1

        reasoning_val_loss = _evaluate_reasoning_loss(
            model, sp, reasoning_val_examples, batch_size, max_seq_len, device
        )
        pretrain_val_loss, pretrain_val_ppl_value = pretrain_val_ppl(
            model, pretrain_val_data, context_length, device, seed=seed
        )

        log.append({
            "epoch": epoch,
            "train_loss": sum(epoch_losses) / len(epoch_losses),
            "reasoning_val_loss": reasoning_val_loss,
            "pretrain_val_loss": pretrain_val_loss,
            "pretrain_val_ppl": pretrain_val_ppl_value,
        })

        if checkpoint_path:
            save_checkpoint(checkpoint_path, model, optimizer, step, config)

    return log


def _evaluate_reasoning_loss(model, sp, examples, batch_size, max_seq_len, device):
    model.eval()
    losses = []
    rng = __import__("random").Random(0)  # fixed order for eval, not shuffled meaningfully
    with torch.no_grad():
        for x, y, mask in make_batches(examples, sp, batch_size, max_seq_len, rng):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x)
            losses.append(masked_lm_loss(logits, y, mask).item())
    model.train()
    return sum(losses) / len(losses)
