"""Reasoning finetuning for the Assamese decoder-only LM.

Starts from the Phase 2 pretrained checkpoint, keeps the tokenizer and
vocabulary fixed, LoRA finetune (not full-parameter -- see
docs-phase-3/finetune_method_choice.md's 2026-09-13 update for why: the
regularization argument against overfitting a tiny 6000-example set, not
the usual compute-saving argument, which doesn't apply at 24.3M params).

LoRA targets W_q and W_v only, per the original LoRA paper's own ablation
(Hu et al. 2021, Table 5/6): adapting {W_q, W_v} matches the quality of
adapting all four attention matrices while needing less capacity, and a
small rank (r=1-4 in their experiments) already suffices for that pair.
model.py is never modified -- LoRALinear wraps the existing q_proj/v_proj
nn.Linear submodules on an already-loaded pretrained model instance.

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

Reuses save_checkpoint/load_config from train.py unchanged. LR schedule is
its own (not train.py's lr_at_step): linear warmup then held constant at
peak_lr, not cosine decay -- see docs-phase-3/forgetting_threshold_review.md's
2026-09-14 addendum. A cosine horizon requires knowing total_steps upfront,
but this run's real length is decided by early stopping, not a fixed epoch
count; committing to a horizon guess either under- or over-decays depending
on when the run actually stops. Warmup-then-constant is the standard choice
for exactly this case (short LoRA finetune of a priori unknown length).
Per-epoch checkpoints during training stay in a LoRA-native
state dict (base weights + lora_A/lora_B), resumable by rebuilding the
same LoRA-injected model before loading. The final deliverable checkpoint
is produced by merge_lora_to_plain: LoRA's delta is folded back into
q_proj/v_proj's weights, yielding a plain DecoderLM state dict -- the
same shape as a pretraining checkpoint (spec's literal "same
resume-capable format as pretraining" requirement) and, critically, the
exact shape Phase 2's attention-analysis toolkit already expects, so that
tool is reused completely unchanged on the finetuned checkpoint.
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from model import DecoderLM  # noqa: E402
from train import load_config, save_checkpoint  # noqa: E402

BOS_ID = 1  # <s>, reused here as the prompt/answer boundary marker
EOS_ID = 2  # </s>, terminal end of sequence
PAD_ID = 0
MAX_SEQ_LEN = 128  # comfortably above the generator's measured worst case (~67 tokens,
# see data/reasoning/test_generate_reasoning_data.py's token-budget self-check)

# Fixed seed for every pretrain_val_ppl call, baseline and per-epoch alike --
# NOT the training run's own seed. Using the training seed here (the
# original bug, fixed 2026-09-14) meant seed-1/seed-2 runs compared a
# threshold computed on seed-0's window sample against PPL measured on a
# different window sample every epoch: an invalid comparison that tripped a
# real stop at a 0.06% margin, an order of magnitude inside known sampling
# noise (~0.6%, see pretrain_val_ppl's own docstring). See
# docs-phase-3/forgetting_threshold_review.md.
PRETRAIN_EVAL_SEED = 0


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank delta:
    output = base(x) + (alpha/rank) * x @ lora_A^T @ lora_B^T.

    lora_B starts at zero (lora_A does not), so the wrapped layer computes
    exactly the frozen base layer's output before any training happens --
    the same zero-init convention as the original LoRA paper, so injecting
    this wrapper into an already-loaded pretrained model changes nothing
    about its behavior until training actually updates lora_A/lora_B."""

    def __init__(self, base_linear, rank, alpha):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out

    def merged_weight(self):
        """base.weight + the folded-in LoRA delta, same shape as the base
        weight -- used to export a plain nn.Linear-compatible weight."""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


def inject_lora(model, rank, alpha):
    """Replaces q_proj and v_proj in every block with a LoRALinear wrapper,
    in place. Call this AFTER loading pretrained weights into `model` --
    the wrapper freezes and wraps whatever nn.Linear is there at call time."""
    for block in model.blocks:
        block.attn.q_proj = LoRALinear(block.attn.q_proj, rank, alpha)
        block.attn.v_proj = LoRALinear(block.attn.v_proj, rank, alpha)
    return model


def freeze_non_lora_params(model):
    """Sets requires_grad on every parameter: True for lora_A/lora_B and
    token_embedding.weight, False for everything else (including
    q_proj/v_proj's own frozen base weights, already set False by
    LoRALinear.__init__, and every other parameter: k_proj, out_proj, FFN,
    LayerNorm).

    token_embedding.weight is trainable but gradient-masked to BOS_ID/EOS_ID
    rows only (see docs-phase-3/lora_eos_untrained_finding.md): <s>/</s> are
    never emitted mid-corpus during pretraining, so their rows are still
    untouched random init under the plain lora_A/lora_B-only freeze, and the
    model never learns to terminate generation. requires_grad has no
    per-row granularity, so the row restriction is enforced by a gradient
    hook instead -- every other row's gradient is zeroed before it ever
    reaches the optimizer. With tie_embeddings=True, token_embedding.weight
    and head.weight are the same Parameter object (model.py's
    `self.head.weight = self.token_embedding.weight`), so named_parameters
    surfaces it once and this hook covers both the input-embedding and
    tied-output-head gradient contributions.

    Callers must build the optimizer via build_optimizer(), not a plain
    single-group AdamW(...): weight decay is applied to the parameter value
    directly, independent of the gradient mask above, so an unsplit
    optimizer would still shrink every untrained embedding row every step."""
    for name, p in model.named_parameters():
        p.requires_grad = "lora_A" in name or "lora_B" in name or name == "token_embedding.weight"

    def _mask_bos_eos_rows(grad):
        mask = torch.zeros_like(grad)
        mask[[BOS_ID, EOS_ID]] = 1.0
        return grad * mask

    model.token_embedding.weight.register_hook(_mask_bos_eos_rows)


def build_optimizer(model, weight_decay=0.0, betas=(0.9, 0.95), lr=1e-3):
    """AdamW over model's currently-trainable parameters, split into two
    groups so weight decay never reaches token_embedding.weight: decay
    shrinks a parameter's value every step regardless of its gradient, so
    even a correctly gradient-masked embedding row would still get decayed
    if it shared a group with lora_A/lora_B. token_embedding.weight (if
    trainable) gets weight_decay=0.0; everything else gets `weight_decay`.
    lr is a placeholder here -- finetune()'s own warmup schedule overwrites
    every group's lr before the first optimizer step."""
    other_params, embedding_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (embedding_params if name == "token_embedding.weight" else other_params).append(p)

    groups = [{"params": other_params, "weight_decay": weight_decay}]
    if embedding_params:
        groups.append({"params": embedding_params, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, betas=betas, lr=lr)


def merge_lora_to_plain(lora_model, config):
    """Builds a fresh, plain DecoderLM (no LoRA wrappers) with q_proj/v_proj
    set to the merged (base + LoRA delta) weight, everything else copied
    unchanged. The result is indistinguishable in structure from a Phase 2
    pretrained checkpoint's model -- safe to hand to unmodified Phase 2
    eval/attention-analysis code."""
    plain_model = DecoderLM(config)
    plain_model.token_embedding.load_state_dict(lora_model.token_embedding.state_dict())
    plain_model.ln_f.load_state_dict(lora_model.ln_f.state_dict())
    if not config["tie_embeddings"]:
        plain_model.head.load_state_dict(lora_model.head.state_dict())

    for plain_block, lora_block in zip(plain_model.blocks, lora_model.blocks):
        plain_block.ln1.load_state_dict(lora_block.ln1.state_dict())
        plain_block.ln2.load_state_dict(lora_block.ln2.state_dict())
        plain_block.ff.load_state_dict(lora_block.ff.state_dict())
        plain_block.attn.k_proj.load_state_dict(lora_block.attn.k_proj.state_dict())
        plain_block.attn.out_proj.load_state_dict(lora_block.attn.out_proj.state_dict())
        with torch.no_grad():
            plain_block.attn.q_proj.weight.copy_(lora_block.attn.q_proj.merged_weight())
            plain_block.attn.q_proj.bias.copy_(lora_block.attn.q_proj.base.bias)
            plain_block.attn.v_proj.weight.copy_(lora_block.attn.v_proj.merged_weight())
            plain_block.attn.v_proj.bias.copy_(lora_block.attn.v_proj.base.bias)

    return plain_model


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


def pretrain_val_ppl(model, val_data, context_length, device, num_batches=20, batch_size=16, seed=0):
    """Perplexity on a random sample of the ORIGINAL Phase 2 pretrain
    validation split (plain next-token prediction, no masking) -- the
    forgetting signal from docs-phase-3/sample_size_and_stopping_criterion.md,
    tracked alongside (never conflated with) the reasoning-task loss above.

    Samples num_batches * batch_size independent random windows spread
    across the FULL val_data range (train.py's get_batch is deliberately
    not reused here: its step-keyed formula only ever advances through a
    handful of contiguous windows near a fixed offset, which underlies
    real training's need for deterministic resumability, but is not a
    representative sample of a large val set for a one-off eval like this
    -- caught empirically on 2026-09-14 when a baseline PPL check on real
    data came back inflated, traced to the sample being confined to a
    narrow slice of the full val set)."""
    model.eval()
    max_start = len(val_data) - context_length - 1
    rng = random.Random(seed)
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for _ in range(num_batches):
            starts = [rng.randint(0, max_start) for _ in range(batch_size)]
            x = torch.stack([val_data[s : s + context_length] for s in starts]).to(device)
            y = torch.stack([val_data[s + 1 : s + context_length + 1] for s in starts]).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * y.numel()
            total_count += y.numel()
    model.train()
    avg_loss = total_loss / total_count
    return avg_loss, math.exp(avg_loss)


def _snapshot_trainable(model):
    """Deep-copies only the trainable (LoRA) parameters -- cheap, since the
    frozen base weights never change during training and don't need
    snapshotting to reconstruct the model's state at any given epoch."""
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def _restore_trainable(model, snapshot):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in snapshot:
                p.copy_(snapshot[name])


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
    warmup_steps,
    grad_clip_norm,
    seed,
    checkpoint_path=None,
    config=None,
    device="cpu",
    max_seq_len=MAX_SEQ_LEN,
    early_stopping_patience=None,
    ppl_forgetting_threshold=None,
    on_epoch_end=None,
):
    """Runs up to num_epochs full passes over train_examples. Logs both the
    reasoning-task validation loss and the pretrain-val PPL after every
    epoch (the forgetting signal from
    docs-phase-3/sample_size_and_stopping_criterion.md).

    on_epoch_end: optional callable invoked once per completed epoch, after
    that epoch's training, eval, and logging are already done -- receives
    the epoch's own log dict (so a caller can do live per-epoch logging,
    e.g. a CSV row) and its truthy return breaks the loop there. The
    current epoch is never cut short mid-batch. In early-stopping mode a
    requested stop reuses the existing best-snapshot restore-and-save tail
    below, same as a natural early stop; without early stopping, the
    epoch's checkpoint is already saved before the callback runs, so the
    break is a clean exit either way.

    Without early_stopping_patience (default): runs the full num_epochs,
    checkpointing every epoch to checkpoint_path (each write overwrites the
    last, so only the final epoch's weights persist) -- the original
    behavior, kept for the self-test and any caller that wants the log
    inspected by hand afterward.

    With early_stopping_patience=N: tracks the best (lowest)
    reasoning_val_loss seen so far, keeping only its trainable-parameter
    snapshot in memory (cheap -- LoRA params only, not the frozen base).
    Stops after N epochs with no improvement, or immediately if
    ppl_forgetting_threshold is set and pretrain_val_ppl ever exceeds it
    (a real forgetting event overrides reasoning-loss patience). model's
    weights are restored to the best epoch before returning, and exactly
    one checkpoint write happens (the best epoch), not one per epoch --
    avoids both saving an overfit final epoch and a Drive-storage blowup
    from one full-size checkpoint file per epoch.

    Returns a list of per-epoch dicts: {epoch, train_loss, reasoning_val_loss,
    pretrain_val_loss, pretrain_val_ppl, elapsed_s}, covering only the epochs
    actually run (shorter than num_epochs if early stopping triggered).
    """
    model.to(device)
    model.train()
    rng = random.Random(seed)
    t0 = time.time()

    log = []
    step = 0
    best_loss = float("inf")
    best_epoch = None
    best_snapshot = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        epoch_losses = []
        for x, y, mask in make_batches(train_examples, sp, batch_size, max_seq_len, rng):
            lr = peak_lr * (step + 1) / warmup_steps if step < warmup_steps else peak_lr
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
            model, pretrain_val_data, context_length, device, seed=PRETRAIN_EVAL_SEED
        )

        log.append({
            "epoch": epoch,
            "train_loss": sum(epoch_losses) / len(epoch_losses),
            "reasoning_val_loss": reasoning_val_loss,
            "pretrain_val_loss": pretrain_val_loss,
            "pretrain_val_ppl": pretrain_val_ppl_value,
            "elapsed_s": time.time() - t0,
        })

        stop_requested = on_epoch_end is not None and bool(on_epoch_end(log[-1]))

        if early_stopping_patience is None:
            if checkpoint_path:
                save_checkpoint(checkpoint_path, model, optimizer, step, config)
            if stop_requested:
                break
            continue

        if reasoning_val_loss < best_loss:
            best_loss = reasoning_val_loss
            best_epoch = epoch
            best_snapshot = _snapshot_trainable(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        forgetting_triggered = (
            ppl_forgetting_threshold is not None and pretrain_val_ppl_value > ppl_forgetting_threshold
        )
        patience_exhausted = epochs_without_improvement >= early_stopping_patience

        if forgetting_triggered or patience_exhausted or stop_requested:
            break

    if early_stopping_patience is not None and best_snapshot is not None:
        _restore_trainable(model, best_snapshot)
        if checkpoint_path:
            save_checkpoint(checkpoint_path, model, optimizer, step, config)

    return log


def _evaluate_reasoning_loss(model, sp, examples, batch_size, max_seq_len, device):
    model.eval()
    losses = []
    rng = random.Random(0)  # fixed order for eval, not shuffled meaningfully
    with torch.no_grad():
        for x, y, mask in make_batches(examples, sp, batch_size, max_seq_len, rng):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x)
            losses.append(masked_lm_loss(logits, y, mask).item())
    model.train()
    return sum(losses) / len(losses)
