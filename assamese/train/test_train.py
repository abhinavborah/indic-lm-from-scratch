#!/usr/bin/env python3
"""Checkpoint-resume self-check: an interrupted run must reproduce the
exact same loss trajectory and final weights as an uninterrupted one.
Proves the project's mandatory checkpoint-resume requirement actually
works, not just that saving/loading doesn't crash.

Run directly: python3 test_train.py
"""

import copy
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model import DecoderLM  # noqa: E402

from train import get_batch, load_checkpoint, load_config, tokenize_corpus, train  # noqa: E402

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "model_config.json"
TOKENIZER_PATH = Path(__file__).parent.parent / "tokenizer" / "assamese_bpe_8000.model"
SAMPLE_TEXT_PATH = Path(__file__).parent.parent / "data" / "splits" / "train.txt"

NUM_STEPS = 20
CHECKPOINT_STEP = 10
BATCH_SIZE = 2
PEAK_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 5
GRAD_CLIP = 1.0


def build_test_model_config():
    """Real architecture from the actual config; dropout is not touched --
    determinism comes from per-step RNG reseeding in train.py, not from
    disabling dropout."""
    full_config = load_config(CONFIG_PATH)
    return copy.deepcopy(full_config["model"])


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.1)


def run_uninterrupted(data, model_config):
    torch.manual_seed(0)
    model = DecoderLM(model_config)
    optimizer = make_optimizer(model)
    losses = train(
        model, data, optimizer, NUM_STEPS, BATCH_SIZE, model_config["context_length"],
        WARMUP_STEPS, PEAK_LR, MIN_LR, GRAD_CLIP,
    )
    return model, losses


def run_interrupted(data, model_config, checkpoint_path):
    torch.manual_seed(0)
    model = DecoderLM(model_config)
    optimizer = make_optimizer(model)
    losses_first_half = train(
        model, data, optimizer, CHECKPOINT_STEP, BATCH_SIZE, model_config["context_length"],
        WARMUP_STEPS, PEAK_LR, MIN_LR, GRAD_CLIP, total_steps=NUM_STEPS,
        checkpoint_path=checkpoint_path, checkpoint_every=CHECKPOINT_STEP, config=model_config,
    )

    # Simulate a crash: brand-new model/optimizer, nothing carried over in memory.
    resumed_model = DecoderLM(model_config)
    resumed_optimizer = make_optimizer(resumed_model)
    resumed_step = load_checkpoint(checkpoint_path, resumed_model, resumed_optimizer)
    assert resumed_step == CHECKPOINT_STEP, \
        f"expected to resume at step {CHECKPOINT_STEP}, got {resumed_step}"

    losses_second_half = train(
        resumed_model, data, resumed_optimizer, NUM_STEPS, BATCH_SIZE, model_config["context_length"],
        WARMUP_STEPS, PEAK_LR, MIN_LR, GRAD_CLIP, start_step=resumed_step,
    )
    return resumed_model, losses_first_half + losses_second_half


def test_checkpoint_resume_matches_uninterrupted():
    # CPU matmul reduction order can be non-deterministic across threads,
    # which is invisible normally but breaks a bit-exact resume comparison
    # like this one -- pin both down for this test only (real training
    # doesn't need bit-exact determinism, just this proof does).
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    model_config = build_test_model_config()
    data = tokenize_corpus(SAMPLE_TEXT_PATH, TOKENIZER_PATH, max_chars=200_000)

    min_tokens_needed = BATCH_SIZE * model_config["context_length"] * NUM_STEPS
    assert len(data) > min_tokens_needed, \
        f"sample too small ({len(data)} tokens) for {NUM_STEPS} steps of size {min_tokens_needed}"

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = Path(tmp) / "test_checkpoint.pt"

        model_a, losses_a = run_uninterrupted(data, model_config)
        model_b, losses_b = run_interrupted(data, model_config, checkpoint_path)

        for i, (la, lb) in enumerate(zip(losses_a, losses_b)):
            assert abs(la - lb) < 1e-4, \
                f"loss diverged at step {i}: uninterrupted={la} resumed={lb} -- resume is not reproducing the uninterrupted run"

        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            assert torch.allclose(pa, pb, atol=1e-5), \
                "final weights differ between the uninterrupted and resumed runs"

    print(f"checkpoint-resume self-check: OK ({NUM_STEPS} steps, "
          f"interrupted at step {CHECKPOINT_STEP}, trajectories match exactly)")


def test_get_batch_is_deterministic_per_step():
    """Same step index must always yield the same batch, independent of call order."""
    data = torch.arange(10_000)
    a = get_batch(data, batch_size=4, context_length=32, step=7)
    b = get_batch(data, batch_size=4, context_length=32, step=7)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), \
        "get_batch is not deterministic for a fixed step -- resume correctness depends on this"
    print("get_batch determinism check: OK")


def demo():
    test_get_batch_is_deterministic_per_step()
    test_checkpoint_resume_matches_uninterrupted()
    print("assamese train loop + checkpoint-resume self-check: OK")


if __name__ == "__main__":
    demo()
