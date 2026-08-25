#!/usr/bin/env python3
"""Architecture self-check: causal-mask leak test + parameter count.

Run directly: python3 test_model.py
"""

import json
from pathlib import Path

import torch

from model import DecoderLM

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "model_config.json"


def load_model_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_causal_mask_no_leak(model):
    """Changing token t+1 must not change the logits at position t (spec's required test)."""
    torch.manual_seed(0)
    model.eval()

    T = 16
    vocab_size = model.config["vocab_size"]
    ids = torch.randint(0, vocab_size, (1, T))

    changed = ids.clone()
    t = T // 2
    changed[0, t] = (changed[0, t] + 1) % vocab_size

    with torch.no_grad():
        logits_a = model(ids)
        logits_b = model(changed)

    unaffected = logits_a[:, :t, :]
    unaffected_changed = logits_b[:, :t, :]
    assert torch.allclose(unaffected, unaffected_changed, atol=1e-5), \
        "changing a future token altered logits at an earlier position -- causal mask is leaking"

    assert not torch.allclose(logits_a[:, t:, :], logits_b[:, t:, :], atol=1e-5), \
        "changing a token had no effect anywhere -- forward pass looks disconnected from input"

    print("causal mask leak test: OK")


def test_param_count(model):
    """Total trainable params should land near the ~25M target (24.4M tied, per depth_width_tradeoff.md)."""
    n_params = model.num_parameters()
    print(f"total trainable parameters: {n_params:,}")
    assert 20_000_000 <= n_params <= 28_000_000, \
        f"parameter count {n_params:,} is outside the expected ~25M band"


def test_l0_sanity_check(model):
    """Untrained model's average cross-entropy loss should be close to ln(vocab_size)."""
    torch.manual_seed(0)
    model.eval()

    vocab_size = model.config["vocab_size"]
    B, T = 4, 32
    ids = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))

    with torch.no_grad():
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size), targets.view(-1)
        )

    expected = torch.log(torch.tensor(float(vocab_size)))
    print(f"untrained loss: {loss.item():.4f} nats (expected ~{expected.item():.4f})")
    assert abs(loss.item() - expected.item()) < 0.5, \
        "untrained loss is far from ln(vocab_size) -- setup is likely broken"


def demo():
    config = load_model_config()
    model = DecoderLM(config["model"])

    test_param_count(model)
    test_causal_mask_no_leak(model)
    test_l0_sanity_check(model)

    print("hindi model architecture self-check: OK")


if __name__ == "__main__":
    demo()
