#!/usr/bin/env python3
"""Self-check for attention_analysis.py's entropy/distance math and the
return_attention plumbing, independent of any trained checkpoint.

Run directly: python3 test_attention.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model import DecoderLM  # noqa: E402

from attention_analysis import (  # noqa: E402
    compute_attention,
    entropy_per_head,
    load_model_config,
    mean_attention_distance_per_head,
)


def test_entropy_uniform_vs_peaked():
    T = 4
    uniform_head = torch.zeros(T, T)
    for t in range(T):
        uniform_head[t, : t + 1] = 1.0 / (t + 1)
    peaked_head = torch.zeros(T, T)
    for t in range(T):
        peaked_head[t, t] = 1.0  # all mass on self, one-hot

    attn_layer = torch.stack([uniform_head, peaked_head])  # (n_head=2, T, T)
    entropies = entropy_per_head(attn_layer)
    assert entropies[0] > entropies[1], \
        f"uniform head should have higher entropy than peaked head, got {entropies}"
    assert entropies[1] < 1e-6, f"one-hot attention should have ~0 entropy, got {entropies[1]}"
    print(f"entropy_per_head check: OK (uniform={entropies[0]:.4f}, peaked={entropies[1]:.4f})")


def test_mean_attention_distance_known_case():
    T = 3
    always_self = torch.zeros(1, T, T)
    for t in range(T):
        always_self[0, t, t] = 1.0
    always_first = torch.zeros(1, T, T)
    for t in range(T):
        always_first[0, t, 0] = 1.0

    dist_self = mean_attention_distance_per_head(always_self)[0]
    dist_first = mean_attention_distance_per_head(always_first)[0]
    assert math.isclose(dist_self, 0.0, abs_tol=1e-6), f"expected 0 distance for self-attention, got {dist_self}"
    # query t=1 -> distance 1, query t=2 -> distance 2, mean = 1.5
    assert math.isclose(dist_first, 1.5, abs_tol=1e-6), f"expected mean distance 1.5, got {dist_first}"
    print(f"mean_attention_distance_per_head check: OK (self={dist_self:.4f}, first-token={dist_first:.4f})")


def test_compute_attention_shapes():
    model_config = load_model_config()["model"]
    model = DecoderLM(model_config)
    token_ids = torch.randint(0, model_config["vocab_size"], (20,))
    attentions = compute_attention(model, token_ids, device="cpu")
    assert len(attentions) == model_config["n_layer"]
    assert attentions[0].shape == (model_config["n_head"], 20, 20)
    assert torch.allclose(attentions[0].sum(-1), torch.ones(model_config["n_head"], 20), atol=1e-5)
    print("compute_attention shape/normalization check: OK")


def demo():
    test_entropy_uniform_vs_peaked()
    test_mean_attention_distance_known_case()
    test_compute_attention_shapes()
    print("assamese attention_analysis self-check: OK")


if __name__ == "__main__":
    demo()
