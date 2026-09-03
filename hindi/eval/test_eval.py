#!/usr/bin/env python3
"""Self-check for eval_lm.py's windowing and metric math, independent of
any trained checkpoint, uses a freshly-initialized real-architecture
model on synthetic token data, so this runs standalone without needing a
completed training run.

Run directly: python3 test_eval.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model import DecoderLM  # noqa: E402

from eval_lm import compute_metrics, evaluate_split, load_model_config  # noqa: E402


def test_compute_metrics_matches_known_values():
    ppl, bpb = compute_metrics(mean_loss_nats=1.0, num_tokens=100, byte_length=50)
    assert math.isclose(ppl, math.e, rel_tol=1e-9), f"expected ppl=e, got {ppl}"
    expected_bpb = 1.0 * math.log2(math.e) * 100 / 50
    assert math.isclose(bpb, expected_bpb, rel_tol=1e-9), f"expected bpb={expected_bpb}, got {bpb}"
    print("compute_metrics arithmetic check: OK")


def test_evaluate_split_covers_every_full_window():
    model_config = load_model_config()["model"]
    context_length = model_config["context_length"]
    model = DecoderLM(model_config)

    num_windows = 7
    num_tokens_in_data = num_windows * context_length + 50  # deliberate partial remainder
    data = torch.randint(0, model_config["vocab_size"], (num_tokens_in_data,))

    mean_loss_nats, total_tokens = evaluate_split(
        model, data, batch_size=4, context_length=context_length, device="cpu",
    )
    assert total_tokens == num_windows * context_length, \
        f"expected {num_windows * context_length} tokens evaluated, got {total_tokens}"
    assert mean_loss_nats == mean_loss_nats and abs(mean_loss_nats) != float("inf"), \
        "loss is NaN or Inf on a freshly-initialized model; forward pass is broken"
    assert mean_loss_nats > 0, "cross-entropy loss cannot be negative"
    print(f"evaluate_split windowing check: OK ({total_tokens:,} tokens over "
          f"{num_windows} windows, partial remainder correctly dropped)")


def demo():
    test_compute_metrics_matches_known_values()
    test_evaluate_split_covers_every_full_window()
    print("hindi eval_lm self-check: OK")


if __name__ == "__main__":
    demo()
