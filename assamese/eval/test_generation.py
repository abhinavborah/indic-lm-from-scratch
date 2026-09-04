#!/usr/bin/env python3
"""Self-check for eval_generation.py's diversity/repetition math and the
generation loop's plumbing, independent of any trained checkpoint.

Run directly: python3 test_generation.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model import DecoderLM  # noqa: E402

from eval_generation import (  # noqa: E402
    distinct_n,
    generate,
    has_repetition_loop,
    load_model_config,
    repetition_rate,
    sample_prefix_windows,
)


def test_repetition_rate_on_known_sequence():
    seq = torch.tensor([1, 2, 1, 2, 1, 2])
    # bigrams: (1,2),(2,1),(1,2),(2,1),(1,2): 3 of 5 are repeats of an
    # earlier bigram (the 3rd, 4th, and 5th each repeat one already seen)
    rate = repetition_rate(seq, n=2)
    assert rate == 3 / 5, f"expected 3/5 repeated bigrams, got {rate}"
    no_repeat = torch.tensor([1, 2, 3, 4, 5])
    assert repetition_rate(no_repeat, n=2) == 0.0
    print("repetition_rate check: OK")


def test_distinct_n_on_known_sequence():
    seq = torch.tensor([1, 1, 1, 1])
    assert distinct_n(seq, n=1) == 1 / 4, f"expected 1/4 unique unigrams, got {distinct_n(seq, 1)}"
    all_unique = torch.tensor([1, 2, 3, 4])
    assert distinct_n(all_unique, n=1) == 1.0
    print("distinct_n check: OK")


def test_has_repetition_loop_detects_known_loop():
    looping = torch.tensor([9, 8, 7, 5, 6, 5, 6, 5, 6])
    assert has_repetition_loop(looping, min_phrase_len=2, min_repeats=3) is True
    non_looping = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert has_repetition_loop(non_looping, min_phrase_len=2, min_repeats=3) is False
    print("has_repetition_loop check: OK")


def test_sample_prefix_windows_spread_and_non_overlapping():
    data = torch.arange(100_000)
    windows = sample_prefix_windows(data, num_prefixes=10, window_len=256)
    assert len(windows) == 10
    assert all(w.shape[0] == 256 for w in windows)
    starts = sorted(w[0].item() for w in windows)
    assert starts == sorted(set(starts)), "expected non-overlapping/distinct windows"
    print("sample_prefix_windows check: OK")


def test_generate_produces_correct_length_and_greedy_is_deterministic():
    model_config = load_model_config()["model"]
    context_length = model_config["context_length"]
    model = DecoderLM(model_config)
    prefix = torch.randint(0, model_config["vocab_size"], (32,))
    gen_len = 16

    out_a = generate(model, prefix, gen_len, context_length, device="cpu", temperature=None, seed=0)
    out_b = generate(model, prefix, gen_len, context_length, device="cpu", temperature=None, seed=0)
    assert out_a.shape[0] == gen_len
    assert torch.equal(out_a, out_b), "greedy decoding must be deterministic for a fixed prefix"
    print("generate() length/determinism check: OK")


def demo():
    test_repetition_rate_on_known_sequence()
    test_distinct_n_on_known_sequence()
    test_has_repetition_loop_detects_known_loop()
    test_sample_prefix_windows_spread_and_non_overlapping()
    test_generate_produces_correct_length_and_greedy_is_deterministic()
    print("assamese eval_generation self-check: OK")


if __name__ == "__main__":
    demo()
