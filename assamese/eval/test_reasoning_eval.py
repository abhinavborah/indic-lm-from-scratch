#!/usr/bin/env python3
"""Self-check for reasoning_eval.py's generation-stopping and scoring logic,
independent of any trained checkpoint (uses a stub model with forced
outputs, not a real forward pass).

Run directly: python3 test_reasoning_eval.py
"""

import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from reasoning_eval import EOS_ID, evaluate_split, generate_answer  # noqa: E402

TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "tokenizer" / "assamese_bpe_8000.model"


class ForcedOutputModel:
    """Ignores its input entirely; each call returns logits whose argmax at
    the last position is the next id in a fixed, pre-scripted sequence --
    lets the stopping/cap logic be tested without a real forward pass."""

    def __init__(self, forced_ids, vocab_size=8000):
        self.forced_ids = forced_ids
        self.vocab_size = vocab_size
        self.calls = 0

    def __call__(self, x):
        logits = torch.full((1, x.shape[1], self.vocab_size), -100.0)
        next_id = self.forced_ids[self.calls]
        logits[0, -1, next_id] = 100.0
        self.calls += 1
        return logits


def test_generate_answer_stops_at_eos_before_emitting_it():
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    real_ids = [50, 51]  # arbitrary in-vocab ids, no semantic meaning needed
    model = ForcedOutputModel([real_ids[0], real_ids[1], EOS_ID, 9999])
    result = generate_answer(model, sp, "প্ৰশ্ন", context_length=64, device="cpu", max_answer_tokens=10)
    assert model.calls == 3, f"expected exactly 3 forward calls (stop right after EOS), got {model.calls}"
    assert result == sp.decode(real_ids), f"expected decoded real tokens only, got {result!r}"


def test_generate_answer_respects_cap_when_eos_never_emitted():
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    forced = [100, 101, 102, 103, 104, 105, 106, 107]  # no EOS_ID anywhere
    model = ForcedOutputModel(forced)
    generate_answer(model, sp, "প্ৰশ্ন", context_length=64, device="cpu", max_answer_tokens=5)
    assert model.calls == 5, f"expected exactly max_answer_tokens=5 calls when EOS never fires, got {model.calls}"


def test_evaluate_split_reports_strict_and_lenient_separately():
    examples = [
        {"prompt": "p1", "answer": "ৰাম"},   # strict and lenient correct (clean EOS stop)
        {"prompt": "p2", "answer": "গীতা"},  # only lenient correct (right answer, then degenerate loop)
        {"prompt": "p3", "answer": "সীতা"},   # both wrong
    ]
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    ram_ids = sp.encode("ৰাম", out_type=int)
    gita_looping_ids = sp.encode("গীতা০গীতা০", out_type=int)  # right answer, then a repeat -- the real observed pattern
    wrong_ids = sp.encode("ভুল", out_type=int)

    canned = {"p1": ram_ids, "p2": gita_looping_ids, "p3": wrong_ids}

    def stub_generate_ids(model, sp, prompt, context_length, device):
        return canned[prompt]

    strict_acc, lenient_acc, n, qualitative = evaluate_split(
        model=None, sp=sp, examples=examples, context_length=64, device="cpu",
        generate_ids_fn=stub_generate_ids,
    )
    assert n == 3
    assert abs(strict_acc - 1 / 3) < 1e-9, f"expected 1/3 strict accuracy (only p1), got {strict_acc}"
    assert abs(lenient_acc - 2 / 3) < 1e-9, f"expected 2/3 lenient accuracy (p1 and p2), got {lenient_acc}"
    assert len(qualitative) == 3
    assert qualitative[0]["strict_correct"] is True and qualitative[0]["lenient_correct"] is True
    assert qualitative[1]["strict_correct"] is False and qualitative[1]["lenient_correct"] is True
    assert qualitative[2]["strict_correct"] is False and qualitative[2]["lenient_correct"] is False


def test_lenient_match_requires_prefix_not_substring():
    """Lenient scoring is a PREFIX match, not "expected appears anywhere" --
    a wrong answer that happens to contain the right one as a substring
    must not count as correct."""
    examples = [{"prompt": "p1", "answer": "ৰাম"}]
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    contains_but_not_prefix_ids = sp.encode("গীতাৰাম", out_type=int)  # "ৰাম" appears, but not at the start

    def stub_generate_ids(model, sp, prompt, context_length, device):
        return contains_but_not_prefix_ids

    _, lenient_acc, _, _ = evaluate_split(
        model=None, sp=sp, examples=examples, context_length=64, device="cpu",
        generate_ids_fn=stub_generate_ids,
    )
    assert lenient_acc == 0.0, "expected substring-but-not-prefix to score as wrong under the lenient rule"


def demo():
    test_generate_answer_stops_at_eos_before_emitting_it()
    print("generate_answer EOS-stop check: OK")
    test_generate_answer_respects_cap_when_eos_never_emitted()
    print("generate_answer max-cap check: OK")
    test_evaluate_split_reports_strict_and_lenient_separately()
    print("evaluate_split strict/lenient scoring check: OK")
    test_lenient_match_requires_prefix_not_substring()
    print("lenient prefix-not-substring check: OK")
    print("assamese reasoning_eval self-check: OK")


if __name__ == "__main__":
    demo()
