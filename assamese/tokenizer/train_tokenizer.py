#!/usr/bin/env python3
"""Byte-level BPE tokenizer training for Assamese, via sentencepiece.

Uses sentencepiece as the BPE engine (course guidance: use a library rather
than hand-rolling merge counting), but layered with this project's own
decisions rather than stock defaults:
  - `preprocess.py`'s purity filter runs first -- the trainer only ever sees
    already-cleaned Assamese text, not raw scrape/OCR/Sangraha noise.
  - `byte_fallback=True` + `character_coverage=1.0` gives byte-level
    fallback for anything outside the learned vocab, so encoding never hits
    a hard UNK on unseen characters (the closest sentencepiece equivalent to
    a from-scratch byte-level BPE's guarantee).
  - `vocab_size` is NOT fixed here -- `vocab_sweep.py` picks it empirically
    via fertility/UNK-rate on held-out text, this module just trains one
    candidate at a time.
  - `special_tokens` lets Phase 3 finetuning add its own control tokens
    (e.g. role markers) without touching the core training call.

Do not run this against the real corpus yet -- scaffold + unit test only
until preprocessing (dedup/split/clean) is finished. See test_tokenizer.py
for the current, small-sample self-check.
"""

from pathlib import Path

import sentencepiece as spm

DEFAULT_SPECIAL_TOKENS = ()  # e.g. ("<|user|>", "<|assistant|>") once Phase 3 needs them


def train(input_path, model_prefix, vocab_size, special_tokens=DEFAULT_SPECIAL_TOKENS):
    """Train a byte-level BPE model on an already-cleaned text file.

    input_path: plain text, one cleaned segment per line (preprocess.py's output).
    model_prefix: writes `<model_prefix>.model` and `<model_prefix>.vocab`.
    """
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        byte_fallback=True,
        character_coverage=1.0,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        user_defined_symbols=list(special_tokens),
    )
    return f"{model_prefix}.model"


def load(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp
