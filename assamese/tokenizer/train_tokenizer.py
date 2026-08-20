#!/usr/bin/env python3
"""Byte-level BPE tokenizer training for Assamese, via sentencepiece.

Uses sentencepiece as the BPE engine (course guidance: use a library rather
than hand-rolling merge counting), but layered with this project's own
decisions rather than stock defaults:
  - `preprocess.py`'s purity filter runs first -- the trainer only ever sees
    already-cleaned Assamese text, not raw scrape/OCR/Sangraha noise.
  - `byte_fallback=True` + `character_coverage=0.9995` gives byte-level
    fallback for anything outside the learned vocab, so encoding never hits
    a hard UNK on unseen characters (the closest sentencepiece equivalent to
    a from-scratch byte-level BPE's guarantee). Coverage is 0.9995, not 1.0:
    at full-corpus scale the real character alphabet is far noisier than a
    small proxy sample suggested (5,828 unique characters found in the real
    Assamese train split, mostly rare stray symbols/noise that slipped past
    line-level purity filtering), and 1.0 demands a dedicated vocab slot for
    every single one of them -- at 5,000 vocab that alone exceeds the whole
    budget before any real subword merging happens. 0.9995 is sentencepiece's
    own default for exactly this (see doc/options.md: "use 1.0 for languages
    with small alphabets... 0.9995 for large character sets"); the rare tail
    still round-trips losslessly via byte_fallback instead of forcing a
    top-level vocab slot.
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


DEFAULT_INPUT_SENTENCE_SIZE = 5_000_000  # sentencepiece's own recommended way to bound
# training cost on a large corpus (see doc/options.md) -- samples this many lines from
# the input rather than loading the entire multi-hundred-million-token train split.
# Matches the HackMD-sanctioned "train on a sample, justified" approach, just done via
# the library's own mechanism instead of a hand-built sample file.


def train(input_path, model_prefix, vocab_size, special_tokens=DEFAULT_SPECIAL_TOKENS,
          input_sentence_size=DEFAULT_INPUT_SENTENCE_SIZE):
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
        character_coverage=0.9995,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        user_defined_symbols=list(special_tokens),
    )
    return f"{model_prefix}.model"


def load(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp
