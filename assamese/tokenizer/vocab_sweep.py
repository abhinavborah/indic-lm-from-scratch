#!/usr/bin/env python3
"""Scaffold for the vocab-size sweep: fertility + UNK-rate on held-out text.

Structure: train one candidate model per vocab size, measure fertility
(tokens per whitespace word) and UNK rate on a held-out sample. This was
run against the real corpus for the 5K/8K/10K candidates reported in
report/phase1_dataset_statistics.md, in addition to an earlier proxy sweep
over a wider range of sizes on a smaller sample.
"""

from train_tokenizer import train, load

DEFAULT_CANDIDATE_SIZES = (4_000, 8_000, 16_000)  # Assamese's smaller real corpus likely wants a smaller vocab than Hindi's


def fertility(sp, held_out_lines):
    """Average sentencepiece tokens per whitespace word, over held-out text."""
    total_tokens, total_words = 0, 0
    for line in held_out_lines:
        words = line.split()
        if not words:
            continue
        total_words += len(words)
        total_tokens += len(sp.encode(line, out_type=int))
    return total_tokens / total_words if total_words else float("nan")


def unk_rate(sp, held_out_lines):
    unk_id = sp.unk_id()
    total_tokens, unk_tokens = 0, 0
    for line in held_out_lines:
        ids = sp.encode(line, out_type=int)
        total_tokens += len(ids)
        unk_tokens += sum(1 for i in ids if i == unk_id)
    return unk_tokens / total_tokens if total_tokens else float("nan")


def sweep(cleaned_corpus_path, held_out_lines, model_prefix_base,
          candidate_sizes=DEFAULT_CANDIDATE_SIZES):
    results = []
    for vocab_size in candidate_sizes:
        model_path = train(cleaned_corpus_path, f"{model_prefix_base}_{vocab_size}", vocab_size)
        sp = load(model_path)
        results.append({
            "vocab_size": vocab_size,
            "fertility": fertility(sp, held_out_lines),
            "unk_rate": unk_rate(sp, held_out_lines),
        })
    return results
