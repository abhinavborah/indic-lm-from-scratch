#!/usr/bin/env python3
"""Generates the Phase 1 dataset-statistics figures for both languages.

Kept separate from the data/tokenizer computation scripts (per this
project's convention of keeping visualization and computation apart) --
every number plotted here is copied from what token_progress.md and the
vocab_sweep fertility runs already computed and reported this session, not
recomputed here.
"""

import matplotlib.pyplot as plt

FIGURES_DIR = __file__.rsplit("/", 1)[0] + "/figures"


def plot_manual_vs_downloaded():
    languages = ["Hindi", "Assamese"]
    manual = [135_301_892, 74_452_715]
    downloaded = [431_013_561, 246_026_167]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(languages))
    ax.bar(x, manual, label="Manual", color="#2a78d6")
    ax.bar(x, downloaded, bottom=manual, label="Downloaded", color="#eb6834")
    ax.set_xticks(list(x))
    ax.set_xticklabels(languages)
    ax.set_ylabel("Rough tokens (whitespace count)")
    ax.set_xlabel("Language")
    ax.set_title("Manual vs. Downloaded Tokens by Language")
    ax.legend()
    ax.axhline(100_000_000, color="gray", linestyle="--", linewidth=1)
    ax.text(1.4, 105_000_000, "100M manual floor", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/manual_vs_downloaded.png", dpi=150)
    plt.close(fig)


def plot_train_val_test_split():
    languages = ["Hindi", "Assamese"]
    train = [555_124_385, 314_223_263]
    val = [5_453_827, 3_112_133]
    test = [5_737_241, 3_143_486]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(languages))
    ax.bar(x, train, label="Train (98%)", color="#2a78d6")
    ax.bar(x, val, bottom=train, label="Validation (1%)", color="#1baf7a")
    bottoms = [t + v for t, v in zip(train, val)]
    ax.bar(x, test, bottom=bottoms, label="Test (1%)", color="#eda100")
    ax.set_xticks(list(x))
    ax.set_xticklabels(languages)
    ax.set_ylabel("Rough tokens (whitespace count)")
    ax.set_xlabel("Language")
    ax.set_title("Train / Validation / Test Split Sizes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/train_val_test_split.png", dpi=150)
    plt.close(fig)


def plot_fertility_vs_vocab():
    vocab_sizes = [5000, 8000, 10000, 15000, 24000, 32000]
    hindi = [2.1086, 1.9772, 1.9239, 1.8390, 1.7574, 1.7149]
    assamese = [2.369, 2.1769, 2.099, 1.9790, 1.8626, 1.8024]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(vocab_sizes, hindi, marker="o", label="Hindi", color="#2a78d6")
    ax.plot(vocab_sizes, assamese, marker="o", label="Assamese", color="#eb6834")
    ax.axvline(8000, color="#2a78d6", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(5000, color="#eb6834", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks(vocab_sizes)
    ax.set_xticklabels([f"{v // 1000}K" for v in vocab_sizes])
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Fertility (tokens/word)")
    ax.set_title("Vocabulary Size vs. Fertility (proxy sweep, 32M-char sample)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fertility_vs_vocab.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_manual_vs_downloaded()
    plot_train_val_test_split()
    plot_fertility_vs_vocab()
    print(f"figures written to {FIGURES_DIR}")
