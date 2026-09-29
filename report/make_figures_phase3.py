#!/usr/bin/env python3
"""Generates the Phase 3 finetuning figures for both languages: per-seed
train/reasoning-val loss curves, and pretrain-val PPL tracked across epochs
as a forgetting diagnostic (not an auto-stop signal, see
report/phase3_reasoning_eval.md).

Same convention as make_figures_phase2.py: visualization kept separate
from computation, reads directly from the small per-epoch CSV logs
committed under report/logs/phase3_finetune/, one real run per seed, no
recomputation or hardcoded numbers.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).parent / "figures"
LOGS_DIR = Path(__file__).parent / "logs" / "phase3_finetune"

LANGUAGE_COLORS = {"hindi": "#2a78d6", "assamese": "#eb6834"}
SEED_STYLES = {0: "-", 1: "--", 2: ":"}
SEEDS = [0, 1, 2]


def load_finetune_log(language, seed):
    path = LOGS_DIR / f"{language}_finetune_seed{seed}_log.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    reasoning_val_loss = [float(r["reasoning_val_loss"]) for r in rows]
    pretrain_val_ppl = [float(r["pretrain_val_ppl"]) for r in rows]
    return epochs, train_loss, reasoning_val_loss, pretrain_val_ppl


def plot_finetune_loss_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, language in zip(axes, ("hindi", "assamese")):
        color = LANGUAGE_COLORS[language]
        for seed in SEEDS:
            epochs, train_loss, reasoning_val_loss, _ = load_finetune_log(language, seed)
            style = SEED_STYLES[seed]
            ax.plot(epochs, train_loss, color=color, alpha=0.35, linewidth=1.0, linestyle=style,
                     label=f"seed {seed} (train)")
            ax.plot(epochs, reasoning_val_loss, color=color, alpha=0.9, linewidth=1.8,
                     linestyle=style, label=f"seed {seed} (val)")
        ax.set_xlabel("Epoch")
        ax.set_title(f"{language.capitalize()}: LoRA Finetuning Loss (3 seeds)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Cross-entropy loss (nats)")
    fig.suptitle("Phase 3 Reasoning Finetuning: Train and Validation Loss, Model H vs. Model L")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase3_finetune_loss_curves.png", dpi=150)
    plt.close(fig)


def plot_pretrain_ppl_forgetting():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, language in zip(axes, ("hindi", "assamese")):
        color = LANGUAGE_COLORS[language]
        for seed in SEEDS:
            epochs, _, _, pretrain_val_ppl = load_finetune_log(language, seed)
            ax.plot(epochs, pretrain_val_ppl, color=color, linewidth=1.6,
                     linestyle=SEED_STYLES[seed], label=f"seed {seed}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pretrain-val perplexity")
        ax.set_title(f"{language.capitalize()}: Pretrain-Val PPL During Finetuning")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Phase 3 Catastrophic-Forgetting Diagnostic: Pretrain-Val PPL vs. Epoch")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase3_pretrain_ppl_forgetting.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_finetune_loss_curves()
    plot_pretrain_ppl_forgetting()
    print(f"figures written to {FIGURES_DIR}")
