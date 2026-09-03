#!/usr/bin/env python3
"""Generates the Phase 2 loss-curve figures for both languages.

Kept separate from the training/eval scripts (per this project's
convention of keeping visualization and computation apart), reads
directly from each language's real loss-log CSV rather than recomputing
or hardcoding any numbers.

The loss-log CSVs themselves are small (tens of KB) and committed to
this repo under report/logs/, unlike the checkpoints and tokenized
corpus .bin files, which stay Drive-only per the large-artifact policy.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

FIGURES_DIR = __file__.rsplit("/", 1)[0] + "/figures"
CHECKPOINTS_DIR = Path(__file__).parent / "logs"

LANGUAGE_COLORS = {"hindi": "#2a78d6", "assamese": "#eb6834"}


def load_loss_log(language):
    path = CHECKPOINTS_DIR / f"{language}_loss_log.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    return steps, train_loss, val_loss


def plot_loss_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, language in zip(axes, ("hindi", "assamese")):
        steps, train_loss, val_loss = load_loss_log(language)
        color = LANGUAGE_COLORS[language]
        ax.plot(steps, train_loss, color=color, alpha=0.5, linewidth=0.8, label="Train loss")
        ax.plot(steps, val_loss, color=color, linewidth=1.6, label="Validation loss")
        ax.set_xlabel("Training step")
        ax.set_title(f"{language.capitalize()}: Training and Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Cross-entropy loss (nats)")
    fig.suptitle("Phase 2 Loss Curves: Model H (Hindi) vs. Model L (Assamese)")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/phase2_loss_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_loss_curves()
    print(f"figures written to {FIGURES_DIR}")
