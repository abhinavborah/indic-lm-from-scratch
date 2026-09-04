#!/usr/bin/env python3
"""Attention analysis, per spec section 2.3: heatmaps (at least one early
layer, one late layer, multiple heads), attention entropy per head/layer,
and mean attention distance, on real example sentences from the held-out
test split (2026-08-31 decision: real corpus text, not hand-crafted
sentences, since this is Phase 2's language-modeling/attention evaluation,
not yet Phase 3's reasoning task).

Uses model.py's return_attention flag (additive, does not change logits
or existing checkpoint compatibility; see model.py's forward docstring).

Run directly: python3 attention_analysis.py --checkpoint-path /path/to/hindi_checkpoint.pt
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import sentencepiece as spm
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "train"))

from model import DecoderLM  # noqa: E402
from train import load_checkpoint  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "model_config.json"
TOKENIZER_PATH = ROOT / "tokenizer" / "hindi_bpe_8000.model"
DEFAULT_DATA_DIR = ROOT / "data" / "splits"
FIGURES_DIR = ROOT.parent / "report" / "figures"

NUM_EXAMPLE_SENTENCES = 4
MIN_SENTENCE_TOKENS = 10
MAX_SENTENCE_TOKENS = 40

# matplotlib's default font (DejaVu Sans) has no Devanagari glyphs, so
# heatmap tick labels render as blank boxes without a script-capable
# fallback font, found and fixed after the first real run produced
# unreadable Devanagari labels. Registering Kohinoor Devanagari as a
# fallback (not the sole font) keeps Latin/ASCII pieces like "4." and
# the byte-fallback "<0xNN>" tokens legible too, which a single forced
# non-Latin font would not have covered.
fm.fontManager.addfont("/System/Library/Fonts/Kohinoor.ttc")
plt.rcParams["font.family"] = ["DejaVu Sans", "Kohinoor Devanagari"]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def sample_example_sentences(text_path, tokenizer_path, num_sentences):
    """Real segments from the held-out test split (blank-line-delimited,
    same unit build_splits.py writes), short enough for a legible
    query-vs-key heatmap, evenly spread through the file rather than
    clustered at the start."""
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    with open(text_path, encoding="utf-8") as f:
        segments = [s.strip() for s in f.read().split("\n\n") if s.strip()]

    candidates = []
    for seg in segments:
        ids = sp.encode(seg, out_type=int)
        if MIN_SENTENCE_TOKENS <= len(ids) <= MAX_SENTENCE_TOKENS:
            candidates.append((seg, ids))

    n = min(num_sentences, len(candidates))
    step = max(1, len(candidates) // n)
    chosen = [candidates[i * step] for i in range(n)]
    return sp, chosen


@torch.no_grad()
def compute_attention(model, token_ids, device):
    """token_ids: 1D tensor. Returns a list of n_layer tensors, each
    (n_head, T, T), attention weights for this single example."""
    model.eval()
    x = token_ids.unsqueeze(0).to(device)
    _, attentions = model(x, return_attention=True)
    return [a[0].cpu() for a in attentions]


def entropy_per_head(attn_layer):
    """attn_layer: (n_head, T, T). Entropy of each query position's
    distribution over key positions, averaged over query positions
    (skipping position 0, whose distribution is trivially a single point
    under causal masking) and returned per head."""
    n_head, T, _ = attn_layer.shape
    eps = 1e-12
    entropies = []
    for h in range(n_head):
        row_entropies = []
        for t in range(1, T):
            p = attn_layer[h, t, : t + 1]
            row_entropies.append(-(p * (p + eps).log()).sum().item())
        entropies.append(sum(row_entropies) / len(row_entropies))
    return entropies


def mean_attention_distance_per_head(attn_layer):
    """attn_layer: (n_head, T, T). For each query position t, the
    attention-weighted average of |t - key_position| over valid keys
    (0..t), averaged over query positions and returned per head."""
    n_head, T, _ = attn_layer.shape
    distances = []
    for h in range(n_head):
        row_distances = []
        for t in range(1, T):
            p = attn_layer[h, t, : t + 1]
            key_positions = torch.arange(t + 1, dtype=torch.float32)
            dist = (p * (t - key_positions)).sum().item()
            row_distances.append(dist)
        distances.append(sum(row_distances) / len(row_distances))
    return distances


def plot_heatmaps(attn_layer, pieces, layer_label, out_path):
    """One subplot per head, query (rows) vs key (columns) positions,
    tick labels are the actual token pieces for this example sentence."""
    n_head, T, _ = attn_layer.shape
    ncols = min(3, n_head)
    nrows = (n_head + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axes = axes.flatten() if n_head > 1 else [axes]
    for h in range(n_head):
        ax = axes[h]
        im = ax.imshow(attn_layer[h].numpy(), cmap="viridis", vmin=0, vmax=attn_layer[h].max().item())
        ax.set_xticks(range(T))
        ax.set_xticklabels(pieces, rotation=90, fontsize=6)
        ax.set_yticks(range(T))
        ax.set_yticklabels(pieces, fontsize=6)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        ax.set_title(f"Head {h}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for h in range(n_head, len(axes)):
        axes[h].axis("off")
    fig.suptitle(f"Hindi, {layer_label}: attention weights, all heads")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(checkpoint_path, data_dir=None):
    full_config = load_model_config()
    model_config = full_config["model"]
    n_layer = model_config["n_layer"]

    device = pick_device()
    print(f"device={device}")

    model = DecoderLM(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.1)
    step = load_checkpoint(checkpoint_path, model, optimizer, map_location=device)
    model.to(device)
    print(f"loaded checkpoint at step {step:,}")

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    sp, examples = sample_example_sentences(data_dir / "test.txt", TOKENIZER_PATH, NUM_EXAMPLE_SENTENCES)
    print(f"sampled {len(examples)} example sentences from test.txt")

    early_layer, late_layer = 0, n_layer - 1
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    entropy_by_layer = [[] for _ in range(n_layer)]
    distance_by_layer = [[] for _ in range(n_layer)]

    for ex_idx, (seg_text, ids) in enumerate(examples):
        token_ids = torch.tensor(ids, dtype=torch.long)
        pieces = [sp.id_to_piece(i) for i in ids]
        attentions = compute_attention(model, token_ids, device)

        for layer_idx, attn_layer in enumerate(attentions):
            entropy_by_layer[layer_idx].append(entropy_per_head(attn_layer))
            distance_by_layer[layer_idx].append(mean_attention_distance_per_head(attn_layer))

        if ex_idx == 0:
            plot_heatmaps(attentions[early_layer], pieces, f"layer {early_layer} (early)",
                          FIGURES_DIR / "phase2_attention_hindi_early_layer.png")
            plot_heatmaps(attentions[late_layer], pieces, f"layer {late_layer} (late)",
                          FIGURES_DIR / "phase2_attention_hindi_late_layer.png")
            print(f"heatmaps written for example: {seg_text[:60]}...")

    n_head = model_config["n_head"]
    entropy_summary = []
    distance_summary = []
    for layer_idx in range(n_layer):
        per_head_entropy = [sum(ex[h] for ex in entropy_by_layer[layer_idx]) / len(entropy_by_layer[layer_idx])
                             for h in range(n_head)]
        per_head_distance = [sum(ex[h] for ex in distance_by_layer[layer_idx]) / len(distance_by_layer[layer_idx])
                              for h in range(n_head)]
        entropy_summary.append([round(v, 4) for v in per_head_entropy])
        distance_summary.append([round(v, 4) for v in per_head_distance])

    print("entropy per layer (averaged over heads):")
    for layer_idx, heads in enumerate(entropy_summary):
        print(f"  layer {layer_idx}: mean={sum(heads)/len(heads):.4f} per_head={heads}")
    print("mean attention distance per layer (averaged over heads):")
    for layer_idx, heads in enumerate(distance_summary):
        print(f"  layer {layer_idx}: mean={sum(heads)/len(heads):.4f} per_head={heads}")

    return {
        "language": "hindi", "checkpoint_step": step,
        "num_example_sentences": len(examples), "n_layer": n_layer, "n_head": n_head,
        "entropy_per_layer_per_head": entropy_summary,
        "mean_attention_distance_per_layer_per_head": distance_summary,
        "example_sentences": [seg for seg, _ in examples],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=str(ROOT / "eval" / "attention_metrics.json"))
    args = parser.parse_args()

    output = run(args.checkpoint_path, data_dir=args.data_dir)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
