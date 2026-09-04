"""Decoder-only causal Transformer for Hindi, built from raw nn.Linear /
nn.Embedding / nn.LayerNorm / nn.Dropout, no nn.Transformer*, no
pre-built attention block.

Architecture per docs-phase-2/depth_width_tradeoff.md: pre-LN, RoPE
positional embeddings, tied input/output embeddings.
"""

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    """Split the last dim in half and swap-negate: (x1, x2) -> (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def build_rope_cache(seq_len, head_dim, base=10000.0):
    """Precompute cos/sin tables for rotary position embeddings, shape (seq_len, head_dim) each."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """Rotate q/k by position-dependent angles. x: (B, n_head, T, head_dim)."""
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit causal mask and RoPE on q/k."""

    def __init__(self, d_model, n_head, context_length, dropout):
        super().__init__()
        assert d_model % n_head == 0, "d_model must divide evenly into n_head"
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        causal_mask = torch.triu(
            torch.full((context_length, context_length), float("-inf")), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x, cos, sin, return_attention=False):
        """x: (B, T, d_model) -> (B, T, d_model). Position t only attends to positions <= t.
        return_attention=False (default) changes nothing about the return value or
        computation, so existing checkpoints and tests are unaffected. When True,
        also returns the post-softmax attention weights, shape (B, n_head, T, T),
        for the attention-analysis eval (heatmaps, entropy, mean attention distance)."""
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att_scores = att_scores + self.causal_mask[:T, :T]
        att = F.softmax(att_scores, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        if return_attention:
            return out, att
        return out


class FeedForward(nn.Module):
    """Position-wise 4x-expansion MLP with GELU, applied independently at each position."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-LN transformer block: x + Attn(LN(x)), then x + FF(LN(x))."""

    def __init__(self, d_model, n_head, d_ff, context_length, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, context_length, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, cos, sin, return_attention=False):
        if return_attention:
            attn_out, att = self.attn(self.ln1(x), cos, sin, return_attention=True)
            x = x + attn_out
            x = x + self.ff(self.ln2(x))
            return x, att
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ff(self.ln2(x))
        return x


class DecoderLM(nn.Module):
    """GPT-style decoder-only causal language model."""

    def __init__(self, config):
        super().__init__()
        if config["positional_embedding"] != "rope":
            raise NotImplementedError("only rope positional embeddings are implemented")
        if config["norm_style"] != "pre_ln":
            raise NotImplementedError("only pre_ln is implemented")

        self.config = config
        d_model = config["d_model"]
        n_head = config["n_head"]
        head_dim = d_model // n_head

        self.token_embedding = nn.Embedding(config["vocab_size"], d_model)
        self.drop = nn.Dropout(config["dropout"])
        self.blocks = nn.ModuleList(
            [
                Block(d_model, n_head, config["d_ff"], config["context_length"], config["dropout"])
                for _ in range(config["n_layer"])
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, config["vocab_size"], bias=False)

        cos, sin = build_rope_cache(config["context_length"], head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2-style scaled init for the two per-block projections that feed
        # directly back into the residual stream, so their variance doesn't
        # compound across n_layer additions (standard nanoGPT/GPT-2 practice).
        residual_std = 0.02 / math.sqrt(2 * config["n_layer"])
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

        if config["tie_embeddings"]:
            self.head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        """PyTorch's default nn.Embedding init (std=1) is far too large for this
        use case; every reference small-model implementation checked
        (GPT-2, nanoGPT) overrides it, along with nn.Linear, to std=0.02."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, return_attention=False):
        """idx: (B, T) token ids -> logits: (B, T, vocab_size).
        return_attention=False (default) is unchanged from before this flag
        existed, same computation, same return value, so existing
        checkpoints and tests are unaffected. When True, also returns a list
        of per-layer attention weights (n_layer tensors, each (B, n_head, T, T)),
        for the attention-analysis eval only."""
        T = idx.shape[1]
        assert T <= self.config["context_length"], "sequence longer than context_length"

        x = self.drop(self.token_embedding(idx))
        if return_attention:
            attentions = []
            for block in self.blocks:
                x, att = block(x, self.rope_cos, self.rope_sin, return_attention=True)
                attentions.append(att)
            x = self.ln_f(x)
            return self.head(x), attentions
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.ln_f(x)
        return self.head(x)

    def num_parameters(self):
        """Total trainable parameter count (tied weights counted once)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config_file(cls, path):
        """Build a model from a model_config.json file's "model" section."""
        with open(path, encoding="utf-8") as f:
            full_config = json.load(f)
        return cls(full_config["model"])
