# Phase 2: Architecture, Design Choices, and Justification

Both models (Model H, Hindi; Model L, Assamese) share the same architecture,
built entirely from raw `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, and
`nn.Dropout`. No `nn.Transformer*` module, no HuggingFace model class, no
pre-built attention block is used anywhere in `model.py`. Model H and
Model L never share weights, data, or a vocabulary; only the architecture
shape and hyperparameters are identical, chosen once and applied
independently to each language.

Measured parameter count (`DecoderLM.num_parameters()`, both languages,
tied embeddings): **24,366,336**.

## Configuration

| Hyperparameter | Value |
|---|---|
| n_layer | 12 |
| d_model | 384 |
| n_head | 6 (head_dim = 64) |
| d_ff | 1536 (4x expansion) |
| context_length | 256 |
| dropout | 0.1 |
| vocab_size | 8,000 (both languages, independently trained) |
| positional scheme | RoPE |
| normalization | Pre-LN |
| output head | tied to input token embedding |

## Positional embeddings: RoPE

Self-attention has no inherent notion of token order, so position must be
injected explicitly. The spec allows learned absolute embeddings,
sinusoidal encodings, or a relative scheme such as RoPE. RoPE was chosen.

RoPE (Su et al., "RoFormer: Enhanced Transformer with Rotary Position
Embedding," arXiv:2104.09864) rotates each query/key vector at position
`p` by an angle proportional to `p`, applied inside attention rather than
at the embedding layer. Because rotation is linear, the dot product
between a rotated query at position `m` and a rotated key at position `n`
depends only on the relative offset `m - n`, not on either absolute
position, so relative position falls out of the mechanism algebraically
rather than needing to be learned from added vectors. This adds no
parameters: nothing is stored per position, only a precomputed sine/cosine
table reused every forward pass (a non-persistent buffer in `model.py`,
not a learned tensor).

**How this constrains maximum sequence length.** RoPE has no fixed
embedding table to run out of rows in, unlike learned absolute embeddings,
which have a hard architectural cap at `max_seq_len` rows. RoPE's angles
are computable for any position via a closed-form function, so there is no
hard cap in principle. In practice, this does not mean sequences much
longer than the trained context length are handled reliably: a model
trained only on sequences up to `context_length` has not learned to
interpret the unfamiliar phase values that longer positions would produce,
and the follow-up literature on rotary embeddings finds their length
extrapolation is real but weaker than schemes designed specifically for
it (e.g. ALiBi). The practical ceiling for this project is therefore the
trained `context_length = 256`, enforced by training dynamics rather than
by table size, not the theoretical absence of a hard cap.

At this project's scale (~25M parameters, a few hundred million training
tokens), RoPE is not simply a large-model default imported downward
without evidence: a directly comparable data point, a 28M-parameter
LLaMA-style model trained on TinyStories comparing APE, RoPE, and ALiBi
with everything else held constant, found RoPE and ALiBi both
outperforming learned absolute embeddings at that scale (Suresh,
"Positional Embeddings in Transformers: A Math Guide to RoPE & ALiBi,"
Towards Data Science, 2025). Modern small-model reference implementations
have also converged on RoPE independent of scale: nanoGPT (a GPT-2
reproduction) uses learned absolute embeddings, but Karpathy's newer
from-scratch project nanochat moved to RoPE with no positional embedding
table at all, and most post-2022 small-LM families that publish
architecture details default to RoPE regardless of parameter count.

## Multi-head causal self-attention

For an input sequence `X` of shape `(B, T, d_model)`, `X` is projected
into queries, keys, and values with three separate `nn.Linear` layers,
each producing shape `(B, T, d_model)`. These are reshaped to
`(B, T, n_head, head_dim)` and transposed to `(B, n_head, T, head_dim)`
so each head operates in its own `head_dim = d_model / n_head = 64`
dimensional subspace, computed in parallel across heads via batched
matrix multiplication. RoPE is applied to the query and key tensors at
this point, before the attention scores are computed.

Attention scores are computed as `(Q @ K^T) / sqrt(head_dim)`. The
`1 / sqrt(head_dim)` scaling exists because the dot product of two
random vectors grows with the dimensionality they live in: without it,
as `head_dim` grows, raw dot-product scores would grow large in
magnitude, pushing the softmax into a regime where gradients vanish for
all but the single largest score. Dividing by `sqrt(head_dim)` keeps the
variance of the scores roughly constant regardless of head dimension, so
softmax stays in a well-conditioned range for gradient flow, independent
of the particular head size chosen.

**Causal masking.** An additive mask of shape `(context_length,
context_length)` is built once (`torch.triu(..., diagonal=1)`, stored as
a non-persistent buffer, not a learned parameter): 0 on allowed
positions, `-inf` on future positions. This is added to the attention
scores before softmax, so a `-inf` score becomes exactly 0 probability
after softmax, and position `t` can only ever attend to positions `<= t`.
This is implemented directly, not via any library masking utility.

**Empirical verification.** `test_model.py` (both languages) proves the
mask actually works, not just that it is present: it runs the same input
sequence twice, once unmodified and once with a single token changed at
position `t`, and asserts the logits at every position `< t` are bit-identical
between the two runs. If the causal mask leaked future information, this
test would fail. This is the spec's own suggested verification, and it
passes for both models.

After attention, the heads are concatenated back to `(B, T, d_model)` and
passed through an output projection (`nn.Linear`), also learned from
scratch.

## Transformer block: pre-norm

Each block contains a multi-head causal self-attention sublayer and a
position-wise feed-forward network (two `nn.Linear` layers with GELU,
inner dimension `d_ff = 1536`, a 4x expansion over `d_model`), each
wrapped in a residual connection with dropout applied to both attention
probabilities and sublayer outputs. Normalization is placed before each
sublayer (Pre-LN: `x = x + Sublayer(LayerNorm(x))`), not after the
residual add (Post-LN, the original 2017 Transformer's placement), with a
final LayerNorm after the last block.

Pre-LN was chosen for two reasons. First, the theoretical stability
argument (Xiong et al., "On Layer Normalization in the Transformer
Architecture," ICML 2020): at initialization, Post-LN gradients near the
output layer scale with model depth, requiring a carefully tuned learning
rate warmup to avoid divergence, while Pre-LN's un-normalized residual
path gives gradients a low-resistance shortcut to early layers,
regardless of depth. This is specifically a depth argument, and this
project's 12-layer, ~25M-parameter model is far shallower than the
regime the instability was demonstrated at; a follow-up paper (Takase et
al., "On Layer Normalizations and Residual Connections in Transformers,"
2022) finds Post-LN can match or slightly outperform Pre-LN at 6 layers
or fewer in their experiments, so the stability argument alone does not
unconditionally favor Pre-LN at this depth.

Second, and more decisively: every actively maintained small-model
reference implementation checked (GPT-2, including its smallest 117M/12-layer
configuration; GPT-J; the LLaMA family; nanoGPT, including its smallest
from-scratch demo configs at single-digit millions of parameters) uses
Pre-LN, none uses Post-LN by default at any scale. Given this project
builds every component from raw layers with no library attention block
to lean on, Pre-LN was chosen for having fewer moving parts to get wrong:
no required warmup-schedule tuning, well-behaved gradients from the first
training step, at the cost of a contested few points of final-loss
quality that the shallow-depth literature suggests Post-LN might
otherwise offer.

## Output head: weight tying

The output projection (final hidden state to vocabulary-sized logits) is
tied to the input token embedding: the same `[vocab_size, d_model]`
tensor is used as a row-lookup on the way in and a matrix multiply on
the way out (`W_out = E^T`). This is legal because both matrices map
between the same two spaces, token identity and `d_model`-dimensional
vector space, from opposite directions (Press & Wolf, "Using the Output
Embedding to Improve Language Models," EACL 2017), and it is standard
practice in every small-model reference checked: GPT-2's original release
computes logits directly from the input embedding tensor with no separate
output weight, and nanoGPT does the same explicitly.

At this project's `vocab_size = 8,000` and `d_model = 384`, tying saves
`8000 x 384 = 3,072,000` parameters, 12.3% of the ~25M budget, either
converted into additional transformer capacity (attention and FFN
layers) or left unused as duplicate lookup-table weights if untied. Given
a fixed parameter ceiling, tying was chosen to keep that capacity in the
transformer itself. One complication is logged rather than ignored: a
2026 mechanistic-interpretability paper (Lopardo, Harish, Arnett & Gupta,
"Weight Tying Biases Token Embeddings Towards the Output Space,"
arXiv:2603.26663) shows tied embeddings end up shaped more by output-side
("predict the next token") gradients than input-side ("represent this
token") gradients, measurably reducing early-layer contribution to the
residual stream. Their own framing is that this is a real but
size-dependent cost, and that the parameter-efficiency case for tying is
strongest exactly at the scale this project sits at (embeddings are a
much larger share of the parameter budget at 25M than at billion-parameter
scale), so the efficiency benefit was judged to outweigh the
representational cost their paper identifies, not that the cost does not
exist.

## Depth and width

12 layers, `d_model = 384`, 6 heads (`head_dim = 64`), `d_ff = 1536` was
chosen over shallower/wider alternatives (e.g. 6 layers, `d_model = 512`)
for three reasons.

First, shape sensitivity at a fixed non-embedding parameter count is weak
according to the scaling-law literature: Kaplan et al. ("Scaling Laws for
Neural Language Models," arXiv:2001.08361) found a 40x swing in aspect
ratio (`d_model / n_layer`) changed loss by only a few percent at fixed
parameter count, though their experiments were run at 30 to 60 times this
project's scale, so the finding is informative but not directly verified
at ~25M parameters. Second, a paper studying depth/width directly at
smaller scale (Levine et al., "The Depth-to-Width Interplay in
Self-Attention," arXiv:2006.12467) found that at small network sizes,
excessively deep, narrow networks can perform worse than shallower ones,
an empirically observed failure mode on the narrow-and-deep extreme
specifically, not just a null effect. `head_dim = 64` was kept fixed
(matching every reference small-model configuration checked, including
GPT-2, GPT-Neo-125M, and nanoGPT) to stay clear of that width-starved
regime. Third, and specific to this project's own evaluation needs: the
required attention analysis (heatmaps, entropy, mean attention distance
across early/mid/late layers) needs enough discrete layers to bucket
into a real early/mid/late breakdown. 12 layers gives four layers per
bucket; a 6-layer model would give only two layers per bucket, a
before/after contrast rather than a trend. This consideration is
independent of the scaling-law findings above and does not trade away
any loss-related benefit, since neither paper found evidence favoring
the shallower shape on quality grounds at this budget.

The same shape is used for both Model H and Model L: nothing in the
literature reviewed suggests the optimal shape should differ by language
at the token counts both corpora reach (both are well past the
Chinchilla-optimal token budget for a 25M model), and using an identical
shape keeps the "fully independent but architecturally comparable" story
the cross-language evaluation depends on.

## Training objective

Both models are trained with the causal language modeling objective:
cross-entropy between the logits at position `t` and the true token at
position `t + 1`, averaged over all positions in the batch (`train.py`,
`torch.nn.functional.cross_entropy`), using AdamW, a warmup-then-cosine
learning rate schedule, periodic validation evaluation during training,
and full-state checkpointing (model weights, optimizer state, training
step, and configuration, saved atomically). The learning rate schedule
is a deterministic function of the saved training step (`lr_at_step`),
not a separate stateful scheduler object, so no additional scheduler
state is needed to resume a run exactly: the step count alone is
sufficient to reconstruct the schedule's position. This mirrors
nanoGPT's own `train.py`, which computes LR via a manual `get_lr(it)`
function rather than a `torch.optim.lr_scheduler` object and likewise
saves no separate scheduler state in its checkpoint dict.
