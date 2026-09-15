# Final Report: Model H (Hindi) vs. Model L (Assamese)

This report consolidates the project across all three phases and answers
the four questions the Phase 3 spec requires. It draws on, rather than
restates, the per-phase reports: `phase1_dataset_statistics.md`,
`phase2_lm_eval.md`, `phase2_generation_eval.md`,
`phase2_attention_analysis.md`, `phase2_gap_decomposition.md`,
`phase2_resource_comparison.md`, `phase3_reasoning_eval.md`, and
`phase3_attention_reasoning.md`. Cross-reference those for full detail;
this document synthesizes.

## What was built

Two from-scratch, fully independent monolingual decoder-only transformers
(~24.3M parameters each, identical architecture: 12 layers, d_model=384,
6 heads, RoPE, Pre-LN, tied embeddings). Model H is Hindi, a
higher-resource Indian language; Model L is Assamese, a deliberately
lower-resource one. No pretrained weights, no pretrained tokenizers, no
HuggingFace model classes anywhere; embeddings, multi-head causal
self-attention, and the causal mask are all built from raw `nn.Linear` /
`nn.Embedding` / `nn.LayerNorm` / `nn.Dropout`. The two languages share no
data, no vocabulary, and no weights at any stage, including the Phase 3
LoRA finetune.

## 1. How did data scale and quality differ between Model H and Model L?

**Scale.** After real tokenization, Hindi's training corpus is
834,049,861 tokens; Assamese's is 565,255,443 tokens, roughly 32% smaller.
Both clear the spec's ~500M-token target and the ≥20% manual-collection
floor, but by design Assamese sits closer to the lower-resource end: both
languages' manual and downloaded token counts were tracked and reported
against the corpus scale requirement (see `phase1_dataset_statistics.md`'s
overview table and `manual_vs_downloaded.png`).

**Quality.** Both corpora went through the same cleaning pipeline
(Unicode normalization, script-purity filtering at the line level,
word-level stripping of non-target-script/non-digit tokens, near-
duplicate removal, document-level 98/1/1 train/val/test split by content
hash). Assamese's collection needed a materially wider and more varied
source mix to reach scale (SCERT/NCERT OCR, three Wikimedia dump
projects, a dictionary site, government archives, a broader archive.org
query, YouTube auto-captions, eight news/literary sites, four downloaded
corpora) versus Hindi's narrower, higher-yield set (NCERT OCR, one book
archive, Wikipedia, Vikaspedia, two fully-drained plus two still-active
news sites, one downloaded corpus). This reflects a real quality/scale
tradeoff inherent to a lower-resource language: reaching the same token
target required tapping thinner, more heterogeneous sources, not a
difference in cleaning rigor applied to either language.

**Tokenizer.** Both use a SentencePiece BPE tokenizer with 8,000-token
vocabularies, chosen via the course-prescribed fertility sweep on held-out
text, not guessed. Assamese's real fertility (1.7780 tokens/word) is
substantially higher than Hindi's (1.4940), meaning Assamese needs more,
smaller subword pieces to represent the same amount of text, a direct
consequence of it being the lower-resource, more morphologically complex
language in this pair (see `phase1_dataset_statistics.md`'s vocabulary-
size section for the full sweep, including how the initial 5,000-vocab
proxy estimate for Assamese was revised upward to 8,000 once real
fertility measurements ruled out a severe token-count cost).

## 2. How do language-modeling and reasoning results compare across the two resource tiers?

**Language modeling (Phase 2).** Hindi's test perplexity (38.46) is about
40% lower than Assamese's (53.67), but this gap shrinks to nearly nothing
once measured in bits-per-byte instead of per-token perplexity (0.5952
Hindi vs. 0.5990 Assamese) -- see `phase2_gap_decomposition.md`. Hindi's
generated text also scores higher on every quality metric (BLEU, chrF,
ROUGE-L) and diversity diagnostic (repetition rate, Distinct-1/2) at every
one of the four decoding settings tested (`phase2_generation_eval.md`),
and Hindi's attention is measurably sharper (lower entropy at 10 of 12
layers, `phase2_attention_analysis.md`) than Assamese's more diffuse
pattern.

**Reasoning (Phase 3).** After LoRA finetuning on each language's own
synthetic comparative-reasoning dataset, Hindi again outperforms Assamese
on test_seen (98.4% vs. 65.2% strict exact match, mean of 3 seeds) and
test_unseen_entity (96.8% vs. 58.2%), tracking the same direction as the
Phase 2 LM-quality gap. **The pattern reverses on test_unseen_wording**:
Assamese scores higher (43.8%) than Hindi (26.2%), holding in every
individual seed, not just the mean. This is the one place Model L
outperforms Model H, and it does not simply follow the Phase 2 corpus-
size gap -- see `phase3_reasoning_eval.md` for the full per-seed table and
a candidate explanation (Hindi's higher seen-wording accuracy may carry a
larger template-memorization component that, by construction, does not
transfer to unseen phrasing).

Both models initially showed the same qualitative finetuning failure mode:
neither learned to emit `</s>` reliably to terminate an answer (strict
exact-match accuracy was 0.0000 in every cell, both models, pretrained and
finetuned), traced to LoRA's target-module choice (`q_proj`/`v_proj` only)
never touching the frozen, untrained `</s>` embedding row. This was fixed,
not just documented: the two affected embedding rows were made trainable
via a gradient hook (isolating them from the other ~8,000 rows, since
`requires_grad` has no per-row granularity), with their own zero-weight-decay
optimizer group to stop decay leaking into them. Verified via `torch.allclose`
on all 6 final checkpoints (3 seeds x 2 languages): the two rows now differ
from the pretrained checkpoint, 100 sampled other rows per checkpoint stay
bit-identical. **The fix landed cleanly for Hindi** (strict now equals
lenient exactly, every split) **but left a small residual termination gap
for Assamese** (strict trails lenient by 1-3 points on `test_seen`/
`test_unseen_entity`) alongside a real, measurable drop in Assamese's raw
content accuracy on those same two splits relative to its pre-fix numbers
(consistent across all three seeds) -- `test_unseen_wording` is unaffected
for both languages. See `phase3_reasoning_eval.md` for the full mechanistic
account and a candidate explanation for the Assamese-specific cost.

**Attention shift under finetuning is consistent in direction across
both languages on entropy** (late-layer attention entropy drops after
finetuning in every language/example-type comparison, most strongly on
the long-range "chained" comparisons; early-layer attention is essentially
unchanged), but **mean attention distance is more mixed**: it shortens
(more local attention) in three of four language/example-type comparisons,
but lengthens slightly for Hindi's pairwise case in the post-fix
checkpoints -- a reversal from what was originally measured, attributable
to the finetuned checkpoints themselves changing under the EOS fix, not a
methodology change in how the metric is computed. Which language shows
the larger chained-distance shift has also flipped: Assamese now shows the
larger drop, where Hindi originally did (`phase3_attention_reasoning.md`).

## 3. What tokenizer / corpus factors most affected the lower-resource model?

Two factors, and their relative contributions can be separated using the
BPB-vs-perplexity comparison:

**Tokenizer fertility accounts for a real share of Assamese's apparent
perplexity disadvantage, but not the whole gap.** Perplexity is measured
per token; Assamese's tokenizer produces more tokens per word (fertility
1.7780 vs. 1.4940) to cover the same text, which inflates its per-token
perplexity relative to Hindi even if the two models are comparably good
per unit of actual language content. Correcting for this via
bits-per-byte (which normalizes against raw text bytes rather than token
count) nearly closes the gap entirely (0.5990 vs. 0.5952), the textbook
signature of a tokenization artifact rather than a real quality
difference.

**Corpus scale accounts for the remaining, smaller gap, and it is real.**
The residual BPB difference, Assamese's consistently lower generation-
quality metrics at every decoding setting, and Assamese's measurably more
diffuse attention (higher entropy at 10 of 12 layers) all point the same
direction and are all consistent with the one variable genuinely left
uncontrolled between the two models: Assamese's 32%-smaller real training
corpus. Model architecture, parameter count, training procedure, and
measured throughput were all held identical or near-identical between the
two runs (`phase2_gap_decomposition.md`'s "What is controlled for"
section), ruling out the model itself as a contributing factor.

## 4. What evidence explains the observed differences?

The evidence is convergent across three independent measurement types,
not inferred from a single number:

1. **Intrinsic language-modeling quality** (perplexity, corrected via BPB
   for the tokenization confound).
2. **Generation quality** (BLEU/chrF/ROUGE-L and diversity/repetition
   diagnostics across 4 decoding settings, an entirely separate evaluation
   pipeline from perplexity).
3. **Internal model structure** (attention entropy and mean attention
   distance, a mechanistic probe rather than an output-quality metric).

All three point toward corpus scale, not tokenizer choice or model
capacity, as the dominant remaining factor after tokenization is
accounted for. The Phase 3 reasoning results largely track this same
attribution (Hindi ahead on 2 of 3 splits), with one genuine exception
(test_unseen_wording) that does not reduce to the corpus-size story and is
reported as an open, evidence-grounded observation rather than forced
into the same pattern.

The LoRA-EOS finding is evidence of both a method-level cause and a
resource-tier-dependent consequence, and the two should not be conflated.
Before the fix, both languages failed strict termination identically
(0.0000 in every cell) -- clear evidence that the root cause (LoRA's
target-module choice never touching the frozen, untrained `</s>`
embedding row) is a property of the finetuning method, not of either
language's data. After fixing it, the two languages diverged: Hindi's fix
is completely clean (strict equals lenient exactly, no other numbers
moved), while Assamese retains a small residual termination gap and shows
a measurable, seed-consistent drop in raw content accuracy on `test_seen`/
`test_unseen_entity` that Hindi does not. That divergence *is* plausibly
resource-tier-dependent -- the fix adds a second gradient signal competing
for the same tiny LoRA-scale training budget, and Assamese's smaller,
noisier pretraining signal may make it more sensitive to sharing that
budget than Hindi's is. So: the bug was a method artifact; how cleanly the
fix resolves is itself H-vs-L evidence.

## Reproducibility

See `README.md` for full reproduction steps and Google Drive links to all
checkpoints (pretrained and finetuned, both languages) and corpus
artifacts.
