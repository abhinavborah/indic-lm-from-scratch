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
on test_seen (98.2% vs. 75.4% lenient exact match, mean of 3 seeds) and
test_unseen_entity (95.7% vs. 70.0%), tracking the same direction as the
Phase 2 LM-quality gap. **The pattern reverses on test_unseen_wording**:
Assamese scores higher (44.0%) than Hindi (29.6%), holding in every
individual seed, not just the mean. This is the one place Model L
outperforms Model H, and it does not simply follow the Phase 2 corpus-
size gap -- see `phase3_reasoning_eval.md` for the full per-seed table and
a candidate explanation (Hindi's higher seen-wording accuracy may carry a
larger template-memorization component that, by construction, does not
transfer to unseen phrasing).

Both models show the same qualitative finetuning failure mode: neither
learns to emit `</s>` reliably to terminate an answer (strict exact-match
accuracy is 0.0000 in every cell, both models, pretrained and finetuned),
traced to LoRA's target-module choice (`q_proj`/`v_proj` only) never
touching the frozen, untrained `</s>` embedding row. Both models still
show strong content-correctness gains under lenient (prefix-match)
scoring, confirming the finetune itself worked despite this shared
termination bug -- see `phase3_reasoning_eval.md` for the full mechanistic
account.

**Attention shift under finetuning is consistent in direction across
both languages**: late-layer attention becomes sharper and more local
after finetuning in every language/example-type comparison tested (entropy
and mean attention distance both drop), most strongly on the long-range
"chained" comparisons; early-layer attention is essentially unchanged.
The magnitude of this shift is somewhat larger for Hindi than Assamese
(`phase3_attention_reasoning.md`).

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
into the same pattern. The shared LoRA-EOS finding (both languages fail
strict termination identically) is evidence that at least one Phase 3
result is a property of the finetuning method (LoRA's target-module
choice interacting with untrained special-token embeddings under weight
tying) rather than of either language's data or resource tier -- an
important distinction for correctly attributing which Phase 3 results are
about H-vs-L and which are about the finetuning method itself.

## Reproducibility

See `README.md` for full reproduction steps and Google Drive links to all
checkpoints (pretrained and finetuned, both languages) and corpus
artifacts.
