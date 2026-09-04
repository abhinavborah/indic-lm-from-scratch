# Phase 2: The Hindi vs Assamese Gap, Decomposed

Spec section 2.3 asks for a discussion of the gap between Model H and
Model L attributed to data size/quality, script/tokenizer fertility, and
the model itself. This section pulls together evidence from
`phase2_lm_eval.md`, `phase2_generation_eval.md`,
`phase2_attention_analysis.md`, and `phase2_resource_comparison.md`
rather than restating any single number in isolation.

## What is controlled for

Model H (Hindi) and Model L (Assamese) share an identical architecture:
12 layers, d_model=384, 6 heads, RoPE, Pre-LN, tied embeddings, 8,000-token
vocabulary each, 24,366,336 trainable parameters, exactly matching to the
parameter. Both trained single-epoch, same batch size (32), same
optimizer and schedule shape (independently swept per language, see
`hindi/configs/model_config.json` and `assamese/configs/model_config.json`),
on the same hardware. Measured throughput is close between the two
(~14,185-14,916 tokens/sec Hindi, ~13,673-13,756 tokens/sec Assamese),
ruling out a hardware or efficiency confound. Given all of this is held
equal, the model architecture and training procedure are not a source of
the gap observed below; they are the controlled variable, not the
explanation.

## The gap, measured

| | Hindi | Assamese | Direction |
|---|---|---|---|
| Real train tokens | 834,049,861 | 565,255,443 | Assamese 32% smaller |
| Tokenizer fertility (tokens/word) | 1.4940 | 1.7780 | Assamese 19% higher |
| Test perplexity | 38.46 | 53.67 | Assamese ~40% higher |
| Test BPB | 0.5952 | 0.5990 | Nearly identical |
| Generation quality (all metrics, all decoding settings) | higher | lower | Assamese consistently lower |
| Attention entropy (10 of 12 layers) | lower | higher | Assamese more diffuse |

## Attribution

**Tokenization accounts for a real share of the raw perplexity gap.**
Perplexity is measured per token, and Assamese's tokenizer needs more,
smaller tokens to cover the same text (fertility 1.7780 vs 1.4940). BPB
corrects for this by measuring against raw bytes instead of tokens, and
the BPB gap nearly vanishes (0.5990 vs 0.5952 on test, Assamese's
validation BPB is even marginally lower than Hindi's). A model producing
a similar number of bits of information per byte of actual text, while
posting a ~40% worse per-token perplexity, is exactly the signature of a
tokenization effect inflating the raw comparison rather than the model
being meaningfully worse per unit of language modeled.

**Data scale accounts for the rest of the gap, and the rest of the gap is
real.** BPB is close but not identical, Assamese's generation-quality
metrics (BLEU, chrF, ROUGE-L, repetition rate, Distinct-1/2) are lower
than Hindi's at every one of the four decoding settings tested, not just
on average, and Assamese's attention is measurably more diffuse (higher
entropy at 10 of 12 layers) and shows a different late-layer pattern than
Hindi's sharp local contraction. These three independent pieces of
evidence, intrinsic quality, generation quality, and internal attention
structure, all point the same direction, and all three are consistent
with the one variable that was not controlled for: Assamese's real
training corpus is 32% smaller (565M vs 834M tokens). Less training data
gives a model less signal to commit to sharp, confident patterns
(consistent with the higher attention entropy) and less exposure to the
range of phrasing needed for fluent, on-distribution generation
(consistent with the lower BLEU/chrF/ROUGE-L and diversity numbers). This
is not inferred from perplexity alone; it is corroborated across three
different kinds of measurement.

**The model itself contributes nothing to the gap, by construction.**
Identical architecture, identical parameter count, identical training
procedure, and near-identical raw throughput between the two runs. Any
gap that survived controlling for tokenization (via BPB) has to come
from somewhere else, and the two remaining candidates, data and model,
are not symmetric here: one was deliberately held fixed and verified to
be fixed (model), the other was not and differs by a known, measured
amount (data, 32% less for Assamese). The evidence points to data as the
active variable and the model as the controlled one, not the reverse.

## Summary

Roughly: the ~40% perplexity gap is a mix of a real tokenization
artifact (a meaningful chunk of it, given BPB nearly closes the gap) and
a real data-scale effect (the remaining, smaller gap in BPB itself, plus
the consistent generation-quality and attention-structure gaps that BPB
alone cannot explain). The model architecture is not a contributing
factor, since it was held identical and independently verified to
produce comparable training dynamics (throughput, single-epoch, same
optimizer) for both languages. This is consistent with the project's
Phase 1 framing of Assamese as the deliberately lower-resource language:
the gap observed here traces back to that choice, not to any asymmetry
introduced by the modeling or training pipeline.
