# Phase 3: Attention Analysis (Post-Finetune)

## Method

Reuses Phase 2's attention toolkit (`<lang>/eval/attention_analysis.py`)
unchanged in its actual attention-computation logic, pointed at real
comparative-reasoning prompts instead of generic held-out corpus text,
and at both a pretrained and a LoRA-finetuned (seed 0, merged) checkpoint
so heatmaps and entropy/distance metrics are directly comparable on the
task that actually changed.

Two real `test_seen` examples per language, picked for a genuine
short-range vs. long-range contrast, not randomly sampled:
- **pairwise**: 2 entities, 1 premise -- the answer only ever needs to
  attend back to one nearby premise.
- **chained**: 3 entities, 2 premises -- the correct entity is only
  inferable by combining both premises, a real long-range dependency
  across the full prompt.

Each sequence matches `finetune.py`'s exact training format: `<question
tokens> <s> <answer tokens> </s>`, so the heatmap shows attention FROM the
answer-generating position back onto the premise/entity tokens, the most
diagnostic view for whether finetuning changed how the model attends
across entities while reasoning.

Compared at one early layer (layer 0) and one late layer (layer 11 of 12)
per model, per spec's "at least one early and one late layer" requirement.

## Results: mean attention entropy and distance, early vs. late layer

Entropy in nats (higher = more diffuse attention across heads); mean
attention distance in tokens (higher = more long-range). Numbers below are
from the final checkpoints (post EOS-termination-fix, seed 0 merged, same
convention as before) -- pretrained-checkpoint numbers are identical to
the originally reported ones (the pretrained checkpoint never changed);
finetuned numbers shifted somewhat and are reported fresh, not reused.

**Hindi**

| Example | Layer | Metric | Pretrained | Finetuned (seed 0) |
|---|---|---|---|---|
| pairwise | early (0) | entropy | 2.008 | 1.993 |
| pairwise | early (0) | distance | 6.421 | 6.487 |
| pairwise | late (11) | entropy | 1.627 | 1.571 |
| pairwise | late (11) | distance | 4.618 | 4.827 |
| chained | early (0) | entropy | 2.577 | 2.562 |
| chained | early (0) | distance | 12.108 | 12.129 |
| chained | late (11) | entropy | 2.109 | 1.870 |
| chained | late (11) | distance | 8.242 | 6.705 |

Heatmaps below: each figure has one subplot per attention head. Row =
query position, column = key position, both labeled with the actual
token pieces of the example sentence. Color = attention weight from that
query to that key (viridis: dark purple near 0, yellow near 1, per-head
colorbar on the right of each subplot). The blank upper-right triangle in
every subplot is the causal mask -- a query can only attend to keys at or
before its own position. Each pair below is the same example and layer,
pretrained on the left, finetuned on the right.

*Pairwise, early layer (0)*

![Hindi pairwise early layer pretrained](figures/phase3_attention_reasoning_hindi_early_layer_pretrained_pairwise.png)
![Hindi pairwise early layer finetuned](figures/phase3_attention_reasoning_hindi_early_layer_finetuned_seed0_pairwise.png)

Entropy 2.008 to 1.993, distance 6.421 to 6.487 -- early-layer attention
on the short two-entity example is essentially unmoved by finetuning.

*Pairwise, late layer (11)*

![Hindi pairwise late layer pretrained](figures/phase3_attention_reasoning_hindi_late_layer_pretrained_pairwise.png)
![Hindi pairwise late layer finetuned](figures/phase3_attention_reasoning_hindi_late_layer_finetuned_seed0_pairwise.png)

Entropy drops 1.627 to 1.571 (more selective), but distance lengthens
slightly, 4.618 to 4.827 -- the one case in this report where distance
moves the opposite way from entropy (discussed below).

*Chained, early layer (0)*

![Hindi chained early layer pretrained](figures/phase3_attention_reasoning_hindi_early_layer_pretrained_chained.png)
![Hindi chained early layer finetuned](figures/phase3_attention_reasoning_hindi_early_layer_finetuned_seed0_chained.png)

Entropy 2.577 to 2.562, distance 12.108 to 12.129 -- like the pairwise
case, early-layer attention on the longer three-entity example barely
shifts.

*Chained, late layer (11)*

![Hindi chained late layer pretrained](figures/phase3_attention_reasoning_hindi_late_layer_pretrained_chained.png)
![Hindi chained late layer finetuned](figures/phase3_attention_reasoning_hindi_late_layer_finetuned_seed0_chained.png)

Entropy 2.109 to 1.870 and distance 8.242 to 6.705 -- the largest shift
in either metric for Hindi, on the example that actually requires
combining both premises.

**Assamese**

| Example | Layer | Metric | Pretrained | Finetuned (seed 0) |
|---|---|---|---|---|
| pairwise | early (0) | entropy | 2.213 | 2.194 |
| pairwise | early (0) | distance | 5.570 | 5.570 |
| pairwise | late (11) | entropy | 1.671 | 1.565 |
| pairwise | late (11) | distance | 6.061 | 5.635 |
| chained | early (0) | entropy | 2.573 | 2.522 |
| chained | early (0) | distance | 9.238 | 9.304 |
| chained | late (11) | entropy | 2.209 | 1.878 |
| chained | late (11) | distance | 11.335 | 9.477 |

Same reading as the Hindi heatmaps above: one subplot per head, row =
query position, column = key position (token pieces labeled), color =
attention weight (viridis, dark to yellow = 0 to 1), blank upper-right
triangle is the causal mask. Pretrained on the left, finetuned on the
right in each pair.

*Pairwise, early layer (0)*

![Assamese pairwise early layer pretrained](figures/phase3_attention_reasoning_assamese_early_layer_pretrained_pairwise.png)
![Assamese pairwise early layer finetuned](figures/phase3_attention_reasoning_assamese_early_layer_finetuned_seed0_pairwise.png)

Entropy 2.213 to 2.194, distance unchanged at 5.570 -- no real shift at
this layer for the short example.

*Pairwise, late layer (11)*

![Assamese pairwise late layer pretrained](figures/phase3_attention_reasoning_assamese_late_layer_pretrained_pairwise.png)
![Assamese pairwise late layer finetuned](figures/phase3_attention_reasoning_assamese_late_layer_finetuned_seed0_pairwise.png)

Entropy 1.671 to 1.565 and distance 6.061 to 5.635 -- both metrics move
the same direction here, unlike Hindi's equivalent pair.

*Chained, early layer (0)*

![Assamese chained early layer pretrained](figures/phase3_attention_reasoning_assamese_early_layer_pretrained_chained.png)
![Assamese chained early layer finetuned](figures/phase3_attention_reasoning_assamese_early_layer_finetuned_seed0_chained.png)

Entropy 2.573 to 2.522, distance 9.238 to 9.304 -- minor movement, same
pattern as every other early-layer pair in this report.

*Chained, late layer (11)*

![Assamese chained late layer pretrained](figures/phase3_attention_reasoning_assamese_late_layer_pretrained_chained.png)
![Assamese chained late layer finetuned](figures/phase3_attention_reasoning_assamese_late_layer_finetuned_seed0_chained.png)

Entropy 2.209 to 1.878 and distance 11.335 to 9.477 -- the largest shift
in either metric for Assamese, and larger than Hindi's equivalent pair
(discussed below).

## Discussion

**Early-layer attention is still essentially unchanged by finetuning**,
both languages, both example types (entropy and distance shift by a few
percent at most at layer 0, several entries under 1%). This is consistent
with LoRA's target choice: `q_proj`/`v_proj` adapters exist in every
layer, but the model's low-level token-composition behavior in early
layers is apparently not where the reasoning-task adaptation happens.

**Late-layer entropy still drops after finetuning in all four
(language x example-type) comparisons** (Hindi chained: 2.109 to 1.870;
Assamese chained: 2.209 to 1.878; both pairwise cases too, smaller
magnitude) -- the "more selective attention" finding holds. **Mean
attention distance is more mixed than originally reported**: it still
shortens for three of the four comparisons (Hindi chained: 8.242 to
6.705; Assamese pairwise: 6.061 to 5.635; Assamese chained: 11.335 to
9.477), but **Hindi pairwise distance now lengthens slightly instead of
shortening** (4.618 to 4.827, versus a shortening in the originally
reported run). This reversal was not present before the EOS-termination
fix -- the finetuned checkpoints being compared are different models
(the fix adds a second, small gradient signal through the embedding
table), so a small, single-example metric on the shortest/simplest prompt
type flipping sign is plausible rather than alarming, but it means the
"late-layer attention becomes sharper and more local, no exceptions"
claim from the original analysis is too strong and is revised here rather
than repeated.

This is still, on balance, the opposite of what "attention broadens to
gather more context for reasoning" would predict for the entropy metric
and for distance on three of four cases; finetuning appears to teach the
late layers to attend more selectively in general, likely to the specific
entity tokens that decide the comparison. The chained examples (the
genuinely long-range case) still show the largest absolute shift in both
entropy and distance, both languages -- consistent with finetuning
affecting head specialization most where the reasoning task actually
requires combining multiple premises.

**Which language shows the larger effect has flipped.** Originally Hindi
showed the larger chained-distance drop (2.399 tokens vs. Assamese's
1.427); with the post-fix checkpoints, **Assamese now shows the larger
drop** (1.858 tokens vs. Hindi's 1.537). Whether this tracks the
reasoning-accuracy trade-off documented in `phase3_reasoning_eval.md`
(Assamese absorbed more of a content-accuracy cost from the fix than
Hindi did) or is a smaller-sample artifact (one example per language per
type) is not resolved here -- this analysis is a qualitative
attention-shift check per spec's requirement, not a statistically powered
claim across many prompts, and that limitation is more visible now that a
methodology change flipped which language "wins" on this one metric.

## Reproduction

```bash
python3 hindi/eval/attention_reasoning_prompts.py \
    --pretrained-checkpoint <path>/hindi_checkpoint.pt \
    --finetuned-checkpoint <path>/hindi_finetuned_seed0_merged.pt
```

Writes `hindi/eval/attention_reasoning_metrics.json` and 8 heatmap PNGs
(`phase3_attention_reasoning_hindi_*.png`) to `report/figures/`. Same
command for `assamese`.
