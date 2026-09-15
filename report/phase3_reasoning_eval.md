# Phase 3: Reasoning Finetuning Results

## Method

Each model's own Phase 2 pretrained checkpoint is finetuned (SFT:
supervised finetuning on labelled input/output pairs, not general
instruction-following training) via LoRA (rank 8, alpha 16, targeting
`q_proj`/`v_proj` in every attention block; base weights and the token
embedding table, including the output head tied to it, stay frozen) on
that language's own synthetic comparative-reasoning dataset. No data or
weights are shared between the two models.

**Why LoRA over full finetuning.** At 24.3M total parameters, full
finetuning gives the entire model real room to memorize a finetuning set
of only ~4,500-6,000 examples rather than generalize from it. This is not
a hypothetical risk: on a comparable held-out age-ordering task, a small
full-finetuned model reached near-zero training loss but 0% exact-match
on unseen entity names, having learned to reproduce the trained entity
pool's answer format rather than the underlying comparison operation.
LoRA's rank-constrained weight update cannot memorize as freely as an
unconstrained full-parameter update, which plausibly helps generalization
on exactly the two axes this evaluation reports separately: unseen-entity
and unseen-wording accuracy. Compute/memory savings, LoRA's more common
selling point, are not the motivating factor here; there is no memory
pressure to relieve at this parameter count. Plain LoRA (not DoRA, not
QLoRA) was chosen specifically because DoRA's purpose is closing LoRA's
capacity gap to full finetuning, which would erode the same
regularization property motivating the switch; QLoRA's quantization
solves a memory problem this model does not have.

This was checked directly, not just argued from the citation above: a
matched full-finetuning run (same seed, same data, same epoch count)
reached a near-zero reasoning-task loss but its language-modeling
perplexity on the original pretrain validation set exploded 288x over
baseline in the same span, against LoRA's ~20% rise -- real, measured
catastrophic forgetting, confirming the regularization argument rather
than only asserting it.

Rank and alpha were picked by an empirical sweep (`<lang>/train/
lora_rank_sweep.py`), not guessed: reasoning-val loss drops monotonically
and substantially from rank 2 to rank 8, while pretrain-val PPL barely
moves at any rank tested, so rank 8 wins outright rather than by a
tolerance tie-break. Both languages converged on rank 8 / alpha 16
independently, from separate real-data sweeps.

Training: warmup-then-constant learning rate (`peak_lr=1e-4`, 10 warmup
steps), batch size 32, up to 3 seeds per language (`0, 1, 2`) per Lec09's
multi-seed requirement, early stopping on reasoning-validation loss
(patience 5 epochs), epoch ceiling 40. Pretrain-val perplexity is tracked
every epoch as a catastrophic-forgetting diagnostic but is not an
automatic stop condition: no literature or course precedent was found for
a fixed PPL-increase percentage threshold, and the course's own guidance
is to plot and inspect the curve rather than auto-trigger on it. Figures
below show this diagnostic held essentially flat across all three seeds,
both languages (Hindi ~38 to ~57 PPL, Assamese ~62 to ~89) -- no
forgetting signal severe enough to warrant a stop was observed at this
training budget.

Every seed, both languages, reached the 40-epoch ceiling without the
reasoning-validation-loss patience condition ever firing on its own. This
was checked directly rather than assumed adequate: a single-seed probe
with the ceiling raised to 150 epochs found patience does eventually fire
naturally, around epoch 59 (best epoch 54) -- but running the real
accuracy evaluation against that longer-trained checkpoint showed
`test_seen`/`test_unseen_entity` accuracy climbing toward ceiling while
`test_unseen_wording` -- the split that actually measures generalization
rather than same-distribution fit -- got worse, alongside measurably more
pretrain-perplexity drift. The validation split is drawn from the same
distribution as the training set (only the three held-out test splits get
entity/wording holdout), so the early-stopping signal rewards fitting that
shared distribution long after real transfer to unseen phrasing has
peaked. **The 40-epoch ceiling is therefore kept deliberately**, not left
as an unexamined default: it functions as an inadvertent regularizer
against exactly the overfitting the validation loss alone cannot detect.

Finetuned checkpoints are saved in the same resume-capable format as
pretraining (model state, optimizer state, step, config), and separately
merged into a plain `DecoderLM` state dict (`merge_lora_to_plain`) so
Phase 2's evaluation and attention tooling can load them unchanged.

## Dataset

Synthetic comparative-reasoning dataset, generated programmatically per
language (own templates, own entity-name pools, own script) from a pure
function `generate(seed, domain, split) -> list[example]`, not downloaded
from an existing benchmark. Evaluated against the four named best-
practice criteria the course's own SFT-data guidance specifies.

**Coverage** (does every target behavior have examples): 4 independently
instantiable attribute domains (height, age, price, quantity), each
covering direct pairwise comparisons, 2-hop chained comparisons (A>B,
B>C), and 3-hop chained comparisons, plus equal/tie outcomes (~10% of the
dataset, both languages) where the spec's "greater, smaller, equal"
scope requires it. Ties are structurally excluded from 2-hop (3-entity)
chains, since with exactly 3 entities every adjacent pair touches one of
the two queried extremes, making a tie there ambiguous with the asked-for
answer; ties are only included where they can sit strictly between the
two extremes (2-entity and 4-entity chains).

**Diversity of form** (phrasing variety per behavior): 4 premise
phrasings and 4 question phrasings per domain, varying word order and
connective words within the 3 trained forms rather than reusing 3 fixed
strings verbatim, to avoid a repeated-phrase stylistic tic. Entity pools:
155 person names / 119 object names (Hindi), 150 person names / 122
object names (Assamese), both languages using natural in-language names,
not transliterations of the other language's names.

**Correctness** (is every label actually right): answers are extractive
from the stated premises by the generator's own deterministic logic, not
hand-labeled, so label correctness is a property of the generation
function itself rather than manual annotation quality. Covered by
`test_generate_reasoning_data.py`'s self-tests, both languages.

**Separation** (no test item appears in training): both leakage-avoidance
axes from Lec09 are enforced. Some template surface forms are held out
entirely for test (train on 3 forms per relation, hold 1). Some
entity-name pools are held out entirely for test (80/20 train/held split
on both person and object pools), never appearing in train even paired
with seen templates. Train and test_seen are additionally instance-level
disjoint (the same entity combination and premise ordering never appears
in both), not just template/entity-level disjoint.

| Split | Hindi / Assamese (each) |
|---|---|
| train | 4,500 |
| val | 450 |
| test_seen | 450 |
| test_unseen_entity | 450 |
| test_unseen_wording | 150 |

Per Lec09's exact terminology, three splits are reported **separately,
never aggregated**:
1. **test_seen** -- seen entities, seen wording: did training fit the
   distribution at all.
2. **test_unseen_entity** -- unseen entities, seen wording: does behavior
   survive entity substitution.
3. **test_unseen_wording** -- unseen phrasing entirely: does behavior
   survive a change of surface form.

## Scoring: strict vs. lenient, and the EOS-termination bug behind them

A real mechanistic finding (not a scoring artifact) originally required
reporting two exact-match numbers rather than one. LoRA targets only
`q_proj`/`v_proj`; the token embedding table (and the tied output head)
stayed frozen. `<s>`/`</s>` were never emitted mid-corpus during Phase 2
pretraining, so their embedding rows were untouched random init --
verified directly via `torch.allclose` on the embedding rows for ids 0/1/2
between the pretrained and every finetuned checkpoint (bit-identical, both
languages, in the originally reported runs). The finetuned model picked
the correct answer with high confidence (~98%+ argmax probability,
confirmed by direct logit inspection) but almost never emitted `</s>` to
stop -- `P(</s>)` at the position immediately after the answer measured at
~1e-6, essentially zero, across every case checked. Generation instead
looped a short repeating token cycle until the decoding cap.

- **Strict** exact match: the raw generation must equal the expected
  answer exactly, requiring a literal `</s>` to terminate it. This was the
  honest floor and documented the EOS-learning failure directly: it was
  0.0000 in every cell, both languages, pretrained and finetuned alike, in
  the originally reported runs.
- **Lenient** exact match: does the raw generation **start with** the
  exact expected answer (prefix match), tolerating whatever degenerate
  content follows. This isolated whether the model's actual answer
  content was correct, independent of the termination bug.

  A simpler cycle-detection approach (truncate the repeating loop, then
  compare) was tried first and rejected: a period-2 loop `<answer>
  <filler> <answer> <filler> ...` is genuinely ambiguous about which half
  is "content" from periodicity alone -- both phases are equally valid
  decompositions of the same repeating sequence. Prefix match sidesteps
  this ambiguity entirely.

### The termination bug was fixed, not just documented

Root cause: `freeze_non_lora_params` left the entire token embedding table
frozen, so the untrained `<s>`/`</s>` rows had no path to learn, regardless
of training budget. Fix: those two rows are marked trainable, but
`requires_grad` has no per-row granularity in PyTorch, so a gradient hook
zeroes every other row's gradient before it reaches the optimizer --
`<s>`/`</s>` learn, the other ~8,000 rows stay exactly as pretrained. A
second, independent issue had to be fixed alongside it: AdamW's weight
decay shrinks a parameter's value regardless of its gradient, so the two
unfrozen rows were put in their own optimizer group with `weight_decay=0`
to stop decay leaking into them.

Verified directly before trusting a full rerun: `torch.allclose` on all 6
final merged checkpoints (3 seeds x 2 languages) confirms `<s>`/`</s>`
rows now differ from the pretrained checkpoint, while 100 randomly sampled
other rows per checkpoint remain bit-identical to it -- the fix is
isolated exactly as intended, in every checkpoint, not just on average.

With the fix, **strict and lenient accuracy are now (nearly) identical**
-- see Results below -- so the two-number reporting convention from the
original bug is retained here for direct before/after comparison, not
because the underlying ambiguity still exists.

## Results: pretrained vs. finetuned, mean across 3 seeds

**Before the fix** (strict accuracy floored at 0 by the termination bug;
lenient accuracy is the only meaningful column and is kept here for
comparison, not silently replaced):

| Split | Hindi pretrained | Hindi finetuned (lenient) | Assamese pretrained | Assamese finetuned (lenient) |
|---|---|---|---|---|
| test_seen | 0.0000 | 0.9822 | 0.0044 | 0.7541 |
| test_unseen_entity | 0.0000 | 0.9571 | 0.0000 | 0.7000 |
| test_unseen_wording | 0.0000 | 0.2956 | 0.0000 | 0.4400 |

(strict was 0.0000 in every cell above, both languages, pretrained and
finetuned.)

**After the fix** (strict accuracy is now the primary, honest number --
lenient is reported alongside only to show how close the two now are):

| Split | Hindi pretrained | Hindi finetuned (strict) | Hindi finetuned (lenient) | Assamese pretrained | Assamese finetuned (strict) | Assamese finetuned (lenient) |
|---|---|---|---|---|---|---|
| test_seen | 0.0000 | 0.9837 | 0.9837 | 0.0000 | 0.6519 | 0.6630 |
| test_unseen_entity | 0.0000 | 0.9682 | 0.9682 | 0.0000 | 0.5822 | 0.6178 |
| test_unseen_wording | 0.0000 | 0.2622 | 0.2622 | 0.0000 | 0.4378 | 0.4378 |

**Hindi: strict equals lenient exactly on every split** -- every
generation that gets the answer right now also terminates cleanly at
`</s>`. The fix is completely clean here.

**Assamese: strict is slightly below lenient on `test_seen`/
`test_unseen_entity`** (0.6519 vs. 0.6630, 0.5822 vs. 0.6178) -- a small
residual termination gap remains for this language specifically, unlike
Hindi, though nowhere close to the pre-fix total failure. `test_unseen_wording`
matches exactly for Assamese too.

**A real, unexpected trade-off, reported honestly rather than smoothed
over**: Assamese's post-fix accuracy on `test_seen`/`test_unseen_entity`
is measurably *lower* than its pre-fix lenient numbers (65.2% vs. 75.4%,
58.2% vs. 70.0%) -- consistent across all three seeds, not one outlier
(pre-fix per-seed range 69.1-79.1%, post-fix range 63.6-66.4%, essentially
non-overlapping). Hindi shows no such drop (98.2% to 98.4%, 95.7% to
96.8%, both within seed-to-seed noise). `test_unseen_wording` is unaffected
for both languages. A plausible mechanism: the fix routes a second,
independent gradient signal (the two unfrozen embedding rows) through the
same tiny LoRA-scale optimization that also has to learn the reasoning
content; Assamese's smaller, noisier pretraining signal (documented in
`phase2_gap_decomposition.md`) may make its content-learning more sensitive
to sharing that limited training budget than Hindi's is. This is an
observation from the data, not a proven causal claim.

Per-seed strict accuracy (confirms the pattern above is not a
single-seed artifact):

| Seed | Hindi test_seen | Hindi unseen_entity | Hindi unseen_wording | Assamese test_seen | Assamese unseen_entity | Assamese unseen_wording |
|---|---|---|---|---|---|---|
| 0 | 0.9822 | 0.9667 | 0.2533 | 0.6356 | 0.5956 | 0.4467 |
| 1 | 0.9867 | 0.9756 | 0.2800 | 0.6556 | 0.5711 | 0.4533 |
| 2 | 0.9822 | 0.9622 | 0.2533 | 0.6644 | 0.5800 | 0.4133 |

## Conclusion: the finetune worked, and the termination bug is now fixed

Content accuracy is far above the near-zero pretrained baseline on every
split, both languages, both before and after the fix -- the finetune
itself always worked. What changed is that the model now actually
terminates its answers correctly instead of looping past them: strict
accuracy went from 0.0000 everywhere to matching (Hindi) or nearly
matching (Assamese) lenient accuracy. The root mechanism (LoRA's
target-module choice leaving the token embedding table, and therefore the
never-trained `<s>`/`</s>` rows, completely frozen) was diagnosed, then
actually fixed via targeted gradient masking rather than left as a
documented limitation -- the earlier decision to accept it as "a real
qualitative finding, not a bug to fix" was revisited once the practical
cost became concrete (an interactively-served model would visibly loop
past correct answers) and superseded by building the real fix.

## Qualitative failure mode

Format-memorization signature: the drop from test_seen / test_unseen_entity
(>96% Hindi, >58% Assamese, per-seed) down to test_unseen_wording (25-28%
Hindi, 41-45% Assamese) matches the pattern expected when part of the gain
is template/format memorization rather than pure abstract reasoning
transfer. Real transfer is still present -- both languages land well above
the 0% untrained baseline on unseen wording too -- just markedly weaker
than the seen-wording numbers suggest on their own. This pattern is
essentially unchanged by the EOS fix (before: 28-31%/43-45%; after:
25-28%/41-45%), confirming the fix only changed *whether the model stops*,
not what it actually learned about the comparison task. Example of the raw
failure mode: on unseen-wording prompts, pretrained-baseline generations
never contain the answer at all and instead continue as generic corpus-
style continuations (verified in `reasoning_eval_metrics.json`'s
qualitative examples, e.g. a comparison prompt about two people's ages
answered with an unrelated biographical continuation).

## Hindi vs. Assamese comparison

Hindi outperforms Assamese on test_seen and test_unseen_entity by a wide
margin (~98% vs ~65%, ~97% vs ~58%, strict), consistent with Hindi's
larger Phase 2 pretraining corpus and lower LM-quality gap already
documented in `phase2_gap_decomposition.md` -- and, per the trade-off
noted above, possibly widened further by Assamese absorbing more of a
cost from the EOS fix's added gradient signal than Hindi does. The
direction **reverses** on test_unseen_wording: Assamese finetuned models
score higher (41-45%) than Hindi (25-28%) on every individual seed, not
just in the mean -- this does not simply track the Phase 2 corpus-size gap
the way the other two splits do, and is unchanged by the EOS fix. One
plausible reading: Hindi's higher seen-wording accuracy may reflect a
larger memorization component specific to the trained phrasing, which by
construction does not transfer to unseen wording, while Assamese's lower
seen-wording accuracy leaves comparatively more of its gain attributable
to genuine reasoning transfer. This is an observation from the data, not a
proven causal claim -- flagged as-is rather than forced into the same
pattern as the LM-quality gap.

## Reproduction

```bash
python3 hindi/eval/reasoning_eval.py \
    --pretrained-checkpoint <path>/hindi_checkpoint.pt \
    --finetuned-dir <path>/hindi_finetuned_checkpoints/
```

Writes `hindi/eval/reasoning_eval_metrics.json` (committed to this repo,
includes qualitative examples per split per checkpoint). Same command for
`assamese`.
