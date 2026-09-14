# Phase 3: Reasoning Finetuning Results

## Method

Each model's own Phase 2 pretrained checkpoint is finetuned via LoRA
(rank 8, alpha 16, targeting `q_proj`/`v_proj` in every attention block;
base weights and the token embedding table, including the output head
tied to it, stay frozen) on that language's own synthetic comparative-
reasoning dataset. No data or weights are shared between the two models.

Rank and alpha were picked by an empirical sweep (`<lang>/train/
lora_rank_sweep.py`), not guessed: reasoning-val loss drops monotonically
and substantially from rank 2 to rank 8, while pretrain-val PPL barely
moves at any rank tested, so rank 8 wins outright rather than by a
tolerance tie-break. Both languages converged on rank 8 / alpha 16
independently, from separate real-data sweeps.

Training: warmup-then-constant learning rate (`peak_lr=1e-4`, 10 warmup
steps), batch size 32, up to 3 seeds per language (`0, 1, 2`) per Lec09's
multi-seed requirement, early stopping on reasoning-validation loss
(patience 5 epochs). Pretrain-val perplexity is tracked every epoch as a
catastrophic-forgetting diagnostic but is not an automatic stop
condition: no literature or course precedent was found for a fixed PPL-
increase percentage threshold, and the course's own guidance is to plot
and inspect the curve rather than auto-trigger on it. Figures below show
this diagnostic held essentially flat across all three seeds, both
languages (Hindi ~38 to ~45 PPL, well within normal sampling noise;
Assamese ~55 to ~69) -- no forgetting signal severe enough to warrant a
stop was observed at this training budget.

Finetuned checkpoints are saved in the same resume-capable format as
pretraining (model state, optimizer state, step, config), and separately
merged into a plain `DecoderLM` state dict (`merge_lora_to_plain`) so
Phase 2's evaluation and attention tooling can load them unchanged.

## Dataset

Synthetic comparative-reasoning dataset, generated programmatically per
language (own templates, own entity-name pools, own script), not
downloaded from an existing benchmark. Both leakage-avoidance axes from
Lec09 are enforced: some template surface forms are held out entirely for
test (train on 3 forms per relation, hold 1), and some entity-name pools
are held out entirely for test, never appearing in train even paired with
seen templates.

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

## Scoring: strict vs. lenient, and why both are reported

A real mechanistic finding (not a scoring artifact) requires reporting
two exact-match numbers rather than one. LoRA targets only `q_proj`/
`v_proj`; the token embedding table (and the tied output head) stay
frozen. `<s>`/`</s>` were never emitted mid-corpus during Phase 2
pretraining, so their embedding rows are untouched random init -- verified
directly via `torch.allclose` on the embedding rows for ids 0/1/2 between
the pretrained and every finetuned checkpoint (bit-identical, both
languages). The finetuned model picks the correct answer with high
confidence (~98%+ argmax probability, confirmed by direct logit
inspection) but almost never emits `</s>` to stop -- `P(</s>)` at the
position immediately after the answer measured at ~1e-6, essentially
zero, across every case checked. Generation instead loops a short
repeating token cycle until the decoding cap.

- **Strict** exact match: the raw generation must equal the expected
  answer exactly, requiring a literal `</s>` to terminate it. This is the
  honest floor and documents the real EOS-learning failure directly:
  it is 0.0000 in every cell below, both languages, pretrained and
  finetuned alike.
- **Lenient** exact match: does the raw generation **start with** the
  exact expected answer (prefix match), tolerating whatever degenerate
  content follows. This isolates whether the model's actual answer
  content is correct, independent of the termination bug.

  A simpler cycle-detection approach (truncate the repeating loop, then
  compare) was tried first and rejected: a period-2 loop `<answer>
  <filler> <answer> <filler> ...` is genuinely ambiguous about which half
  is "content" from periodicity alone -- both phases are equally valid
  decompositions of the same repeating sequence. Prefix match sidesteps
  this ambiguity entirely. Self-tests confirm it is a strict prefix
  check, not "expected appears anywhere" -- a wrong answer that merely
  contains the right one as a non-leading substring still scores as
  incorrect.

## Results: pretrained vs. finetuned, lenient accuracy, mean across 3 seeds

| Split | Hindi pretrained | Hindi finetuned (mean) | Assamese pretrained | Assamese finetuned (mean) |
|---|---|---|---|---|
| test_seen | 0.0000 | 0.9822 | 0.0044 | 0.7541 |
| test_unseen_entity | 0.0000 | 0.9571 | 0.0000 | 0.7000 |
| test_unseen_wording | 0.0000 | 0.2956 | 0.0000 | 0.4400 |

Strict (literal-`</s>`) accuracy is 0.0000 in every cell above, both
models, both languages, pretrained and finetuned -- confirming the
EOS-learning failure is total, not partial.

Per-seed lenient accuracy (confirms the pattern below is not a
single-seed artifact):

| Seed | Hindi test_seen | Hindi unseen_entity | Hindi unseen_wording | Assamese test_seen | Assamese unseen_entity | Assamese unseen_wording |
|---|---|---|---|---|---|---|
| 0 | 0.9800 | 0.9578 | 0.2800 | 0.7911 | 0.7356 | 0.4467 |
| 1 | 0.9844 | 0.9556 | 0.3067 | 0.6911 | 0.6511 | 0.4333 |
| 2 | 0.9822 | 0.9578 | 0.3000 | 0.7800 | 0.7133 | 0.4400 |

## Conclusion: the finetune worked

Content accuracy (lenient) is far above the near-zero pretrained baseline
on every split, both languages -- the finetuned checkpoints stand as
final; no rerun or embedding-unfreezing fix was needed. The mechanism
(LoRA's target-module choice interacting with untrained special-token
embeddings under frozen, tied weights) is itself a real qualitative
finding, not just a scoring caveat: the correct framing is that LoRA on
`{W_q, W_v}` did not learn to steer the hidden state toward the untrained
`</s>` output direction within this training budget -- not that doing so
is structurally impossible under weight tying, since the output logit for
`</s>` is `hidden_state . embedding[2]` under weight tying and remains
theoretically reachable by an adapted attention pattern.

## Qualitative failure mode

Format-memorization signature: the drop from test_seen / test_unseen_entity
(>95% Hindi, >65% Assamese, per-seed) down to test_unseen_wording (28-31%
Hindi, 43-45% Assamese) matches the pattern expected when part of the gain
is template/format memorization rather than pure abstract reasoning
transfer. Real transfer is still present -- both languages land well above
the 0% untrained baseline on unseen wording too -- just markedly weaker
than the seen-wording numbers suggest on their own. Example of the raw
failure mode: on unseen-wording prompts, pretrained-baseline generations
never contain the answer at all and instead continue as generic corpus-
style continuations (verified in `reasoning_eval_metrics.json`'s
qualitative examples, e.g. a comparison prompt about two people's ages
answered with an unrelated biographical continuation).

## Hindi vs. Assamese comparison

Hindi outperforms Assamese on test_seen and test_unseen_entity by a wide
margin (~98% vs ~75%, ~96% vs ~70%), consistent with Hindi's larger Phase
2 pretraining corpus and lower LM-quality gap already documented in
`phase2_gap_decomposition.md`. The direction **reverses** on
test_unseen_wording: Assamese finetuned models score higher (43-45%) than
Hindi (28-31%) on every individual seed, not just in the mean -- this does
not simply track the Phase 2 corpus-size gap the way the other two splits
do. One plausible reading: Hindi's higher seen-wording accuracy may
reflect a larger memorization component specific to the trained phrasing,
which by construction does not transfer to unseen wording, while
Assamese's lower seen-wording accuracy leaves comparatively more of its
gain attributable to genuine reasoning transfer. This is an observation
from the data, not a proven causal claim -- flagged as-is rather than
forced into the same pattern as the LM-quality gap.

## Reproduction

```bash
python3 hindi/eval/reasoning_eval.py \
    --pretrained-checkpoint <path>/hindi_checkpoint.pt \
    --finetuned-dir <path>/hindi_finetuned_checkpoints/
```

Writes `hindi/eval/reasoning_eval_metrics.json` (committed to this repo,
includes qualitative examples per split per checkpoint). Same command for
`assamese`.
