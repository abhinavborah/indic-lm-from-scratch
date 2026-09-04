# Phase 2: Resource-Level Comparison

Both models were trained locally on an Apple M4 Pro (MPS backend), not
Colab. The project started on Colab, but a hindi run hit Colab's
unpublished free-tier daily resource cap mid-training (step ~26,200 of a
planned run), and the venue was switched to local training for the
remainder of Phase 2 rather than continue fighting an undocumented quota.
Both real training runs reported here (hindi's final run, assamese's only
run) are local, same hardware, same architecture, so the comparison below
isolates data-scale and language differences rather than hardware
differences.

## Training cost

| | Hindi (Model H) | Assamese (Model L) |
|---|---|---|
| Real train tokens | 834,049,861 | 565,255,443 |
| Steps (one epoch, batch=32, context=256) | 101,812 | 69,000 |
| True active training time | 17.28 hours | 11.92 hours |
| Throughput (measured) | ~14,185-14,916 tokens/sec | ~13,673-13,756 tokens/sec |
| Parameter count | 24,366,336 | 24,366,336 (identical) |

True training time is the sum of active compute across both segments of
each run (each was interrupted once and resumed; the reported figure
sums both segments' own elapsed counters, a lower bound on real calendar
time since it excludes whatever gap existed while each run sat paused).
This correction matters: an earlier report of ~9.3h/2.9h for hindi and
assamese reflected only the post-resume segment of each run, not the
true total, caught and corrected before it reached this report.

Assamese trained in about 69% of hindi's time, roughly proportional to
its smaller real corpus (565M vs 834M tokens, a ratio of 0.68), not a
throughput difference: measured tokens/sec is close between the two
languages (~13.7-14.9K either way), consistent with both models sharing
the exact same architecture and parameter count. The training-time gap
is a direct consequence of Assamese's smaller collected corpus, not of
the model being any cheaper to run per token.

## Evaluation cost

Both PPL/BPB evaluation (full held-out val and test splits, no sampling)
and generation-quality evaluation (50 held-out prefixes, 4 decoding
settings each) are forward-only, no gradients, and run in minutes rather
than hours: intrinsic LM evaluation covers 8.36-8.63M tokens for Hindi
and 5.63-5.86M tokens for Assamese (val and test combined range, smaller
for Assamese for the same corpus-size reason as everything else in this
document), each in well under ten minutes; generation evaluation produces
400 total generations (50 prefixes x 4 decoding settings x 2 languages)
in a comparable timeframe. Evaluation cost is negligible next to training
cost for both languages, and identical in structure between them, since
the same scripts run unmodified for both.

## What resource cost actually tracks here

Given identical architecture, identical parameter count, and comparable
per-token throughput, the entire resource-cost gap between Model H and
Model L traces back to one thing: how much real training data each
language's corpus collection produced. This is consistent with the
project's data-collection framing from Phase 1: Assamese was the
deliberately lower-resource language, and its smaller real corpus
(565M vs 834M tokens, about 68% of Hindi's) is a direct, expected
consequence of that choice, not an artifact of the training or
evaluation pipeline treating the two languages differently.
