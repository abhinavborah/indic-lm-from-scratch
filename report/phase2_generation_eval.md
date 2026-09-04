# Phase 2: Generation Quality Evaluation

Continuations generated from 50 held-out prefixes per language (test
split, evenly spread across the full file, never used for training),
under greedy decoding and temperatures 0.5, 1.0, 1.5, per spec section
2.3. Context budget: 64 prefix tokens, up to 192 generated tokens (see
`report/phase2_architecture.md`'s context_length discussion for why this
split, not a retrain at a larger context_length). Scored against the real
continuation that follows each prefix in the held-out text (teacher-forced
reference).

## Results

### Hindi (Model H)

| Setting | BLEU-4 | chrF | ROUGE-L (F1) | Repetition rate | Distinct-1 | Distinct-2 | Repetition-loop fraction |
|---|---|---|---|---|---|---|---|
| Greedy | 1.64 | 0.90 | 0.0154 | 0.8838 | 0.1006 | 0.1162 | 0.9800 |
| Temp 0.5 | 2.60 | 0.92 | 0.0133 | 0.6680 | 0.2211 | 0.3320 | 0.3200 |
| Temp 1.0 | 2.10 | 0.90 | 0.0110 | 0.0492 | 0.7424 | 0.9508 | 0.0000 |
| Temp 1.5 | 0.09 | 0.83 | 0.0095 | 0.0009 | 0.9398 | 0.9991 | 0.0000 |

### Assamese (Model L)

| Setting | BLEU-4 | chrF | ROUGE-L (F1) | Repetition rate | Distinct-1 | Distinct-2 | Repetition-loop fraction |
|---|---|---|---|---|---|---|---|
| Greedy | 0.27 | 0.83 | 0.0047 | 0.8982 | 0.0874 | 0.1018 | 0.9600 |
| Temp 0.5 | 0.58 | 0.86 | 0.0064 | 0.5931 | 0.2871 | 0.4069 | 0.2800 |
| Temp 1.0 | 0.46 | 0.81 | 0.0063 | 0.0226 | 0.8454 | 0.9774 | 0.0000 |
| Temp 1.5 | 0.10 | 0.75 | 0.0029 | 0.0015 | 0.9531 | 0.9985 | 0.0000 |

Assamese's quality metrics are consistently lower than Hindi's at every
decoding setting, the same direction as the PPL/BPB gap
(`report/phase2_lm_eval.md`), consistent with Assamese's smaller real
training corpus.

## Sample outputs (first prefix, both languages)

**Hindi**, prefix: "चित्र. 4.3— चंुबक द्वारा आकर्षित होने वाली वस्‍तुअों का पता लगाना आप भूमि पर गिरे हुए अलग-अलग प्र..."

- Greedy: "चंुबक द्वारा प्रदर्शित होने वाली व्षिकतुअों का पता लगाना ·······················" (degenerates into repeated punctuation)
- Temp 1.0: "गांठ, जो पत्तों या फूल को एकत्रित करके पकाती है पक्षियों को दूसरी जगह संरक्षित करना पडे़गा..." (fluent, grammatical Hindi, topic drifts from the source, expected at 25M parameters)
- Temp 1.5: "गां अफ्रीक जो अंगों द्वारा चिप forपुरी उत्पन्न करते हैं वह कहना पक्ष स्थिति अलग..." (incoherent, includes a stray Latin-script token)

**Assamese**, greedy sample: "হ্ৰদ। এই হ্ৰদটো উত্তৰ-পূব ভাৰতৰ উত্তৰ-পূব সীমান্তৱৰ্তী অঞ্চল। এই হ্ৰদটো উত্তৰ-পূ..." (phrase-level repetition of "উত্তৰ-পূব", a different degeneration signature than Hindi's punctuation loop, same underlying failure class)

## Why these metrics are or are not informative here

**BLEU** is a poor fit for open-ended generation and the numbers here
show why directly: scores are low (0.09-2.60 out of 100) across every
setting and both languages, including at temp 1.0, where the qualitative
sample above is fluent, grammatical Hindi. BLEU penalizes any deviation
from the single reference continuation's exact wording, and open-ended
generation from a held-out prefix has no single correct continuation,
many fluent completions exist that share no n-grams with the one real
continuation that happened to follow in the corpus. A low BLEU score
here does not mean the generation is bad; it means BLEU is measuring the
wrong thing for this task.

**chrF** is more informative for this project specifically because it
operates on character n-grams rather than word n-grams, so it is
insensitive to the exact tokenizer and to word-boundary mismatches
between morphologically related forms, a real concern for two
morphologically rich, non-whitespace-trivial Indic scripts with different
fertility (Hindi 1.4940, Assamese 1.7780 tokens/word). chrF stays in a
narrower, more stable 0.75-0.92 range across settings and both languages,
tracking coherence direction more sensibly than BLEU does (falls
noticeably at temp 1.5, where the qualitative samples degrade).

**ROUGE-L** (longest common subsequence) sits in between: like BLEU it
requires matching against the one available reference, so it shares
BLEU's open-ended-generation weakness, but because it rewards partial,
non-contiguous overlap rather than exact n-gram matches, it is somewhat
less brittle. Scores are uniformly low here (0.003-0.015) for the same
reason BLEU's are: one reference continuation is a thin target for a
free-generation task.

**Repetition rate, Distinct-1/2, and the repetition-loop fraction** are
the most informative diagnostics for this evaluation, because they
measure something the metrics above cannot: whether the model produces
degenerate, repetitive text at all, independent of whether it matches
any particular reference. The pattern across both languages is the same
and matches well-documented neural text degeneration: greedy decoding
collapses into near-total repetition (88-90% repeated bigrams, 96-98% of
generations hit a detected repetition loop), temperature 1.0 is a clear
sweet spot (repetition rate under 5%, Distinct-2 above 0.95, zero
detected loops), and temperature 1.5 pushes diversity further but at the
cost of coherence collapsing (visible directly in the temp 1.5 sample
above, and in BLEU's near-zero score at that setting for both languages).
This is consistent with why unlikelihood training and similar
degeneration-mitigation techniques exist in the literature (considered
for this project, not adopted, see `CONTEXT.md`'s 2026-08-31 decision):
greedy decoding on a model this size reliably produces the failure mode
those techniques target.
