#!/usr/bin/env python3
"""Phase 3 attention comparison, spec section 3.2: reuses the Phase 2
attention toolkit (attention_analysis.py's run_with_examples) completely
unchanged in its actual attention-computation logic, pointed at real
comparative-reasoning prompts instead of generic corpus text, and at both
a pretrained and a finetuned checkpoint -- so pretrained-vs-finetuned
heatmaps and entropy/distance metrics are directly comparable on the
actual task that changed, per spec's "especially on comparative-reasoning
prompts."

Sequence format matches finetune.py's training format exactly: <question
tokens> <s> <answer tokens> </s>. This lets the heatmap show attention
FROM the answer-generating position back onto the premise/entity tokens --
the most diagnostic view for whether finetuning changed how the model
attends across entities while reasoning, versus just showing attention
over the bare question with no answer context.

Two real test_seen examples picked for a genuine short-range vs long-range
contrast, not randomly sampled: one pairwise comparison (2 entities, 1
premise -- the answer only ever needs to look back at one nearby premise)
and one 3-entity chained comparison (3 entities, 2 premises -- the correct
entity is only inferable by combining both premises, a real long-range
dependency across the full prompt).

Each (checkpoint, example) pair gets its own labeled run, since
run_with_examples only plots a heatmap for the first example in a batch
by design (Phase 2's original intent: many generic sentences, one
representative heatmap) -- calling it once per example, rather than
extending that logic, keeps every actual attention/plotting function
untouched.

Run directly: python3 attention_reasoning_prompts.py --pretrained-checkpoint PATH
    --finetuned-checkpoint PATH
"""

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "train"))

from attention_analysis import run_with_examples  # noqa: E402

LANGUAGE = "assamese"
TOKENIZER_PATH = ROOT / "tokenizer" / f"{LANGUAGE}_bpe_8000.model"
REASONING_DIR = ROOT / "data" / "reasoning"

BOS_ID = 1  # <s>, must match finetune.py exactly
EOS_ID = 2  # </s>


def pick_reasoning_examples():
    with open(REASONING_DIR / "test_seen.jsonl", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    pairwise = next(e for e in examples if e["num_entities"] == 2 and not e["has_tie"])
    chained = next(e for e in examples if e["num_entities"] == 3 and not e["has_tie"])
    return {"pairwise": pairwise, "chained": chained}


def build_full_sequence_example(sp, example):
    """Matches finetune.py's exact training sequence format. Returns a
    (label_text, token_ids) tuple, the shape run_with_examples expects."""
    prompt_ids = sp.encode(example["prompt"], out_type=int)
    answer_ids = sp.encode(example["answer"], out_type=int)
    full_ids = prompt_ids + [BOS_ID] + answer_ids + [EOS_ID]
    label = f'{example["prompt"]} <s> {example["answer"]} </s>'
    return label, full_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--finetuned-checkpoint", required=True)
    args = parser.parse_args()
    pretrained_ckpt = Path(args.pretrained_checkpoint)
    finetuned_ckpt = Path(args.finetuned_checkpoint)

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))

    reasoning_examples = pick_reasoning_examples()
    assert pretrained_ckpt.exists(), f"pretrained checkpoint not found at {pretrained_ckpt}"
    assert finetuned_ckpt.exists(), f"finetuned checkpoint not found at {finetuned_ckpt}"

    results = {}
    for checkpoint_name, checkpoint_path in [
        ("pretrained", pretrained_ckpt), ("finetuned_seed0", finetuned_ckpt),
    ]:
        results[checkpoint_name] = {}
        for kind, example in reasoning_examples.items():
            label, ids = build_full_sequence_example(sp, example)
            print(f"--- {checkpoint_name}, {kind} (domain={example['domain']}, "
                  f"entities={example['num_entities']}, tokens={len(ids)}) ---")
            print(f"  {label}")
            result = run_with_examples(
                checkpoint_path, sp, [(label, ids)], checkpoint_label=f"{checkpoint_name}_{kind}"
            )
            results[checkpoint_name][kind] = result

    out_path = Path(__file__).resolve().parent / "attention_reasoning_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
