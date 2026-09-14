#!/usr/bin/env python3
"""Pretrained-vs-finetuned reasoning accuracy for the Hindi model, per spec
deliverable 3.1: "pretrained vs. finetuned accuracy (or exact match) on the
synthetic test set." Evaluated separately on the 3 mandatory Lec09 splits
(test_seen, test_unseen_entity, test_unseen_wording) -- per
docs-phase-3/finetune_protocol.md and dataset_leakage_methodology.md, these
are NEVER aggregated together: a gain that shows up on test_seen but not
test_unseen_wording is format-memorization, not a reasoning gain, and
averaging the three would hide exactly that distinction.

Greedy decoding only (no temperature sampling): the task needs one
canonical answer per prompt for exact-match scoring, not generation
diversity -- unlike eval_generation.py's separate quality-metrics eval.

Answer generation: encode `<question tokens> <s>`, greedy-decode until
`</s>` or MAX_ANSWER_TOKENS (24, comfortably above the real max of 4 tokens
measured on test_seen -- the cap exists only to bound a degenerate
non-stopping generation, a known documented risk of greedy decoding on
this model, not because real answers approach it).

Two exact-match scores are reported, never conflated -- see
docs-phase-3/lora_eos_untrained_finding.md for why both exist. LoRA
targets only q_proj/v_proj; the token embedding table (and the output
head, tied to it) stays frozen, and `<s>`/`</s>` were never trained during
pretraining (never emitted mid-corpus). Confirmed by direct inspection:
the finetuned model chooses the right answer with high confidence but
almost never emits `</s>` afterward, instead looping a short repeating
token cycle until the generation cap.
- `strict` exact match: raw generation, stopped only at a literal `</s>`
  -- the honest floor, reflects this real decoding failure.
- `lenient` exact match: does the raw generation START WITH the exact
  expected answer (prefix match), tolerating whatever degenerate content
  follows. A cycle-detection heuristic (truncate the loop, then compare)
  was tried first and rejected: a period-2 loop `<answer> <filler>
  <answer> <filler> ...` is genuinely ambiguous about which half is
  "content" from periodicity alone, so it kept the filler attached and
  never actually recovered a clean match. Prefix match sidesteps that
  ambiguity entirely and is the simpler rule.

Run directly: python3 reasoning_eval.py --pretrained-checkpoint PATH
    --finetuned-dir DIR [--out PATH]
"""

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "train"))
from model import DecoderLM  # noqa: E402
from train import load_checkpoint  # noqa: E402

LANGUAGE = "hindi"
CONFIG_PATH = ROOT / "configs" / "model_config.json"
TOKENIZER_PATH = ROOT / "tokenizer" / f"{LANGUAGE}_bpe_8000.model"
REASONING_DIR = ROOT / "data" / "reasoning"

BOS_ID = 1  # <s>, prompt/answer boundary -- must match finetune.py exactly
EOS_ID = 2  # </s>, terminal end of sequence
MAX_ANSWER_TOKENS = 24
SPLITS = ["test_seen", "test_unseen_entity", "test_unseen_wording"]
NUM_QUALITATIVE_EXAMPLES = 8  # per split per model, mix of correct/incorrect


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_plain_model(checkpoint_path, model_config, device):
    """Loads a plain (non-LoRA) DecoderLM checkpoint -- works identically for
    a Phase 2 pretrained checkpoint and a merge_lora_to_plain-exported
    finetuned checkpoint, since both share the exact same state dict shape."""
    model = DecoderLM(model_config)
    optimizer = torch.optim.AdamW(model.parameters())
    step = load_checkpoint(checkpoint_path, model, optimizer, map_location=device)
    model.to(device)
    model.eval()
    return model, step


@torch.no_grad()
def generate_answer_ids(model, sp, prompt, context_length, device, max_answer_tokens=MAX_ANSWER_TOKENS):
    """Raw greedy generation, stopped only at a literal </s> or the token
    cap. Returns the generated token id list (no decoding) -- callers
    decode under whichever stopping rule (strict or lenient) they need."""
    prompt_ids = sp.encode(prompt, out_type=int)
    ids = torch.tensor(prompt_ids + [BOS_ID], dtype=torch.long, device=device).unsqueeze(0)
    generated = []
    for _ in range(max_answer_tokens):
        window = ids[:, -context_length:]
        logits = model(window)
        next_id = torch.argmax(logits[0, -1, :]).item()
        if next_id == EOS_ID:
            break
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    return generated


def generate_answer(model, sp, prompt, context_length, device, max_answer_tokens=MAX_ANSWER_TOKENS):
    """Strict decoding: sp.decode of the raw generation, unmodified."""
    ids = generate_answer_ids(model, sp, prompt, context_length, device, max_answer_tokens)
    return sp.decode(ids).strip()


def evaluate_split(model, sp, examples, context_length, device, generate_ids_fn=generate_answer_ids):
    strict_correct = 0
    lenient_correct = 0
    qualitative = []
    for ex in examples:
        raw_ids = generate_ids_fn(model, sp, ex["prompt"], context_length, device)
        expected = ex["answer"].strip()
        predicted = sp.decode(raw_ids).strip()
        strict_is_correct = predicted == expected
        lenient_is_correct = predicted.startswith(expected)
        strict_correct += strict_is_correct
        lenient_correct += lenient_is_correct
        if len(qualitative) < NUM_QUALITATIVE_EXAMPLES:
            qualitative.append({
                "prompt": ex["prompt"], "expected": expected, "predicted": predicted,
                "strict_correct": strict_is_correct, "lenient_correct": lenient_is_correct,
            })
    n = len(examples)
    return strict_correct / n, lenient_correct / n, n, qualitative


def load_splits():
    splits = {}
    for name in SPLITS:
        with open(REASONING_DIR / f"{name}.jsonl", encoding="utf-8") as f:
            splits[name] = [json.loads(line) for line in f]
    return splits


def evaluate_checkpoint(checkpoint_path, model_config, sp, splits, device):
    model, step = load_plain_model(checkpoint_path, model_config, device)
    result = {"checkpoint": str(checkpoint_path), "step": step, "splits": {}}
    for name, examples in splits.items():
        strict_acc, lenient_acc, n, qualitative = evaluate_split(
            model, sp, examples, model_config["context_length"], device
        )
        result["splits"][name] = {
            "strict_accuracy": round(strict_acc, 4), "lenient_accuracy": round(lenient_acc, 4),
            "n": n, "qualitative_examples": qualitative,
        }
        print(f"    {name}: strict={strict_acc:.4f} lenient={lenient_acc:.4f} (n={n})")
    del model
    return result


def discover_finetuned_checkpoints(finetuned_dir):
    return sorted(Path(finetuned_dir).glob(f"{LANGUAGE}_finetuned_seed*_merged.pt"))


def main():
    parser = argparse.ArgumentParser(description="Pretrained vs finetuned reasoning accuracy, hindi.")
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--finetuned-dir", required=True)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "reasoning_eval_metrics.json"))
    args = parser.parse_args()

    with open(CONFIG_PATH, encoding="utf-8") as f:
        model_config = json.load(f)["model"]

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))

    splits = load_splits()
    print(f"splits: " + ", ".join(f"{name}={len(exs)}" for name, exs in splits.items()))

    device = pick_device()
    print(f"device: {device}")

    pretrained_path = Path(args.pretrained_checkpoint)
    assert pretrained_path.exists(), f"pretrained checkpoint not found at {pretrained_path}"
    print("--- pretrained (no finetune) ---")
    pretrained_result = evaluate_checkpoint(pretrained_path, model_config, sp, splits, device)

    finetuned_paths = discover_finetuned_checkpoints(args.finetuned_dir)
    assert finetuned_paths, f"no finetuned merged checkpoints found under {args.finetuned_dir}"
    finetuned_results = {}
    for path in finetuned_paths:
        seed_label = path.stem.split("_seed")[1].split("_merged")[0]
        print(f"--- finetuned seed {seed_label} ---")
        finetuned_results[f"seed{seed_label}"] = evaluate_checkpoint(path, model_config, sp, splits, device)

    mean_accuracy_per_split = {}
    for name in SPLITS:
        strict_accs = [r["splits"][name]["strict_accuracy"] for r in finetuned_results.values()]
        lenient_accs = [r["splits"][name]["lenient_accuracy"] for r in finetuned_results.values()]
        mean_accuracy_per_split[name] = {
            "strict": round(sum(strict_accs) / len(strict_accs), 4),
            "lenient": round(sum(lenient_accs) / len(lenient_accs), 4),
        }

    print("\n--- summary (mean across finetuned seeds, split separately, never aggregated) ---")
    for name in SPLITS:
        p = pretrained_result["splits"][name]
        m = mean_accuracy_per_split[name]
        print(f"{name}: pretrained(strict={p['strict_accuracy']:.4f} lenient={p['lenient_accuracy']:.4f}) "
              f"finetuned_mean(strict={m['strict']:.4f} lenient={m['lenient']:.4f})")

    output = {
        "language": LANGUAGE,
        "pretrained": pretrained_result,
        "finetuned": finetuned_results,
        "finetuned_mean_accuracy_per_split": mean_accuracy_per_split,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
