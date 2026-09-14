#!/usr/bin/env python3
"""Generation-quality evaluation, per spec section 2.3: greedy decoding
and temperatures 0.5, 1.0, 1.5 from held-out prefixes, scored against the
real continuation that follows each prefix in the held-out text (teacher-
forced reference, per spec's "(or teacher-forced targets)" option).

Prefix/generation split of the context budget: context_length=256, prefix
64 tokens, generation up to 192 tokens. This is a deliberate scaling of a
general 200-token repetition-loop diagnostic down to fit this project's
own context_length rather than retraining at a larger one (decided
2026-08-31); 192 is close to that reference figure, not an
arbitrary shrink.

50 held-out prefixes per language, evenly spread across the full test
split (same deterministic-spacing convention as evaluate_val_loss
elsewhere in this project), not sampled from just the start.

Run directly: python3 eval_generation.py --checkpoint-path /path/to/assamese_checkpoint.pt
"""

import argparse
import json
import sys
from pathlib import Path

import sacrebleu
import sentencepiece as spm
import torch
from rouge_score import rouge_scorer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "train"))

from model import DecoderLM  # noqa: E402
from train import load_checkpoint  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "model_config.json"
TOKENIZER_PATH = ROOT / "tokenizer" / "assamese_bpe_8000.model"
DEFAULT_DATA_DIR = ROOT / "data" / "splits"

NUM_PREFIXES = 50
PREFIX_LEN = 64
DECODING_SETTINGS = ["greedy", "temp_0.5", "temp_1.0", "temp_1.5"]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def tokenize_text(text_path, tokenizer_path):
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    ids = sp.encode(text, out_type=int)
    return torch.tensor(ids, dtype=torch.long), sp


def sample_prefix_windows(data, num_prefixes, window_len):
    """num_prefixes windows of window_len tokens, evenly spread across the
    full split rather than clustered at the start (same reasoning as
    evaluate_val_loss elsewhere: val/test.txt is built by concatenating
    sources in a fixed, unshuffled order, so the first windows would
    otherwise only ever sample whichever source came first)."""
    max_start = len(data) - window_len - 1
    total_windows = max(1, max_start // window_len)
    n = min(num_prefixes, total_windows)
    windows = []
    for i in range(n):
        step = (i * total_windows) // n
        s = step * window_len
        windows.append(data[s : s + window_len])
    return windows


@torch.no_grad()
def generate(model, prefix_ids, gen_len, context_length, device, temperature=None, seed=0):
    """Autoregressive generation from a fixed prefix. temperature=None means
    greedy (argmax); otherwise samples from softmax(logits / temperature).
    Seeded per call so temperature-based generation is reproducible across
    reruns, same determinism discipline as the rest of this project."""
    torch.manual_seed(seed)
    model.eval()
    ids = prefix_ids.clone().unsqueeze(0).to(device)
    for _ in range(gen_len):
        window = ids[:, -context_length:]
        logits = model(window)
        next_logits = logits[0, -1, :]
        if temperature is None:
            next_id = torch.argmax(next_logits).view(1, 1)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).view(1, 1)
        ids = torch.cat([ids, next_id.to(device)], dim=1)
    return ids[0, prefix_ids.shape[0]:].cpu()


def repetition_rate(token_ids, n=2):
    """Fraction of n-grams that are repeats of an earlier n-gram in the
    same sequence."""
    ids = token_ids.tolist()
    if len(ids) < n:
        return 0.0
    ngrams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    seen = set()
    repeated = 0
    for g in ngrams:
        if g in seen:
            repeated += 1
        seen.add(g)
    return repeated / len(ngrams)


def distinct_n(token_ids, n):
    """Unique n-grams over total n-grams, standard diversity diagnostic."""
    ids = token_ids.tolist()
    if len(ids) < n:
        return 0.0
    ngrams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def has_repetition_loop(token_ids, min_phrase_len=2, min_repeats=3):
    """A phrase of at least min_phrase_len tokens repeating at least
    min_repeats times at the end of the generation, checked within
    whatever was actually generated (scaled down from the 200-token
    reference window to this project's own gen_len, see module docstring).
    Flags True/False, does not attempt to locate every possible loop."""
    ids = token_ids.tolist()
    for phrase_len in range(min_phrase_len, len(ids) // min_repeats + 1):
        tail = ids[-phrase_len * min_repeats :]
        if len(tail) < phrase_len * min_repeats:
            continue
        phrase = tail[:phrase_len]
        if all(tail[i * phrase_len : (i + 1) * phrase_len] == phrase for i in range(min_repeats)):
            return True
    return False


def compute_quality_metrics(hypotheses, references):
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="intl")
    chrf = sacrebleu.corpus_chrf(hypotheses, references)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l_scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for hyp, ref in zip(hypotheses, references)]
    return {
        "bleu4": round(bleu.score, 4),
        "chrf": round(chrf.score, 4),
        "rouge_l_f1": round(sum(rouge_l_scores) / len(rouge_l_scores), 4),
    }


def run(checkpoint_path, data_dir=None):
    full_config = load_model_config()
    model_config = full_config["model"]
    context_length = model_config["context_length"]
    gen_len = context_length - PREFIX_LEN

    device = pick_device()
    print(f"device={device}")

    model = DecoderLM(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), weight_decay=0.1)
    step = load_checkpoint(checkpoint_path, model, optimizer, map_location=device)
    model.to(device)
    print(f"loaded checkpoint at step {step:,}")

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    data, sp = tokenize_text(data_dir / "test.txt", TOKENIZER_PATH)
    windows = sample_prefix_windows(data, NUM_PREFIXES, context_length)
    print(f"sampled {len(windows)} prefix windows from test.txt")

    prefixes = [w[:PREFIX_LEN] for w in windows]
    references_ids = [w[PREFIX_LEN:] for w in windows]
    references_text = [sp.decode(r.tolist()) for r in references_ids]

    results = {}
    sample_outputs = []
    for setting in DECODING_SETTINGS:
        temperature = None if setting == "greedy" else float(setting.split("_")[1])
        generations = [
            generate(model, prefix, gen_len, context_length, device, temperature=temperature, seed=i)
            for i, prefix in enumerate(prefixes)
        ]
        hypotheses_text = [sp.decode(g.tolist()) for g in generations]

        quality = compute_quality_metrics(hypotheses_text, references_text)
        rep_rates = [repetition_rate(g) for g in generations]
        dist1 = [distinct_n(g, 1) for g in generations]
        dist2 = [distinct_n(g, 2) for g in generations]
        loop_flags = [has_repetition_loop(g) for g in generations]

        results[setting] = {
            **quality,
            "repetition_rate": round(sum(rep_rates) / len(rep_rates), 4),
            "distinct_1": round(sum(dist1) / len(dist1), 4),
            "distinct_2": round(sum(dist2) / len(dist2), 4),
            "repetition_loop_fraction": round(sum(loop_flags) / len(loop_flags), 4),
        }
        print(f"{setting}: bleu4={quality['bleu4']:.2f} chrf={quality['chrf']:.2f} "
              f"rougeL={quality['rouge_l_f1']:.4f} rep_rate={results[setting]['repetition_rate']:.4f} "
              f"distinct1={results[setting]['distinct_1']:.4f} distinct2={results[setting]['distinct_2']:.4f} "
              f"loop_frac={results[setting]['repetition_loop_fraction']:.4f}")

        sample_outputs.append({
            "setting": setting,
            "prefix": sp.decode(prefixes[0].tolist()),
            "generated": hypotheses_text[0],
            "reference": references_text[0],
        })

    return {
        "language": "assamese", "checkpoint_step": step, "num_prefixes": len(windows),
        "prefix_len": PREFIX_LEN, "gen_len": gen_len, "results": results,
        "sample_outputs": sample_outputs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=str(ROOT / "eval" / "generation_metrics.json"))
    args = parser.parse_args()

    output = run(args.checkpoint_path, data_dir=args.data_dir)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
