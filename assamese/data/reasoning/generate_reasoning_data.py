#!/usr/bin/env python3
"""Synthetic comparative/transitive reasoning dataset generator for Assamese.

Generates Q-then-A examples over four attribute domains (height, age, price,
quantity), each a comparison over 2-4 entities: direct pairwise comparisons
and 2-hop/3-hop transitive chains (X's height is more than Y's, Y's is more
than Z's, who has the least?). No numbers ever appear in the rendered text;
values exist only internally to fix a strict ordering.

Independent of hindi/data/reasoning/generate_reasoning_data.py: same
generation logic and split methodology, but its own entity pools and
Assamese-grammar templates, per this project's fully-independent-languages
rule. Do not import across the hindi/assamese boundary.

Deliberate scope cut: the "equal" comparison outcome (spec mentions greater,
smaller, equal as illustrative) is not generated in this first build. Spec's
own emphasis is "especially transitive/ordering comparisons," which this
covers fully; equal-value examples add template/answer-format complexity
(a third answer class) for comparatively little coverage gain. Revisit only
if time permits after the core three-way split works.

Five output splits, per docs-phase-3/dataset_leakage_methodology.md:
  train, val                 seen entities, seen (trained) wording
  test_seen                  seen entities, seen wording (held-out instances)
  test_unseen_entity         held-out entity pool, seen (trained) wording
  test_unseen_wording        seen entity pool, held-out (4th) wording form

Run directly to regenerate all five files:
    python3 generate_reasoning_data.py --out-dir .
"""

import argparse
import json
import random
from pathlib import Path

from entities import POOLS
from templates import DOMAINS, HELD_FORM_ID, TRAIN_FORM_IDS, render_premise, render_question

MASTER_SEED = 20260913

# (num_entities: weight) for the mix of pairwise / 2-hop / 3-hop chains.
CHAIN_LENGTH_WEIGHTS = {2: 0.45, 3: 0.40, 4: 0.15}

SPLIT_TARGETS = {
    "train": 4500,
    "val": 450,
    "test_seen": 450,
    "test_unseen_entity": 450,
    "test_unseen_wording": 150,
}


def _sample_chain(rng, pool, num_entities):
    """Pick num_entities distinct names and a strict descending order for them.
    Returns names ordered largest-to-smallest; only order matters, not values."""
    names = rng.sample(pool, num_entities)
    values = rng.sample(range(0, 100_000), num_entities)
    order = sorted(zip(names, values), key=lambda pair: -pair[1])
    return [name for name, _ in order]


def _make_example(rng, domain_name, pool, form_ids, num_entities):
    attr_noun = DOMAINS[domain_name]["attr_noun"]
    ordered = _sample_chain(rng, pool, num_entities)

    premise_form = rng.choice(form_ids)
    question_form = rng.choice(form_ids)

    premise_sentences = [
        render_premise(premise_form, ordered[i], ordered[i + 1], attr_noun)
        for i in range(num_entities - 1)
    ]
    rng.shuffle(premise_sentences)

    direction = rng.choice(["largest", "smallest"])
    answer = ordered[0] if direction == "largest" else ordered[-1]

    question_entities = list(ordered)
    rng.shuffle(question_entities)
    question = render_question(question_form, question_entities, attr_noun, direction)

    prompt = " ".join(premise_sentences) + " " + question
    return {
        "domain": domain_name,
        "num_entities": num_entities,
        "entities": ordered,
        "premise_form_id": premise_form,
        "question_form_id": question_form,
        "direction": direction,
        "prompt": prompt,
        "answer": answer,
    }


def _dedup_key(example):
    """Keys on the semantic chain only (domain, entities, direction), not on
    which template form rendered it. The same chain phrased with a different
    form is still the same reasoning problem, and must not be allowed to
    appear in both train and test_seen under different wording (that would
    silently leak the answer via memorized entity/order, not test anything)."""
    return (
        example["domain"],
        tuple(example["entities"]),
        example["direction"],
    )


def _weighted_num_entities(rng):
    return rng.choices(
        list(CHAIN_LENGTH_WEIGHTS.keys()),
        weights=list(CHAIN_LENGTH_WEIGHTS.values()),
    )[0]


def generate_all(seed=MASTER_SEED, split_targets=None):
    split_targets = split_targets or SPLIT_TARGETS
    rng = random.Random(seed)
    domain_names = list(DOMAINS.keys())

    examples = {split: [] for split in split_targets}

    # seen-entity, seen-wording pool: shared source for train/val/test_seen,
    # partitioned below so no instance crosses between them.
    seen_pool_target = split_targets["train"] + split_targets["val"] + split_targets["test_seen"]
    seen_candidates = {}
    while len(seen_candidates) < seen_pool_target:
        domain_name = rng.choice(domain_names)
        pool = POOLS[DOMAINS[domain_name]["entity_pool"]]["train"]
        num_entities = _weighted_num_entities(rng)
        example = _make_example(rng, domain_name, pool, TRAIN_FORM_IDS, num_entities)
        seen_candidates[_dedup_key(example)] = example
    seen_pool = list(seen_candidates.values())
    rng.shuffle(seen_pool)
    examples["train"] = seen_pool[: split_targets["train"]]
    examples["val"] = seen_pool[split_targets["train"] : split_targets["train"] + split_targets["val"]]
    examples["test_seen"] = seen_pool[split_targets["train"] + split_targets["val"] :]

    unseen_entity_candidates = {}
    while len(unseen_entity_candidates) < split_targets["test_unseen_entity"]:
        domain_name = rng.choice(domain_names)
        pool = POOLS[DOMAINS[domain_name]["entity_pool"]]["held"]
        num_entities = _weighted_num_entities(rng)
        example = _make_example(rng, domain_name, pool, TRAIN_FORM_IDS, num_entities)
        unseen_entity_candidates[_dedup_key(example)] = example
    examples["test_unseen_entity"] = list(unseen_entity_candidates.values())[
        : split_targets["test_unseen_entity"]
    ]

    unseen_wording_candidates = {}
    while len(unseen_wording_candidates) < split_targets["test_unseen_wording"]:
        domain_name = rng.choice(domain_names)
        pool = POOLS[DOMAINS[domain_name]["entity_pool"]]["train"]
        num_entities = _weighted_num_entities(rng)
        example = _make_example(rng, domain_name, pool, (HELD_FORM_ID,), num_entities)
        unseen_wording_candidates[_dedup_key(example)] = example
    examples["test_unseen_wording"] = list(unseen_wording_candidates.values())[
        : split_targets["test_unseen_wording"]
    ]

    return examples


def write_splits(examples, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in examples.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} examples -> {path}")


def _template_variety_stats(examples):
    all_rows = [row for rows in examples.values() for row in rows]
    forms_used = {(row["premise_form_id"], row["question_form_id"]) for row in all_rows}
    domains_used = {row["domain"] for row in all_rows}
    depths_used = {row["num_entities"] - 1 for row in all_rows}
    return {
        "total_examples": len(all_rows),
        "distinct_form_pairs": len(forms_used),
        "domains_used": sorted(domains_used),
        "chain_depths_used": sorted(depths_used),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    args = parser.parse_args()

    all_examples = generate_all(seed=args.seed)
    write_splits(all_examples, args.out_dir)
    print(_template_variety_stats(all_examples))
