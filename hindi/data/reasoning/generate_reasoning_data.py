#!/usr/bin/env python3
"""Synthetic comparative/transitive reasoning dataset generator for Hindi.

Generates Q-then-A examples over four attribute domains (height, age, price,
quantity), each a comparison over 2-4 entities: direct pairwise comparisons
and 2-hop/3-hop transitive chains ("A's height is more than B's, B's is more
than C's, who has the least?"). No numbers ever appear in the rendered text;
values exist only internally to fix a strict ordering.

Includes a minority share of ties (equal values), per spec's "comparison
questions (greater, smaller, equal)" and user direction 2026-09-14: pairwise
(2-entity) ties and interior mid-chain ties (3-hop/4-entity only), targeting
roughly 10% of the dataset overall. 2-hop (3-entity) chains structurally
cannot hold a tie without landing it on the asked-about extremum (with 3
entities, every adjacent pair touches position 0 or position 2), which
violates the never-ambiguous-answer rule -- so ties are absent from 3-entity
examples by construction, not an oversight.

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
from templates import (
    DOMAINS, EQUAL_WORD, HELD_FORM_ID, TRAIN_FORM_IDS, render_equal_premise, render_premise,
    render_question,
)

MASTER_SEED = 20260913

# (num_entities: weight) for the mix of pairwise / 2-hop / 3-hop chains.
CHAIN_LENGTH_WEIGHTS = {2: 0.45, 3: 0.40, 4: 0.15}

# Only num_entities in {2, 4} can carry a tie without ambiguity (see module
# docstring). Their combined population share is 0.45 + 0.15 = 0.60; to land
# the overall dataset's tie share near the targeted ~10%, the per-eligible-
# example tie probability is 0.10 / 0.60 = 1/6.
TIE_PROBABILITY = 1 / 6

SPLIT_TARGETS = {
    "train": 4500,
    "val": 450,
    "test_seen": 450,
    "test_unseen_entity": 450,
    "test_unseen_wording": 150,
}


def _sample_chain(rng, pool, num_entities):
    """Pick num_entities distinct names and a strict descending order for them.
    Returns (ordered_names, tie_index): ordered_names is largest-to-smallest;
    tie_index is the adjacent-pair index (i, i+1) whose values were forced
    equal, or None if this example carries no tie. Only num_entities==2
    (tie_index=0, both entities) or ==4 (tie_index=1, the sole interior pair
    that touches neither extremum) can produce a tie; num_entities==3 never
    does, structurally (see module docstring)."""
    names = rng.sample(pool, num_entities)
    values = rng.sample(range(0, 100_000), num_entities)
    order = sorted(zip(names, values), key=lambda pair: -pair[1])
    ordered_names = [name for name, _ in order]

    tie_index = None
    if num_entities in (2, 4) and rng.random() < TIE_PROBABILITY:
        tie_index = 0 if num_entities == 2 else 1

    return ordered_names, tie_index


def _make_example(rng, domain_name, pool, form_ids, num_entities):
    attr_noun = DOMAINS[domain_name]["attr_noun"]
    ordered, tie_index = _sample_chain(rng, pool, num_entities)

    premise_form = rng.choice(form_ids)
    question_form = rng.choice(form_ids)

    premise_sentences = []
    for i in range(num_entities - 1):
        if i == tie_index:
            premise_sentences.append(render_equal_premise(ordered[i], ordered[i + 1], attr_noun))
        else:
            premise_sentences.append(render_premise(premise_form, ordered[i], ordered[i + 1], attr_noun))
    rng.shuffle(premise_sentences)

    is_pairwise_tie = tie_index == 0 and num_entities == 2
    asked_direction = rng.choice(["largest", "smallest"])

    if is_pairwise_tie:
        direction = "equal"
        answer = EQUAL_WORD
    else:
        direction = asked_direction
        answer = ordered[0] if direction == "largest" else ordered[-1]

    question_entities = list(ordered)
    rng.shuffle(question_entities)
    question = render_question(question_form, question_entities, attr_noun, asked_direction)

    prompt = " ".join(premise_sentences) + " " + question
    return {
        "domain": domain_name,
        "num_entities": num_entities,
        "entities": ordered,
        "premise_form_id": premise_form,
        "question_form_id": question_form,
        "direction": direction,
        "has_tie": tie_index is not None,
        "prompt": prompt,
        "answer": answer,
    }


def _dedup_key(example):
    """Keys on the semantic chain only (domain, entities, direction, has_tie),
    not on which template form rendered it. The same chain phrased with a
    different form is still the same reasoning problem, and must not be
    allowed to appear in both train and test_seen under different wording
    (that would silently leak the answer via memorized entity/order, not
    test anything). has_tie is included because two examples with the same
    entity order can differ in whether an interior pair is exactly tied."""
    return (
        example["domain"],
        tuple(example["entities"]),
        example["direction"],
        example["has_tie"],
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
    tie_count = sum(1 for row in all_rows if row["has_tie"])
    return {
        "total_examples": len(all_rows),
        "distinct_form_pairs": len(forms_used),
        "domains_used": sorted(domains_used),
        "chain_depths_used": sorted(depths_used),
        "tie_count": tie_count,
        "tie_share": round(tie_count / len(all_rows), 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    args = parser.parse_args()

    all_examples = generate_all(seed=args.seed)
    write_splits(all_examples, args.out_dir)
    print(_template_variety_stats(all_examples))
