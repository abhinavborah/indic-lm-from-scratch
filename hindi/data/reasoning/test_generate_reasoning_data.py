#!/usr/bin/env python3
"""Self-check for the reasoning dataset generator. Run directly:
    python3 test_generate_reasoning_data.py
"""

import sys
from pathlib import Path

from entities import POOLS
from generate_reasoning_data import generate_all
from templates import DOMAINS, HELD_FORM_ID, TRAIN_FORM_IDS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tokenizer"))


def demo():
    assert len(set(POOLS["person"]["all"])) == len(POOLS["person"]["all"]), \
        "duplicate person name in pool"
    assert len(set(POOLS["object"]["all"])) == len(POOLS["object"]["all"]), \
        "duplicate object name in pool"
    assert not set(POOLS["person"]["all"]) & set(POOLS["object"]["all"]), \
        "person and object pools must be disjoint"
    assert len(POOLS["person"]["all"]) >= 110, "person pool below the validated 100+ target"
    assert len(POOLS["object"]["all"]) >= 110, "object pool below the validated 100+ target"

    small_targets = {
        "train": 200, "val": 30, "test_seen": 30,
        "test_unseen_entity": 30, "test_unseen_wording": 20,
    }
    examples = generate_all(seed=1, split_targets=small_targets)

    for split, count in small_targets.items():
        assert len(examples[split]) == count, f"{split}: expected {count}, got {len(examples[split])}"

    seen_split_names = ("train", "val", "test_seen")
    held_entity_names = {"height": set(POOLS["person"]["held"]), "age": set(POOLS["person"]["held"]),
                          "price": set(POOLS["object"]["held"]), "quantity": set(POOLS["object"]["held"])}
    train_entity_names = {"height": set(POOLS["person"]["train"]), "age": set(POOLS["person"]["train"]),
                           "price": set(POOLS["object"]["train"]), "quantity": set(POOLS["object"]["train"])}

    for split in seen_split_names:
        for row in examples[split]:
            assert set(row["entities"]) <= train_entity_names[row["domain"]], \
                f"{split} example uses a held-out entity: {row}"
            assert row["premise_form_id"] in TRAIN_FORM_IDS
            assert row["question_form_id"] in TRAIN_FORM_IDS

    for row in examples["test_unseen_entity"]:
        assert set(row["entities"]) <= held_entity_names[row["domain"]], \
            f"test_unseen_entity example uses a train-pool entity: {row}"
        assert row["premise_form_id"] in TRAIN_FORM_IDS
        assert row["question_form_id"] in TRAIN_FORM_IDS

    for row in examples["test_unseen_wording"]:
        assert set(row["entities"]) <= train_entity_names[row["domain"]]
        assert row["premise_form_id"] == HELD_FORM_ID
        assert row["question_form_id"] == HELD_FORM_ID

    seen_pool_union = examples["train"] + examples["val"] + examples["test_seen"]
    keys = [(r["domain"], tuple(r["entities"]), r["direction"], r["premise_form_id"], r["question_form_id"])
            for r in seen_pool_union]
    assert len(keys) == len(set(keys)), "duplicate instance found across train/val/test_seen"

    for split_rows in examples.values():
        for row in split_rows:
            assert row["answer"] in row["entities"], f"answer not among listed entities: {row}"

    # Longest-case token-budget check against the real trained tokenizer:
    # a 3-hop chain (4 entities) rendered with the held-out (longest) forms,
    # must fit in context_length=256 with headroom for the </s> delimiters
    # and the answer span (see finetune_protocol.md's sequence format).
    import sentencepiece as spm

    tokenizer_path = Path(__file__).resolve().parents[2] / "tokenizer" / "hindi_bpe_8000.model"
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))

    worst_case = None
    for row in examples["train"] + examples["test_unseen_wording"]:
        if row["num_entities"] == 4:
            worst_case = row
            if row["premise_form_id"] == HELD_FORM_ID or row["question_form_id"] == HELD_FORM_ID:
                break
    assert worst_case is not None, "no 4-entity (3-hop) example generated to stress-test"

    full_text = worst_case["prompt"] + " " + worst_case["answer"]
    token_ids = sp.encode(full_text, out_type=int)
    # +2 for the two </s> boundary/terminal markers added at finetune-collation time.
    assert len(token_ids) + 2 <= 256, \
        f"worst-case example ({len(token_ids)} tokens) does not fit context_length=256"

    print(f"worst-case 3-hop example token length: {len(token_ids)} (+2 for </s> markers)")
    print("hindi reasoning-data generator self-check: OK")


if __name__ == "__main__":
    demo()
