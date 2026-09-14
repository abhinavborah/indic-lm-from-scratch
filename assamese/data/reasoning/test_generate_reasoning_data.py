#!/usr/bin/env python3
"""Self-check for the reasoning dataset generator. Run directly:
    python3 test_generate_reasoning_data.py
"""

import sys
from pathlib import Path

from entities import POOLS
from generate_reasoning_data import generate_all
from templates import DOMAINS, EQUAL_WORD, HELD_FORM_ID, TRAIN_FORM_IDS

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
    # Keyed on the semantic chain only (not form ids): the same chain phrased
    # differently is still the same problem, and must not span both train
    # and test_seen (see generate_reasoning_data.py's _dedup_key docstring).
    keys = [(r["domain"], tuple(r["entities"]), r["direction"], r["has_tie"]) for r in seen_pool_union]
    assert len(keys) == len(set(keys)), "duplicate chain found across train/val/test_seen"

    for split_rows in examples.values():
        for row in split_rows:
            if row["direction"] == "equal":
                assert row["answer"] == EQUAL_WORD, f"equal-direction answer must be the equal word: {row}"
            else:
                assert row["answer"] in row["entities"], f"answer not among listed entities: {row}"

    # --- tie/equal-case checks (added 2026-09-14 per spec's "greater, smaller,
    # equal" requirement and user direction on scope/share/extremum-safety) ---
    all_rows = [row for rows in examples.values() for row in rows]
    tie_rows = [row for row in all_rows if row["has_tie"]]
    assert tie_rows, "no tie examples generated at all"

    tie_share = len(tie_rows) / len(all_rows)
    assert 0.03 <= tie_share <= 0.20, \
        f"tie share {tie_share:.3f} far from the targeted ~10% (small-batch tolerance 3-20%)"

    for row in tie_rows:
        assert row["num_entities"] in (2, 4), \
            f"only pairwise (2) or 3-hop (4) examples may carry a tie, got num_entities={row['num_entities']}: {row}"
        if row["num_entities"] == 2:
            assert row["direction"] == "equal" and row["answer"] == EQUAL_WORD, \
                f"pairwise tie must have direction='equal' and answer={EQUAL_WORD!r}: {row}"
        else:
            # 3-hop (4-entity) tie: interior pair only, answer stays the single
            # unambiguous extremum entity, exactly as a non-tied example would.
            assert row["direction"] in ("largest", "smallest")
            assert row["answer"] == (row["entities"][0] if row["direction"] == "largest" else row["entities"][-1])

    for row in all_rows:
        if row["num_entities"] == 3:
            assert not row["has_tie"], \
                f"a 2-hop (3-entity) example must never carry a tie (extremum-ambiguity rule): {row}"

    # Longest-case token-budget check against the real trained tokenizer:
    # a 3-hop chain (4 entities) rendered with the held-out (longest) forms,
    # must fit in context_length=256 with headroom for the </s> delimiters
    # and the answer span (see finetune_protocol.md's sequence format).
    import sentencepiece as spm

    tokenizer_path = Path(__file__).resolve().parents[2] / "tokenizer" / "assamese_bpe_8000.model"
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
    print("assamese reasoning-data generator self-check: OK")


if __name__ == "__main__":
    demo()
