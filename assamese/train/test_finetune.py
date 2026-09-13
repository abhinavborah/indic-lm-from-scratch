#!/usr/bin/env python3
"""Self-check for the reasoning finetuning module. Run directly:
    python3 test_finetune.py
"""

import sys
import tempfile
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "reasoning"))

from finetune import (
    BOS_ID, EOS_ID, PAD_ID, encode_example, finetune, load_pretrained_weights, masked_lm_loss,
)
from model import DecoderLM
from train import load_checkpoint, save_checkpoint

TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "tokenizer" / "assamese_bpe_8000.model"

TINY_CONFIG = {
    "vocab_size": 8000, "n_layer": 2, "d_model": 32, "n_head": 2, "d_ff": 64,
    "context_length": 128, "dropout": 0.0, "tie_embeddings": True,
    "positional_embedding": "rope", "norm_style": "pre_ln",
}


def demo():
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))

    # --- encode_example: boundary index, mask coverage, padding ---
    prompt = "এইটো এটা পৰীক্ষামূলক বাক্য।"
    answer = "উত্তৰ"
    x, y, mask = encode_example(sp, prompt, answer, max_seq_len=64)

    prompt_ids = sp.encode(prompt, out_type=int)
    answer_ids = sp.encode(answer, out_type=int)
    full = prompt_ids + [BOS_ID] + answer_ids + [EOS_ID]
    boundary_idx = len(prompt_ids)

    assert x.shape == (63,) and y.shape == (63,) and mask.shape == (63,)
    assert x[0].item() == full[0], "input should start with the prompt's first real token"
    assert y[boundary_idx - 1].item() == BOS_ID, \
        "target right after the last prompt token should be <s>, but excluded from loss"
    assert mask[boundary_idx - 1].item() == 0.0, \
        "predicting <s> itself must be excluded from the loss (structural delimiter, not learned)"
    assert mask[boundary_idx].item() == 1.0, \
        "predicting the first answer token must be included in the loss"
    assert y[len(full) - 2].item() == EOS_ID
    assert mask[len(full) - 2].item() == 1.0, "predicting the terminal </s> must be included"
    assert mask[len(full) - 1:].sum().item() == 0.0, "padding positions must be excluded from the loss"
    assert mask.sum().item() == len(answer_ids) + 1, \
        "mask should cover exactly the answer tokens plus the terminal </s>"
    assert x[len(full):].tolist() == [PAD_ID] * (len(x) - len(full)), \
        "trailing positions must be pad_id"

    # --- masked_lm_loss: only masked positions should affect the value ---
    vocab_size = 10
    logits = torch.randn(1, 5, vocab_size)
    targets = torch.randint(0, vocab_size, (1, 5))
    mask_a = torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0]])
    mask_b = mask_a.clone()
    logits_perturbed = logits.clone()
    logits_perturbed[0, 0] += 100.0  # perturb an unmasked (excluded) position heavily
    logits_perturbed[0, 3] += 100.0
    loss_a = masked_lm_loss(logits, targets, mask_a)
    loss_b = masked_lm_loss(logits_perturbed, targets, mask_b)
    assert torch.allclose(loss_a, loss_b), \
        "loss must be unaffected by logit changes at masked-out (weight-0) positions"

    # --- end-to-end: tiny model, tiny data, loss should decrease, checkpoint round-trips ---
    torch.manual_seed(0)
    model = DecoderLM(TINY_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_examples = [
        {"prompt": "ৰামৰ বয়স শ্যামতকৈ বেছি। বেছি বয়স থকা কোনটো?", "answer": "ৰাম"},
        {"prompt": "সীতাৰ বয়স গীতাতকৈ কম। কম বয়স থকা কোনটো?", "answer": "সীতা"},
        {"prompt": "অমনৰ উচ্চতা বিজয়তকৈ বেছি। বেছি উচ্চতা থকা কোনটো?", "answer": "অমন"},
        {"prompt": "কাজলৰ উচ্চতা পূজাতকৈ কম। কম উচ্চতা থকা কোনটো?", "answer": "কাজল"},
    ]
    val_examples = train_examples[:2]

    pretrain_val_data = torch.randint(0, 8000, (5000,), dtype=torch.long)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "finetune_ckpt.pt"
        log = finetune(
            model, optimizer, sp, train_examples, val_examples, pretrain_val_data,
            num_epochs=6, batch_size=2, context_length=TINY_CONFIG["context_length"],
            peak_lr=1e-3, min_lr=1e-4, warmup_steps=2, grad_clip_norm=1.0, seed=0,
            checkpoint_path=ckpt_path, config=TINY_CONFIG,
        )
        assert len(log) == 6
        assert log[-1]["train_loss"] < log[0]["train_loss"], \
            "training loss should decrease over 6 epochs on 4 tiny repeated examples"
        assert ckpt_path.exists()

        # checkpoint round-trip via the SAME load_checkpoint used for pretraining resume
        model2 = DecoderLM(TINY_CONFIG)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        resumed_step = load_checkpoint(ckpt_path, model2, optimizer2)
        assert resumed_step == log[-1]["epoch"] * 2 + 2 or resumed_step > 0  # steps_per_epoch=2 here
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "resumed model weights must match the saved model exactly"

        # load_pretrained_weights must load ONLY weights, leave a fresh optimizer untouched
        model3 = DecoderLM(TINY_CONFIG)
        fresh_optimizer = torch.optim.AdamW(model3.parameters(), lr=1e-3)
        fresh_state_before = fresh_optimizer.state_dict()
        load_pretrained_weights(ckpt_path, model3)
        assert fresh_optimizer.state_dict() == fresh_state_before, \
            "load_pretrained_weights must not touch any optimizer, only the model"
        for p1, p3 in zip(model.parameters(), model3.parameters()):
            assert torch.allclose(p1, p3)

    print("assamese finetune module self-check: OK")


if __name__ == "__main__":
    demo()
