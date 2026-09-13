#!/usr/bin/env python3
"""Self-check for the reasoning finetuning module. Run directly:
    python3 test_finetune.py
"""

import copy
import sys
import tempfile
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "reasoning"))

from finetune import (
    BOS_ID, EOS_ID, LoRALinear, PAD_ID, encode_example, finetune, freeze_non_lora_params,
    inject_lora, load_pretrained_weights, masked_lm_loss, merge_lora_to_plain,
)
from model import DecoderLM
from train import load_checkpoint, save_checkpoint

TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "tokenizer" / "hindi_bpe_8000.model"

TINY_CONFIG = {
    "vocab_size": 8000, "n_layer": 2, "d_model": 32, "n_head": 2, "d_ff": 64,
    "context_length": 128, "dropout": 0.0, "tie_embeddings": True,
    "positional_embedding": "rope", "norm_style": "pre_ln",
}

LORA_RANK = 4
LORA_ALPHA = 8


def demo():
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))

    # --- encode_example: boundary index, mask coverage, padding ---
    prompt = "यह एक परीक्षण वाक्य है।"
    answer = "उत्तर"
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

    # --- inject_lora: zero-init means behavior is unchanged right after injection ---
    torch.manual_seed(0)
    base_model = DecoderLM(TINY_CONFIG)
    probe_input = torch.randint(0, TINY_CONFIG["vocab_size"], (1, 16))
    base_model.eval()
    with torch.no_grad():
        logits_before = base_model(probe_input)

    lora_model = copy.deepcopy(base_model)
    inject_lora(lora_model, rank=LORA_RANK, alpha=LORA_ALPHA)
    lora_model.eval()
    with torch.no_grad():
        logits_after = lora_model(probe_input)
    assert torch.allclose(logits_before, logits_after, atol=1e-5), \
        "LoRA injection must not change model output before any training (lora_B starts at zero)"
    lora_model.train()

    # --- freeze_non_lora_params: only lora_A/lora_B are trainable ---
    freeze_non_lora_params(lora_model)
    trainable_names = {n for n, p in lora_model.named_parameters() if p.requires_grad}
    assert trainable_names, "at least some parameters must be trainable"
    assert all("lora_A" in n or "lora_B" in n for n in trainable_names), \
        f"only lora_A/lora_B should be trainable, got: {trainable_names}"
    frozen_names = {n for n, p in lora_model.named_parameters() if not p.requires_grad}
    assert any("q_proj.base" in n for n in frozen_names) and any("v_proj.base" in n for n in frozen_names), \
        "q_proj/v_proj base weights must be frozen"
    assert any("k_proj" in n for n in frozen_names), "k_proj must be untouched and frozen (not a LoRA target)"

    # --- end-to-end: tiny model, tiny data, loss should decrease, checkpoint round-trips ---
    torch.manual_seed(0)
    model = DecoderLM(TINY_CONFIG)
    inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    freeze_non_lora_params(model)
    base_snapshot = copy.deepcopy(dict(model.named_parameters()))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )

    train_examples = [
        {"prompt": "राम की उम्र श्याम से ज़्यादा है। ज़्यादा उम्र वाला कौन सा है?", "answer": "राम"},
        {"prompt": "सीता की उम्र गीता से कम है। कम उम्र वाला कौन सा है?", "answer": "सीता"},
        {"prompt": "अमन की ऊंचाई विजय से ज़्यादा है। ज़्यादा ऊंचाई वाला कौन सा है?", "answer": "अमन"},
        {"prompt": "काजल की ऊंचाई पूजा से कम है। कम ऊंचाई वाला कौन सा है?", "answer": "काजल"},
    ]
    val_examples = train_examples[:2]

    pretrain_val_data = torch.randint(0, 8000, (5000,), dtype=torch.long)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "finetune_ckpt.pt"
        log = finetune(
            model, optimizer, sp, train_examples, val_examples, pretrain_val_data,
            num_epochs=6, batch_size=2, context_length=TINY_CONFIG["context_length"],
            peak_lr=1e-2, min_lr=1e-3, warmup_steps=2, grad_clip_norm=1.0, seed=0,
            checkpoint_path=ckpt_path, config=TINY_CONFIG,
        )
        assert len(log) == 6
        assert log[-1]["train_loss"] < log[0]["train_loss"], \
            "training loss should decrease over 6 epochs on 4 tiny repeated examples"
        assert ckpt_path.exists()

        # frozen params must be bit-for-bit unchanged after training; only LoRA params may move
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert not torch.allclose(p, base_snapshot[name]), \
                    f"trainable param {name} did not change during training"
            else:
                assert torch.allclose(p, base_snapshot[name]), \
                    f"frozen param {name} changed during training but should not have"

        # checkpoint round-trip: rebuild the SAME LoRA structure before loading (keys must match)
        model2 = DecoderLM(TINY_CONFIG)
        inject_lora(model2, rank=LORA_RANK, alpha=LORA_ALPHA)
        freeze_non_lora_params(model2)
        optimizer2 = torch.optim.AdamW([p for p in model2.parameters() if p.requires_grad], lr=1e-2)
        resumed_step = load_checkpoint(ckpt_path, model2, optimizer2)
        assert resumed_step > 0
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "resumed model weights must match the saved model exactly"

        # load_pretrained_weights loads only weights into a PLAIN (pre-injection) model
        model3 = DecoderLM(TINY_CONFIG)
        fresh_optimizer = torch.optim.AdamW(model3.parameters(), lr=1e-3)
        fresh_state_before = fresh_optimizer.state_dict()
        pretrain_ckpt_path = Path(tmp) / "pretrain_ckpt.pt"
        save_checkpoint(pretrain_ckpt_path, base_model, torch.optim.AdamW(base_model.parameters()), 100, TINY_CONFIG)
        load_pretrained_weights(pretrain_ckpt_path, model3)
        assert fresh_optimizer.state_dict() == fresh_state_before, \
            "load_pretrained_weights must not touch any optimizer, only the model"
        for p_base, p3 in zip(base_model.parameters(), model3.parameters()):
            assert torch.allclose(p_base, p3)

        # merge_lora_to_plain: merged plain model must reproduce the LoRA model's forward pass
        model.eval()
        merged = merge_lora_to_plain(model, TINY_CONFIG)
        merged.eval()
        with torch.no_grad():
            lora_logits = model(probe_input)
            merged_logits = merged(probe_input)
        assert torch.allclose(lora_logits, merged_logits, atol=1e-5), \
            "merged plain model must produce identical logits to the trained LoRA model"
        assert not any(isinstance(m, LoRALinear) for m in merged.modules()), \
            "merged model must contain no LoRA wrapper modules"

    print("hindi finetune module self-check: OK")


if __name__ == "__main__":
    demo()
