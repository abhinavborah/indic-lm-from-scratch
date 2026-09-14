# Monolingual Transformer LMs: Hindi and Assamese

Two fully independent, from-scratch, ~25M-parameter decoder-only Transformer language models, built for CL3410 (Language Models and Agents). Model H is Hindi, the higher-resource language. Model L is Assamese, the lower-resource language, chosen from the assignment's allowed list. Data, tokenizer, vocabulary, and weights are kept separate throughout. No artifacts are shared between the two models.

## Repository layout

```
├── hindi/
│   ├── data/        # collection scripts, SOURCES.md; splits/ and COLLECTION_LOG.md are gitignored (splits/ on Drive, see Drive links; COLLECTION_LOG.md is a local per-item log, not a graded deliverable)
│   ├── tokenizer/   # training code and trained vocab/model files (hindi_bpe_8000.{model,vocab})
│   ├── model/       # model.py (decoder-only Transformer, from scratch), test_model.py
│   ├── train/       # train.py, smoke_test.py, test_train.py, colab_full_training.ipynb;
│   │                # finetune.py (LoRA), lora_rank_sweep.py, test_finetune.py,
│   │                # colab_finetune.ipynb
│   ├── eval/        # eval_lm.py (PPL/BPB), eval_generation.py (BLEU/chrF/ROUGE-L,
│   │                # diversity/repetition), attention_analysis.py (heatmaps,
│   │                # entropy, distance), test_eval.py, test_generation.py, test_attention.py;
│   │                # reasoning_eval.py (pretrained vs finetuned exact match),
│   │                # attention_reasoning_prompts.py (post-finetune attention comparison),
│   │                # test_reasoning_eval.py
│   └── configs/     # model_config.json (architecture, optimizer, schedule, sweep results)
├── <lang>/data/reasoning/  # synthetic reasoning dataset generator (entities.py, templates.py,
│                           # generate_reasoning_data.py) and its train/val/test_*.jsonl output
├── assamese/
│   └── (same shape as hindi/)
└── report/
    ├── SOURCES.md                        # combined source tracking, both languages
    ├── token_progress.md                 # live token-count tracking against spec targets
    ├── phase1_dataset_statistics.md      # Phase 1 deliverable: full dataset and tokenizer stats
    ├── phase2_architecture.md            # Phase 2 deliverable: architecture design choices and justification
    ├── phase2_lm_eval.md                 # Phase 2 deliverable: PPL/BPB, loss curves, alpha, decontamination
    ├── phase2_generation_eval.md         # Phase 2 deliverable: BLEU/chrF/ROUGE-L, diversity/repetition
    ├── phase2_attention_analysis.md      # Phase 2 deliverable: heatmaps, entropy, mean attention distance
    ├── phase2_resource_comparison.md     # Phase 2 deliverable: training/eval cost, H vs L
    ├── phase2_gap_decomposition.md       # Phase 2: H vs L gap attributed to data/tokenization/model
    ├── phase3_reasoning_eval.md          # Phase 3 deliverable: pretrained vs finetuned reasoning accuracy
    ├── phase3_attention_reasoning.md     # Phase 3 deliverable: post-finetune attention comparison
    ├── final_report.md                   # Phase 3 deliverable: consolidated final report (4 spec questions)
    ├── make_figures.py                   # generates report/figures/*.png (Phase 1)
    ├── make_figures_phase2.py            # generates report/figures/phase2_*.png (Phase 2)
    ├── make_figures_phase3.py            # generates report/figures/phase3_*.png (Phase 3)
    ├── figures/                          # plots referenced by the report files above
    └── logs/                             # <lang>_loss_log.csv (Phase 2 per-step); phase3_finetune/
                                           # <lang>_finetune_seed{0,1,2}_log.csv (Phase 3 per-epoch)
```

## Setup

```bash
uv venv && uv pip install -r requirements.txt
# or, without uv:
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

## Reproduction steps (Phase 1)

Large data artifacts (raw and cleaned corpus text, train/val/test splits) are **not** in git. See the Drive links below. Everything needed to *regenerate* them from scratch is in this repo.

1. **Collect the corpus** (per language, run from `<lang>/data/scripts/`). Each source has its own collection script: `scraper.py` for sitemap-driven news sites, `*_download.py` for one-off sources, `wiki_dump_extract.py`/`wikiquote_dump_extract.py`/`wikisource_dump_extract.py` for Wikimedia dumps, `ocr_pipeline.py` for scanned textbook PDFs under `docs/<lang>/books/`. All are checkpointed (`.state.json`) and safe to interrupt and resume. See `<lang>/data/SOURCES.md` for the full per-source list and status.
2. **Track progress**: `python3 <lang>/data/scripts/token_tracker.py` rescans `data/raw/` and `data/clean/`, applies the cleaning/purity filter and exact-dedup, and updates `report/token_progress.md`.
3. **Build the final split**: `python3 <lang>/data/scripts/build_splits.py` performs dedup and a document-level 98/1/1 train/val/test split. The split is deterministic (hash-based, reproducible from the corpus alone) and written to `<lang>/data/splits/` (gitignored, see Drive links).
4. **Train the tokenizer**: from `<lang>/tokenizer/`,
   ```python
   from train_tokenizer import train
   train("../data/splits/train.txt", "<lang>_bpe_<vocab_size>", vocab_size)
   ```
   Both Hindi and Assamese use vocab_size 8,000. See `report/phase1_dataset_statistics.md` for how this was chosen. Different vocab sizes per language would have been spec-permitted, but the measured numbers did not justify a smaller vocab for Assamese once trained on the real corpus. Training samples via sentencepiece's own `input_sentence_size` mechanism rather than the full split, per course guidance that training on a representative sample is acceptable if justified.
5. **Generate report figures**: `python3 report/make_figures.py`.

## Deliverables (Phase 1)

- Dataset collection + preprocessing code: `<lang>/data/scripts/`
- Per-language dataset statistics: `report/phase1_dataset_statistics.md`
- Per-language train/val/test splits: `<lang>/data/splits/` (Drive, see below)
- Tokenizer training code: `<lang>/tokenizer/train_tokenizer.py`
- Vocabulary + tokenizer model files: `<lang>/tokenizer/*_bpe_*.{model,vocab}`

## Reproduction steps (Phase 2)

Checkpoints and the tokenized `.bin` corpus files are **not** in git, same large-artifact policy as Phase 1. See the Drive links below.

1. **Train**: run `<lang>/train/colab_full_training.ipynb` (e.g.
   `hindi/train/colab_full_training.ipynb`, same notebook shape for
   `assamese`). Cells 0-3 (repo clone, Drive mount, `pip install`) are
   Colab setup only; the actual tokenize/train/checkpoint loop (cells 4
   onward) is plain, portable Python. To run locally instead of on Colab,
   point the notebook's `REPO_DIR` at your local clone and `DATA_DIR`/
   `CHECKPOINT_DIR` at local paths (e.g. `hindi/data/splits` and a
   checkpoint directory of your choice) instead of the Drive mount paths,
   then run the cells top to bottom (skipping the Colab-only cells).
   Batch size and learning rate for each language were chosen by a real
   sweep (`<lang>/train/smoke_test.py`), not guessed; results are recorded
   in `<lang>/configs/model_config.json` under `batch_size_sweep`.
   Auto-resumes from a checkpoint if one already exists at
   `CHECKPOINT_PATH`; to start over, remove or rename that checkpoint file
   before running. Tokenizes `train.txt`/`val.txt` to a `uint16` memmap on
   first run (one-time, reused on every later run).
2. **Evaluate** (PPL/BPB, full held-out val and test splits, no sampling):
   ```bash
   python3 hindi/eval/eval_lm.py --checkpoint-path <checkpoint-dir>/hindi_checkpoint.pt
   ```
   Writes `hindi/eval/lm_metrics.json`. Same command for `assamese`.
   Results are in `report/phase2_lm_eval.md`.
3. **Generate loss-curve figures**: `python3 report/make_figures_phase2.py`,
   reads `<lang>_loss_log.csv` from the checkpoint directory.
4. **Generation quality** (BLEU, chrF, ROUGE-L, repetition/diversity
   diagnostics, 50 held-out prefixes x 4 decoding settings):
   ```bash
   python3 hindi/eval/eval_generation.py --checkpoint-path <checkpoint-dir>/hindi_checkpoint.pt
   ```
   Writes `hindi/eval/generation_metrics.json`. Same command for `assamese`.
   Results are in `report/phase2_generation_eval.md`.
5. **Attention analysis** (heatmaps, entropy, mean attention distance):
   ```bash
   python3 hindi/eval/attention_analysis.py --checkpoint-path <checkpoint-dir>/hindi_checkpoint.pt
   ```
   Writes `hindi/eval/attention_metrics.json` and heatmap PNGs to
   `report/figures/`. Same command for `assamese`. Results are in
   `report/phase2_attention_analysis.md`.

## Deliverables (Phase 2)

- Transformer implementation: `<lang>/model/model.py`, `test_model.py`
  (causal-mask no-leak test, parameter count)
- Model configuration files: `<lang>/configs/model_config.json`
- Parameter counts, architecture justification: `report/phase2_architecture.md`
- Training scripts: `<lang>/train/train.py`, `run_full_training_reference.py`,
  `<lang>/train/colab_full_training.ipynb`
- Training logs and loss curves: `report/logs/*_loss_log.csv`,
  `report/phase2_lm_eval.md`, `report/figures/phase2_loss_curves.png`
- PPL/BPB tables: `report/phase2_lm_eval.md`
- BLEU/chrF/ROUGE-L, generated samples, diversity/repetition stats:
  `report/phase2_generation_eval.md`
- Attention heatmaps, entropy/distance summaries: `report/phase2_attention_analysis.md`
- Resource-level comparison: `report/phase2_resource_comparison.md`
- H vs L gap discussion (data/tokenization/model): `report/phase2_gap_decomposition.md`
- Checkpoints: Drive, see below

## Reproduction steps (Phase 3)

Finetuned checkpoints are **not** in git, same large-artifact policy as Phases 1-2. See the Drive links below.

1. **Generate the reasoning dataset** (per language, run from `<lang>/data/reasoning/`): `python3 generate_reasoning_data.py`. Writes `train.jsonl`, `val.jsonl`, `test_seen.jsonl`, `test_unseen_entity.jsonl`, `test_unseen_wording.jsonl` (4,500/450/450/450/150 examples respectively, both languages). Held-out entity-name pools and held-out template surface forms are the leakage-avoidance axes; see `report/phase3_reasoning_eval.md`'s Dataset section for the exact methodology.
2. **Finetune**: run `<lang>/train/colab_finetune.ipynb` (same notebook shape for both languages). Cells 0-3 (repo clone, Drive mount, `pip install`) are Colab setup only. Loops all 3 seeds (`SEEDS = [0, 1, 2]`) in one execution; a seed whose finetuned checkpoint already exists on Drive is skipped, not retrained, so reconnecting after a disconnect and rerunning from the top picks up at the next un-run seed automatically. LoRA rank/alpha (8/16) were picked by an empirical sweep (`<lang>/train/lora_rank_sweep.py --pretrained-checkpoint <path> --pretrain-val-bin <path>`), not guessed; see `report/phase3_reasoning_eval.md`'s Method section for the sweep result and reasoning. Early stopping (patience on reasoning-validation loss) picks the actual stopping epoch per run; pretrain-val PPL is tracked every epoch as a forgetting diagnostic, not an auto-stop signal. Cell 7 merges each seed's LoRA checkpoint into a plain `DecoderLM` checkpoint, the same format Phase 2's eval/attention tooling already expects.
3. **Generate finetune loss-curve and forgetting-diagnostic figures**: `python3 report/make_figures_phase3.py`, reads the per-epoch CSV logs from `report/logs/phase3_finetune/`.
4. **Reasoning evaluation** (pretrained vs. finetuned exact match, 3 mandatory splits, strict and lenient scoring, never aggregated):
   ```bash
   python3 hindi/eval/reasoning_eval.py \
       --pretrained-checkpoint <checkpoint-dir>/hindi_checkpoint.pt \
       --finetuned-dir <finetune-checkpoint-dir>/hindi
   ```
   Writes `hindi/eval/reasoning_eval_metrics.json`. Same command for `assamese`. Results are in `report/phase3_reasoning_eval.md`.
5. **Post-finetune attention comparison** (pretrained vs. finetuned, early and late layer, on real comparative-reasoning prompts):
   ```bash
   python3 hindi/eval/attention_reasoning_prompts.py \
       --pretrained-checkpoint <checkpoint-dir>/hindi_checkpoint.pt \
       --finetuned-checkpoint <finetune-checkpoint-dir>/hindi/hindi_finetuned_seed0_merged.pt
   ```
   Writes `hindi/eval/attention_reasoning_metrics.json` and heatmap PNGs (`phase3_attention_reasoning_hindi_*.png`) to `report/figures/`. Same command for `assamese`. Results are in `report/phase3_attention_reasoning.md`.

## Deliverables (Phase 3)

- Reasoning dataset generation scripts: `<lang>/data/reasoning/entities.py`, `templates.py`, `generate_reasoning_data.py`
- Reasoning train/val/test splits: `<lang>/data/reasoning/*.jsonl` (small, committed directly, unlike Phase 1's large corpus splits)
- Finetuning scripts and configs: `<lang>/train/finetune.py`, `lora_rank_sweep.py`, `<lang>/train/colab_finetune.ipynb`
- Finetuning logs: `report/logs/phase3_finetune/*.csv`, `report/figures/phase3_finetune_loss_curves.png`, `report/figures/phase3_pretrain_ppl_forgetting.png`
- Pretrained vs. finetuned reasoning accuracy, qualitative examples: `report/phase3_reasoning_eval.md`
- Pretrain vs. finetune attention comparison: `report/phase3_attention_reasoning.md`
- Final comparison tables, error analysis, and the 4 required spec questions: `report/final_report.md`
- Finetuned checkpoints: Drive, see below

## Google Drive links

Large artifacts (raw + cleaned corpus text, train/val/test splits) are hosted on Drive, not committed to git, per the spec's size-limit requirement. Both raw and cleaned versions are included per course guidance.

**[Phase 1 data: Google Drive folder](https://drive.google.com/drive/folders/1ycjrl0WS9C6ZfskwoTjBoqdhnGlsrwMT?usp=sharing)**

Contents (each language's raw/clean/splits are provided both zipped and as plain folders, in full):

```
<folder root>/
├── hindi.zip       # data/raw/, data/clean/, data/splits/
├── assamese.zip    # data/raw/, data/clean/, data/splits/
├── hindi/
│   ├── raw/        # mirrors hindi/data/raw/ in this repo
│   ├── clean/      # mirrors hindi/data/clean/
│   └── splits/     # train.txt, val.txt, test.txt
└── assamese/
    ├── raw/        # mirrors assamese/data/raw/ in this repo
    ├── clean/      # mirrors assamese/data/clean/
    └── splits/     # train.txt, val.txt, test.txt
```

**Phase 2 checkpoints:**
- [Hindi checkpoint: Google Drive folder](https://drive.google.com/drive/folders/1v2GhqtMNP33-JFgCZ7BE2gl1S6rIKnvD?usp=sharing)
- [Assamese checkpoint: Google Drive folder](https://drive.google.com/drive/folders/1VAZTMNabqUhEb-xIl1H5GJHGG0iAOVKb?usp=drive_link)

Separate folder per language (not nested under the Phase 1 root above).
Each contains `<lang>_checkpoint.pt` (hindi: step 101,812; assamese: step
69,000, both one full epoch), `<lang>_train.bin`/`<lang>_val.bin`
(tokenized corpus, `uint16` memmap), and `<lang>_loss_log.csv`.

**Phase 3 finetuned checkpoints:**
- [Phase 3 checkpoints: Google Drive folder](https://drive.google.com/drive/folders/1RdI54dSkRMw3hSC0G-janaNNRwyJO_Cg?usp=drive_link)

One shared folder with a per-language subfolder inside (matches
`colab_finetune.ipynb`'s `FINETUNE_CHECKPOINT_DIR` convention, unlike the
separate per-language Phase 2 folders above):

```
lma_phase3_checkpoints/
├── hindi/
│   ├── hindi_finetuned_seed{0,1,2}.pt          # raw LoRA checkpoint, optimizer state included
│   ├── hindi_finetuned_seed{0,1,2}_merged.pt   # merged plain DecoderLM checkpoint
│   └── logs/
│       └── hindi_finetune_seed{0,1,2}_log.csv  # per-epoch train/val loss, pretrain-val PPL
└── assamese/
    ├── assamese_finetuned_seed{0,1,2}.pt
    ├── assamese_finetuned_seed{0,1,2}_merged.pt
    └── logs/
        └── assamese_finetune_seed{0,1,2}_log.csv
```

## Hard constraints (enforced throughout)

No pretrained models, no pretrained tokenizers, no HuggingFace Transformer model classes, no `nn.Transformer*`. Hindi and Assamese share no data, no vocabulary, and no weights. They are two fully independent pipelines, built and run separately.
