# Monolingual Transformer LMs: Hindi and Assamese

Two fully independent, from-scratch, ~25M-parameter decoder-only Transformer language models, built for CL3410 (Language Models and Agents). Model H is Hindi (higher-resource); Model L is Assamese (lower-resource, from the assignment's allowed list). Separate data, tokenizer, vocabulary, and weights throughout — no data, tokenizer, or model artifacts are shared between the two.

Canonical spec: `docs/LMA_Individual_Project_v1-1.md` (not in this repo; see the course materials).

## Repository layout

```
├── hindi/
│   ├── data/        # collection scripts, SOURCES.md, COLLECTION_LOG.md, splits/ (gitignored, see Drive links)
│   ├── tokenizer/   # training code + trained vocab/model files (hindi_bpe_8000.{model,vocab})
│   ├── model/        # Phase 2
│   ├── train/         # Phase 2
│   ├── eval/          # Phase 2
│   └── configs/       # Phase 2
├── assamese/
│   ├── data/        # same shape as hindi/data/
│   ├── tokenizer/   # assamese_bpe_8000.{model,vocab}
│   ├── model/ train/ eval/ configs/   # Phase 2
└── report/
    ├── SOURCES.md                     # combined source tracking, both languages
    ├── token_progress.md              # live token-count tracking against spec targets
    ├── phase1_dataset_statistics.md   # Phase 1 deliverable: full dataset + tokenizer stats
    ├── make_figures.py                # generates report/figures/*.png
    └── figures/                       # plots referenced by phase1_dataset_statistics.md
```

## Setup

```bash
uv venv && uv pip install -r requirements.txt
# or, without uv:
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

## Reproduction steps (Phase 1)

Large data artifacts (raw/cleaned corpus text, train/val/test splits) are **not** in git — see Drive links below. Everything needed to *regenerate* them from scratch is.

1. **Collect the corpus** (per language, run from `<lang>/data/scripts/`): each source has its own collection script (`scraper.py` for sitemap-driven news sites, `*_download.py` for one-off sources, `wiki_dump_extract.py`/`wikiquote_dump_extract.py`/`wikisource_dump_extract.py` for Wikimedia dumps, `ocr_pipeline.py` for scanned textbook PDFs under `docs/<lang>/books/`). All are checkpointed (`.state.json`) and safe to interrupt/resume. See `<lang>/data/SOURCES.md` for the full per-source list and status.
2. **Track progress**: `python3 <lang>/data/scripts/token_tracker.py` rescans `data/raw/` + `data/clean/`, applies the cleaning/purity filter and exact-dedup, and updates `report/token_progress.md`.
3. **Build the final split**: `python3 <lang>/data/scripts/build_splits.py` — dedup + document-level 98/1/1 train/val/test split, deterministic (hash-based, reproducible from the corpus alone), written to `<lang>/data/splits/` (gitignored, see Drive links).
4. **Train the tokenizer**: from `<lang>/tokenizer/`,
   ```python
   from train_tokenizer import train
   train("../data/splits/train.txt", "<lang>_bpe_<vocab_size>", vocab_size)
   ```
   Both Hindi and Assamese use vocab_size 8,000 (see `report/phase1_dataset_statistics.md` for how this was chosen — different vocab sizes per language would have been spec-permitted, but the real measured numbers didn't justify a smaller vocab for Assamese once trained on the real corpus). Training samples via sentencepiece's own `input_sentence_size` mechanism rather than the full split, per course guidance that training on a representative sample is acceptable if justified.
5. **Generate report figures**: `python3 report/make_figures.py`.

## Deliverables (Phase 1)

- Dataset collection + preprocessing code: `<lang>/data/scripts/`
- Per-language dataset statistics: `report/phase1_dataset_statistics.md`
- Per-language train/val/test splits: `<lang>/data/splits/` (Drive, see below)
- Tokenizer training code: `<lang>/tokenizer/train_tokenizer.py`
- Vocabulary + tokenizer model files: `<lang>/tokenizer/*_bpe_*.{model,vocab}`

## Google Drive links

Large artifacts (raw + cleaned corpus text, train/val/test splits) are hosted on Drive, not committed to git, per the spec's size-limit requirement. Both raw and cleaned versions are included per course guidance.

| Artifact | Link |
|---|---|
| Hindi raw + clean corpus | _pending upload_ |
| Assamese raw + clean corpus | _pending upload_ |
| Hindi train/val/test splits | _pending upload_ |
| Assamese train/val/test splits | _pending upload_ |

## Hard constraints (enforced throughout)

No pretrained models, no pretrained tokenizers, no HuggingFace Transformer model classes, no `nn.Transformer*`. Hindi and Assamese share no data, no vocabulary, no weights — two fully independent pipelines built and run separately.
