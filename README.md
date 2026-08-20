# Monolingual Transformer LMs: Hindi and Assamese

Two fully independent, from-scratch, ~25M-parameter decoder-only Transformer language models, built for CL3410 (Language Models and Agents). Model H is Hindi, the higher-resource language. Model L is Assamese, the lower-resource language, chosen from the assignment's allowed list. Data, tokenizer, vocabulary, and weights are kept separate throughout. No artifacts are shared between the two models.

Canonical spec: `docs/LMA_Individual_Project_v1-1.md` (not in this repo, see the course materials).

## Repository layout

```
├── hindi/
│   ├── data/        # collection scripts, SOURCES.md; splits/ and COLLECTION_LOG.md are gitignored (splits/ on Drive, see Drive links; COLLECTION_LOG.md is a local per-item log, not a graded deliverable)
│   └── tokenizer/   # training code and trained vocab/model files (hindi_bpe_8000.{model,vocab})
├── assamese/
│   ├── data/        # same shape as hindi/data/
│   └── tokenizer/   # assamese_bpe_8000.{model,vocab}
└── report/
    ├── SOURCES.md                     # combined source tracking, both languages
    ├── token_progress.md              # live token-count tracking against spec targets
    ├── phase1_dataset_statistics.md   # Phase 1 deliverable: full dataset and tokenizer stats
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

## Hard constraints (enforced throughout)

No pretrained models, no pretrained tokenizers, no HuggingFace Transformer model classes, no `nn.Transformer*`. Hindi and Assamese share no data, no vocabulary, and no weights. They are two fully independent pipelines, built and run separately.
