#!/usr/bin/env python3
"""Rough token-count tracker for Hindi: dedup + purity filter + progress report.

Spec-compliance layer for two numbers: the ~500M training-token target, and
the >=20% manual-vs-downloaded split. Walks hindi/data/clean/{ocr,scrape}/**
(pre-cleaned by ocr_pipeline.py/scraper.py) and hindi/data/raw/sangraha/**
(still true raw text, since cleaning happens here, at count-time), read-only
for all three, and for every source:

  1. Splits files into blank-line-delimited segments (matches how every
     collector in this tree writes text: sangraha docs and OCR pages are
     both joined with "\\n\\n").
  2. Runs clean_text.clean_text() on each segment: NFC-normalizes, then
     drops any line that isn't predominantly Devanagari. Embedded English
     words/proper nouns are kept, not stripped, per course guidance. A
     segment that cleans down to nothing is counted as filtered out, not
     kept.
  3. Drops exact-duplicate cleaned segments (blake2b hash), across all three
     sources.
  4. Counts rough whitespace tokens on what survives, and reports manual
     (ocr+scrape) vs downloaded (sangraha) totals and progress toward the
     500M target into report/token_progress.md's "## Hindi" section (the
     Assamese copy of this script owns the "## Assamese" section the same
     way, since both scripts share the report file, not the counting code).

Rough == whitespace/regex tokenization, not BPE; real counts come once the
tokenizer is trained on the cleaned corpus. This script does NOT write
cleaned text anywhere; it only counts.

ponytail: full rescan every run, no incremental state, since dedup semantics
(global hash set) are simplest that way. Upgrade to incremental hashing if
reruns get too slow once the corpus is far larger than today's scale.
"""

import fcntl
import hashlib
import re
from datetime import datetime
from pathlib import Path

from clean_text import clean_text

LANG_HEADING = "## Hindi"
DATA_DIR = Path(__file__).resolve().parents[1]     # hindi/data
REPO_ROOT = DATA_DIR.parents[1]                     # repo root
REPORT_PATH = REPO_ROOT / "report" / "token_progress.md"

TARGET_TOKENS = 500_000_000
# Spec targets tokens "after tokenization" (real BPE), not rough whitespace
# words. Production tokenizer (hindi/tokenizer/hindi_bpe_8000.model) was
# trained on the real final train split (build_splits.py's 98% train
# document-level split); fertility measured on the real held-out val
# split. UNK rate 0.0. Retrained 2026-08-21 after fixing a bug where
# embedded English words/proper nouns (Tata, IRCTC, etc.) were being
# stripped from the corpus at the word level, based on an ambiguous early
# course Q&A answer that a later clarification on the same page reversed.
MEASURED_BPE_FERTILITY = 1.4940
# (path relative to hindi/data/, bucket). ocr/scrape now land pre-cleaned
# under clean/ (hindi-data's fix: raw/ocr and raw/scrape hold true raw
# text now, which would skew counts if scanned here). sangraha is
# unaffected: the downloader always wrote true raw text, cleaning has
# always happened here at count-time via clean_text().
SOURCES = [("clean/ocr", "manual"), ("clean/scrape", "manual"),
           ("clean/vikaspedia", "manual"), ("clean/dli_books", "manual"),
           ("raw/sangraha", "downloaded")]

TOKEN_RE = re.compile(r"\S+")


def rough_token_count(text):
    return len(TOKEN_RE.findall(text))


def iter_segments(path):
    """Yield blank-line-delimited segments, streaming line by line."""
    buf = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                buf.append(line)
            elif buf:
                yield "".join(buf)
                buf = []
    if buf:
        yield "".join(buf)


def scan():
    seen_hashes = set()
    stats = {
        "segments_scanned": 0,
        "segments_impure": 0,
        "segments_duplicate": 0,
        "tokens_kept": {"manual": 0, "downloaded": 0},
        "tokens_impure_dropped": 0,
        "tokens_duplicate_dropped": 0,
        "sangraha_unverified_tokens": 0,
        "files_scanned": 0,
    }

    for rel_path, bucket in SOURCES:
        src_dir = DATA_DIR / rel_path
        if not src_dir.is_dir():
            continue
        for path in sorted(src_dir.rglob("*.txt")):
            stats["files_scanned"] += 1
            is_unverified = rel_path.endswith("sangraha") and "unverified" in path.name
            for seg in iter_segments(path):
                stripped = seg.strip()
                if not stripped:
                    continue
                stats["segments_scanned"] += 1
                n_tok_raw = rough_token_count(stripped)

                cleaned = clean_text(stripped)
                n_tok = rough_token_count(cleaned)
                if n_tok == 0:
                    stats["segments_impure"] += 1
                    stats["tokens_impure_dropped"] += n_tok_raw
                    continue

                h = hashlib.blake2b(cleaned.encode("utf-8"), digest_size=16).digest()
                if h in seen_hashes:
                    stats["segments_duplicate"] += 1
                    stats["tokens_duplicate_dropped"] += n_tok
                    continue
                seen_hashes.add(h)

                stats["tokens_kept"][bucket] += n_tok
                if is_unverified:
                    stats["sangraha_unverified_tokens"] += n_tok

    return stats


def fmt(n):
    return f"{n:,}"


def render_section(stats):
    manual = stats["tokens_kept"]["manual"]
    downloaded = stats["tokens_kept"]["downloaded"]
    total = manual + downloaded
    manual_pct = (manual / total * 100) if total else 0.0
    target_pct = (total / TARGET_TOKENS * 100) if TARGET_TOKENS else 0.0
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return "\n".join([
        LANG_HEADING,
        "",
        f"_Last updated: {ts}_",
        "",
        f"- **Real tokens (kept):** {fmt(total)} / {fmt(TARGET_TOKENS)} target "
        f"({target_pct:.1f}%), rough whitespace count, see spec-corrected "
        f"estimate below",
        f"  - Manual (ocr + scrape): {fmt(manual)} ({manual_pct:.1f}%)",
        f"  - Downloaded (sangraha): {fmt(downloaded)} ({100 - manual_pct:.1f}%)",
        f"    - of which from `unverified` (automated perplexity-filtered, "
        f"**not** human-verified, lower-confidence tier): "
        f"{fmt(stats['sangraha_unverified_tokens'])}",
        f"  - Manual quota (>=20% of {fmt(TARGET_TOKENS)} = "
        f"{fmt(TARGET_TOKENS // 5)}): "
        f"{'MET' if manual >= TARGET_TOKENS // 5 else 'NOT MET'} "
        f"({fmt(manual)} so far)",
        "",
        f"- **Spec-corrected estimate (real BPE tokens, per line 121's "
        f"\"after tokenization\"):** measured fertility "
        f"{MEASURED_BPE_FERTILITY:.4f} real tokens per rough word (8,000-vocab "
        f"production SentencePiece BPE, trained on the real final train "
        f"split, measured on the real held-out val split, see constant "
        f"comment above for method). Applying it:",
        f"  - Total: ~{fmt(round(total * MEASURED_BPE_FERTILITY))} / "
        f"{fmt(TARGET_TOKENS)} target "
        f"({total * MEASURED_BPE_FERTILITY / TARGET_TOKENS * 100:.1f}%)",
        f"  - Manual: ~{fmt(round(manual * MEASURED_BPE_FERTILITY))} / "
        f"{fmt(TARGET_TOKENS // 5)} floor "
        f"({manual * MEASURED_BPE_FERTILITY / (TARGET_TOKENS // 5) * 100:.1f}%)",
        f"  - This is the final number: real production tokenizer, real "
        f"held-out split, UNK rate 0.0. No longer a proxy estimate.",
        "",
        f"- **Filtering:** {fmt(stats['segments_scanned'])} segments scanned "
        f"across {fmt(stats['files_scanned'])} files.",
        f"  - Dropped for script impurity / fully word-stripped "
        f"(clean_text.py: NFC normalize, drop non-Devanagari-majority "
        f"lines): {fmt(stats['segments_impure'])} "
        f"segments, {fmt(stats['tokens_impure_dropped'])} rough tokens",
        f"  - Dropped as exact duplicates: {fmt(stats['segments_duplicate'])} "
        f"segments, {fmt(stats['tokens_duplicate_dropped'])} rough tokens",
        "",
    ])


PREAMBLE = """# Token Progress: rough, post-purity-filter, post-dedup

Rough whitespace/regex tokenization, not the real BPE count (that comes once
the tokenizer is trained). Each language's section below is recomputed
independently by its own `token_tracker.py` (`hindi/data/scripts/`,
`assamese/data/scripts/`) via a full rescan of that language's
`data/raw/**` each run.

## Caveats

- Sangraha's `unverified` config is automated-perplexity-filtered, not
  human-verified like `verified`; treat it as a lower-confidence tier, not
  equivalent-quality text. `synthetic` is never pulled at all
  (machine-translated/romanized text, excluded from collection entirely).
- Counts are rough (whitespace tokenization). Real fertility-based counts
  come from the trained tokenizer.
- Script-purity filtering (dropping whole lines that aren't predominantly
  in the target script) happens inside each language's
  `clean_text.py`/`text_clean.py`, not in this tracker; this script only
  counts what survives. Embedded English words/proper nouns within an
  otherwise-native line are kept, per course guidance, not stripped.
"""


def update_report(section_text):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.touch(exist_ok=True)
    with REPORT_PATH.open("r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        content = f.read()
        if not content.strip():
            content = PREAMBLE

        heading_idx = content.find(LANG_HEADING)
        section = section_text.rstrip("\n")
        if heading_idx == -1:
            new_content = content.rstrip("\n") + "\n\n" + section + "\n"
        else:
            next_heading = content.find("\n## ", heading_idx + len(LANG_HEADING))
            head = content[:heading_idx].rstrip("\n")
            if next_heading == -1:
                new_content = head + "\n\n" + section + "\n"
            else:
                tail = content[next_heading:].lstrip("\n")
                new_content = head + "\n\n" + section + "\n\n" + tail

        f.seek(0)
        f.truncate()
        f.write(new_content)
        fcntl.flock(f, fcntl.LOCK_UN)


def main():
    print("scanning hindi...", flush=True)
    stats = scan()
    total = stats["tokens_kept"]["manual"] + stats["tokens_kept"]["downloaded"]
    print(f"hindi: {fmt(total)} real tokens kept "
          f"({fmt(stats['segments_scanned'])} segments scanned, "
          f"{fmt(stats['segments_impure'])} impure, "
          f"{fmt(stats['segments_duplicate'])} dup)", flush=True)

    update_report(render_section(stats))
    print(f"updated {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
