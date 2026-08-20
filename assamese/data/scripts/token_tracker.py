#!/usr/bin/env python3
"""Rough token-count tracker for Assamese -- dedup + purity filter + progress report.

Spec-compliance layer for two numbers: the ~500M training-token target, and
the >=20% manual-vs-downloaded split. Walks
assamese/data/clean/{ocr,scrape}/** (pre-cleaned by ocr_pipeline.py/
scraper.py) and assamese/data/raw/sangraha/** (still true raw text --
cleaning happens here, at count-time) -- read-only for all three, and for
every source:

  1. Splits files into blank-line-delimited segments (matches how every
     collector in this tree writes text -- sangraha docs and OCR pages are
     both joined with "\\n\\n").
  2. Runs text_clean.clean_text() on each segment -- drops lines that aren't
     predominantly Bengali-Assamese-script, then strips individual
     Latin-script words from what's left (English brand names/proper nouns
     embedded mid-sentence). A segment that cleans down to nothing is
     counted as filtered out, not kept.
  3. For scrape/sangraha sources only (not ocr -- see below), runs
     text_clean.is_likely_assamese() as a second, separate filter: Assamese
     and Bengali share the same Unicode block, so step 2's script check
     can't tell them apart. This heuristic checks for the near-total absence
     of ৰ/ৱ (letters Assamese uses constantly, standard Bengali never) as a
     signal of "actually Bengali, not Assamese" -- catches cases like
     Sangraha's automated pipeline mislabeling adjacent-language text.
     Skipped for ocr/ specifically: ocr_pipeline.py's correct_ra() already
     force-converts all Bengali "র" to "ৰ" before writing, which would make
     this check trivially pass on anything and tell us nothing.
  4. Drops exact-duplicate cleaned segments (blake2b hash), across all three
     sources.
  5. Counts rough whitespace tokens on what survives, and reports manual
     (ocr+scrape) vs downloaded (sangraha) totals and progress toward the
     500M target into report/token_progress.md's "## Assamese" section (the
     Hindi copy of this script owns the "## Hindi" section the same way --
     both scripts share the report file, not the counting code).

Rough == whitespace/regex tokenization, not BPE -- real counts come once the
tokenizer is trained on the cleaned corpus. This script does NOT write
cleaned text anywhere; it only counts.

ponytail: full rescan every run, no incremental state -- dedup semantics
(global hash set) are simplest that way. Upgrade to incremental hashing if
reruns get too slow once the corpus is far larger than today's scale.
"""

import fcntl
import hashlib
import re
from datetime import datetime
from pathlib import Path

from text_clean import clean_text, is_likely_assamese

LANG_HEADING = "## Assamese"
DATA_DIR = Path(__file__).resolve().parents[1]     # assamese/data
REPO_ROOT = DATA_DIR.parents[1]                     # repo root
REPORT_PATH = REPO_ROOT / "report" / "token_progress.md"

TARGET_TOKENS = 500_000_000
# Spec targets tokens "after tokenization" (real BPE), not rough whitespace
# words. Production vocab size decided this session: 5,000 (see CONTEXT.md
# 2026-08-20 entry -- Assamese's fertility curve never flattens the way
# Hindi's does even at 24K, so the deciding factors were the embedding-vs-
# depth budget tradeoff at 25M params and the corpus's own data-scarcity/
# duplication risk, not a fertility knee). This is now the REAL number:
# the production tokenizer (assamese/tokenizer/assamese_bpe_5000.model) was
# trained on the real final train split (build_splits.py's 98% train
# document-level split), and fertility was measured on the real held-out
# val split -- 1.9220 real tokens per rough word, UNK rate 0.0. Lower than
# the earlier 32M-char proxy-sample estimate (2.369): the full corpus gives
# the tokenizer far more data to learn efficient merges from than a small
# sample could, so real compression came out better than the proxy
# predicted. This is the final number -- no more re-measuring needed unless
# the corpus or vocab size changes.
MEASURED_BPE_FERTILITY = 1.9220
# (path relative to assamese/data/, bucket). ocr/scrape now land pre-cleaned
# under clean/ (assamese-data's fix -- raw/ocr and raw/scrape hold true raw
# text now, which would skew counts if scanned here). sangraha is
# unaffected -- the downloader always wrote true raw text, cleaning has
# always happened here at count-time via clean_text().
SOURCES = [("clean/ocr", "manual"), ("clean/scrape", "manual"),
           ("clean/vikaspedia", "manual"), ("clean/dipr", "manual"),
           ("clean/jibonorshongram", "manual"), ("clean/newsonair", "manual"),
           ("clean/assam_gazette", "manual"),
           ("clean/jibonorshongram_articles", "manual"),
           ("clean/dli_books", "manual"), ("clean/shodhganga", "manual"),
           ("raw/sangraha", "downloaded"), ("raw/mwirelabs", "downloaded"),
           ("raw/cc100", "downloaded"), ("raw/indiccorpv2", "downloaded")]

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
        "segments_suspect_bengali": 0,
        "segments_duplicate": 0,
        "tokens_kept": {"manual": 0, "downloaded": 0},
        "tokens_impure_dropped": 0,
        "tokens_suspect_bengali_dropped": 0,
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

                cleaned, _dropped_lines = clean_text(stripped)
                n_tok = rough_token_count(cleaned)
                if n_tok == 0:
                    stats["segments_impure"] += 1
                    stats["tokens_impure_dropped"] += n_tok_raw
                    continue

                if not rel_path.endswith("ocr") and not is_likely_assamese(cleaned):
                    stats["segments_suspect_bengali"] += 1
                    stats["tokens_suspect_bengali_dropped"] += n_tok
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
        f"({target_pct:.1f}%) -- rough whitespace count, see spec-corrected "
        f"estimate below",
        f"  - Manual (ocr + scrape): {fmt(manual)} ({manual_pct:.1f}%)",
        f"  - Downloaded (sangraha): {fmt(downloaded)} ({100 - manual_pct:.1f}%)",
        f"    - of which from `unverified` (automated perplexity-filtered, "
        f"**not** human-verified -- lower-confidence tier): "
        f"{fmt(stats['sangraha_unverified_tokens'])}",
        f"  - Manual quota (>=20% of {fmt(TARGET_TOKENS)} = "
        f"{fmt(TARGET_TOKENS // 5)}): "
        f"{'MET' if manual >= TARGET_TOKENS // 5 else 'NOT MET'} "
        f"({fmt(manual)} so far)",
        "",
        f"- **Spec-corrected estimate (real BPE tokens, per line 121's "
        f"\"after tokenization\"):** measured fertility "
        f"{MEASURED_BPE_FERTILITY:.4f} real tokens per rough word (5,000-vocab "
        f"production SentencePiece BPE, trained on the real final train "
        f"split, measured on the real held-out val split -- see constant "
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
        f"(text_clean.py -- strip embedded Latin words, drop non-Bengali-"
        f"Assamese-block-majority lines): {fmt(stats['segments_impure'])} "
        f"segments, {fmt(stats['tokens_impure_dropped'])} rough tokens",
        f"  - Dropped as suspected Bengali-not-Assamese (ৰ/ৱ heuristic, "
        f"scrape+sangraha only -- see caveat below): "
        f"{fmt(stats['segments_suspect_bengali'])} segments, "
        f"{fmt(stats['tokens_suspect_bengali_dropped'])} rough tokens",
        f"  - Dropped as exact duplicates: {fmt(stats['segments_duplicate'])} "
        f"segments, {fmt(stats['tokens_duplicate_dropped'])} rough tokens",
        "",
        "**Note on the ৰ/ৱ heuristic:** this is a frequency heuristic, not a "
        "real language-ID classifier -- two letters' frequency is a weak "
        "signal on its own. A lightweight tool like `langid.py`, fastText "
        "`lid.176`, or `cld3` would do a properly calibrated job; not added "
        "now, just flagging it as a future option. It's skipped entirely for "
        "the ocr/ source since the OCR pipeline's Bengali-to-Assamese "
        "correction pass already force-converts every Bengali \"র\" to \"ৰ\", "
        "which would make this check trivially pass on anything.",
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
  human-verified like `verified` -- treat it as a lower-confidence tier, not
  equivalent-quality text. `synthetic` is never pulled at all
  (machine-translated/romanized text, excluded from collection entirely).
- Counts are rough (whitespace tokenization). Real fertility-based counts
  come from the trained tokenizer.
- Word-level foreign-word stripping (per course guidance: digits are fine,
  but embedded non-language words must go) and script-purity filtering both
  happen inside each language's `clean_text.py`/`text_clean.py`, not in this
  tracker -- this script only counts what survives.
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
    print("scanning assamese...", flush=True)
    stats = scan()
    total = stats["tokens_kept"]["manual"] + stats["tokens_kept"]["downloaded"]
    print(f"assamese: {fmt(total)} real tokens kept "
          f"({fmt(stats['segments_scanned'])} segments scanned, "
          f"{fmt(stats['segments_impure'])} impure, "
          f"{fmt(stats['segments_suspect_bengali'])} suspect-bengali, "
          f"{fmt(stats['segments_duplicate'])} dup)", flush=True)

    update_report(render_section(stats))
    print(f"updated {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
