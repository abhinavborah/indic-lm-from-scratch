#!/usr/bin/env python3
"""Assamese manual-collection corpus: yt-dlp auto-generated 'as' captions
from long-form Assamese podcast/documentary channels.

This is machine transcription (YouTube ASR), not human transcription, a
different tier from OCR/scrape, but still real: verified on real videos this
session that the extracted text is coherent, topical, natural-sounding
Assamese, not garbled machine-translation-from-a-different-language (the
per-video "language" metadata field is often wrong: hi/bn/en-US declared
even for genuinely Assamese-spoken content; but the "as" auto-caption
track itself reads as real Assamese regardless of that mislabel).

Known ASR artifact this session found and filters for: stutter/repetition
loops (e.g. a word repeated 5-10x in a row) that survive the language-purity
filter (still valid in-script Assamese) but are not natural phrasing;
collapse_repeats() strips these before counting/writing clean output.

YouTube's timedtext/caption endpoint rate-limits aggressively and the block
does not clear quickly (observed: still 429ing 10+ minutes after the last
successful fetch, not a simple few-seconds throttle); REQUEST_DELAY_S is
deliberately generous, and BACKOFF_BASE_S/BACKOFF_CAP_S govern the retry
backoff on a 429. A video that still fails after MAX_RETRIES attempts is
pushed to the end of the in-run queue (not marked done) and retried once
the rest of the queue has been attempted, per the checkpoint contract below.

Resumable: assamese/data/.state.json under the "youtube_captions" namespace
tracks every video id already written (success or permanent skip: no
caption track / video unavailable), so a rerun only fetches new videos.
Shares the state file with scraper.py/ocr_pipeline.py via state_io's
flock-protected read-modify-write; safe to run concurrently.
"""

import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]  # assamese/data
RAW_OUT_DIR = DATA_DIR / "raw" / "youtube_captions"
CLEAN_OUT_DIR = DATA_DIR / "clean" / "youtube_captions"
STATE_FILE = DATA_DIR / ".state.json"
LOG_FILE = DATA_DIR / "COLLECTION_LOG.md"
# Each channel gets its OWN top-level state_io namespace (not a shared
# "youtube_captions" dict keyed by channel name); each channel runs as its
# own OS process (separate herdr pane per channel), and state_io's
# load-once-save-wholesale namespace contract only serializes writes within
# a single process's lifetime, not across processes that each hold a stale
# full-namespace snapshot from startup. Found this session: three channels
# in three concurrent processes sharing one "youtube_captions" namespace
# clobbered each other's checkpoints on every save (last writer wins for the
# whole namespace); aboyobbhuyan's 5 real successful downloads survived on
# disk (raw/clean text files are per-channel, unaffected) but its done_ids
# checkpoint was wiped by the other two channels' saves, which would have
# silently re-downloaded and re-appended them on resume. One namespace per
# channel makes each process's read-modify-write independent, like
# scraper.py's per-source sub-dicts are for threads within one process.
NAMESPACE_PREFIX = "youtube_captions"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_io  # noqa: E402
from text_clean import clean_text, is_likely_assamese  # noqa: E402

CAPTION_LANG = "as"
REQUEST_DELAY_S = 20  # generous baseline pacing between videos, not just on failure
MAX_RETRIES = 3
BACKOFF_BASE_S = 45
BACKOFF_CAP_S = 480
LOG_EVERY_N_VIDEOS = 5
BATCH_MAX_SECONDS = 86400  # 24h safety net, matches scraper.py's convention

# Structured as a list so adding a channel is a one-line addition, not a
# rewrite; each name is run as its own pane/process (see COLLECTION_LOG.md
# for per-channel progress), sharing this file's state namespace by keying
# on "name" the same way scraper.py's SOURCES do.
SOURCES = [
    {"name": "aboyobbhuyan", "channel_url": "https://www.youtube.com/@aboyobbhuyan/videos"},
    {"name": "jssunsscripted", "channel_url": "https://www.youtube.com/@JSSUnsscripted/videos"},
    {"name": "unfilteredwithkrishnakshi", "channel_url": "https://www.youtube.com/@Unfilteredwithkrishnakshi/videos"},
]


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {msg}\n")
    print(msg, flush=True)


def discover_video_ids(channel_url):
    r = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "%(id)s", channel_url],
        capture_output=True, text=True, timeout=120,
    )
    ids = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    return ids or None


def vtt_to_text(vtt_path):
    """Strip VTT cue timing/inline timestamp tags and collapse the
    rolling-caption duplicate lines auto-captions emit for overlapping cues."""
    raw = vtt_path.read_text(encoding="utf-8", errors="replace")
    lines = []
    for line in raw.splitlines():
        if not line.strip() or line.startswith(("WEBVTT", "Kind:", "Language:")) or "-->" in line:
            continue
        cleaned = re.sub(r"<[^>]+>", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    deduped, prev = [], None
    for l in lines:
        if l != prev:
            deduped.append(l)
        prev = l
    return "\n".join(deduped)


REPEAT_RUN_RE = re.compile(r"(\S+)(?:\s+\1){2,}")  # same word 3+ times in a row


def collapse_repeats(text):
    """ASR stutter artifact found this session, e.g. 'ইট ইট ইট ইট ইট ইট ই ই
    শক্তি', valid in-script Assamese, so the purity filter lets it through,
    but it isn't natural phrasing. Collapse a repeated-word run down to one
    occurrence rather than dropping the whole line (surrounding words are
    usually genuine)."""
    return REPEAT_RUN_RE.sub(r"\1", text)


def download_captions(video_id):
    """Returns the raw caption text on success, "" if the video genuinely has
    no 'as' caption track (permanent skip), or None on a transient failure
    (retry later)."""
    out_template = str(RAW_OUT_DIR / "_tmp_%(id)s.%(ext)s")
    vtt_path = RAW_OUT_DIR / f"_tmp_{video_id}.{CAPTION_LANG}.vtt"
    for attempt in range(1, MAX_RETRIES + 1):
        r = subprocess.run(
            ["yt-dlp", "--write-auto-sub", "--sub-lang", CAPTION_LANG, "--skip-download",
             "--sub-format", "vtt", "-o", out_template,
             f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=180,
        )
        if vtt_path.exists() and vtt_path.stat().st_size > 0:
            text = vtt_to_text(vtt_path)
            vtt_path.unlink()
            return text
        stderr = r.stderr or ""
        if "Too Many Requests" in stderr or "429" in stderr:
            backoff = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_CAP_S)
            log(f"{video_id}: 429 on attempt {attempt}/{MAX_RETRIES}, backing off {backoff}s")
            time.sleep(backoff)
            continue
        if "no subtitles" in stderr.lower() or "Requested format is not available" in stderr:
            return ""  # confirmed no 'as' track, permanent skip
        log(f"{video_id}: fetch attempt {attempt}/{MAX_RETRIES} failed: "
            f"{stderr.strip().splitlines()[-1] if stderr.strip() else 'unknown error'}")
        time.sleep(REQUEST_DELAY_S)
    return None  # exhausted retries this pass, caller requeues to end


def process_source(source):
    name = source["name"]
    namespace = f"{NAMESPACE_PREFIX}_{name}"
    entry = state_io.load_namespace(
        STATE_FILE, namespace,
        {"done_ids": [], "total_words": 0, "dropped_lines": 0,
         "suspect_language": 0, "repeat_collapsed": 0},
    )
    done = set(entry["done_ids"])

    ids = None
    for attempt in range(3):
        ids = discover_video_ids(source["channel_url"])
        if ids is not None:
            break
        log(f"{name}: channel discovery attempt {attempt + 1}/3 failed, retrying")
        time.sleep(REQUEST_DELAY_S)
    if ids is None:
        log(f"{name}: channel discovery failed 3/3 times, skipping this run")
        return

    queue = deque(v for v in ids if v not in done)
    log(f"{name}: {len(ids)} videos in channel, {len(queue)} new")

    raw_path = RAW_OUT_DIR / f"{name}.txt"
    clean_path = CLEAN_OUT_DIR / f"{name}.txt"
    processed_this_run = 0
    batch_start = time.time()

    with raw_path.open("a", encoding="utf-8") as raw_f, \
         clean_path.open("a", encoding="utf-8") as clean_f:

        requeued_once = set()
        while queue:
            if time.time() - batch_start >= BATCH_MAX_SECONDS:
                log(f"BATCH STOP {name}: time cap hit, {len(queue)} still queued")
                break

            video_id = queue.popleft()
            text = download_captions(video_id)
            time.sleep(REQUEST_DELAY_S)

            if text is None:
                if video_id in requeued_once:
                    log(f"{video_id}: failed again after requeue, leaving for next run")
                    continue  # not marked done, next invocation of the script retries it
                log(f"{video_id}: failed {MAX_RETRIES}/{MAX_RETRIES} fetch attempts, "
                    f"pushed to end of queue, retrying after the rest")
                requeued_once.add(video_id)
                queue.append(video_id)
                continue

            if not text:
                entry["done_ids"].append(video_id)  # confirmed no caption track
                state_io.save_namespace(STATE_FILE, namespace, entry)
                continue

            raw_f.write(text + "\n\n")
            raw_f.flush()

            cleaned, dropped = clean_text(text)
            collapsed = collapse_repeats(cleaned) if cleaned else cleaned
            if collapsed != cleaned:
                entry["repeat_collapsed"] += 1
            cleaned = collapsed

            if cleaned and not is_likely_assamese(cleaned):
                log(f"SUSPECT-LANGUAGE (low ৰ/ৱ frequency, likely Bengali not Assamese): {video_id}")
                entry["suspect_language"] += 1
                cleaned = ""

            if cleaned:
                clean_f.write(cleaned + "\n\n")
                clean_f.flush()

            entry["done_ids"].append(video_id)
            entry["total_words"] += len(cleaned.split())
            entry["dropped_lines"] += dropped
            state_io.save_namespace(STATE_FILE, namespace, entry)

            processed_this_run += 1
            if processed_this_run % LOG_EVERY_N_VIDEOS == 0:
                log(f"{name}: {processed_this_run} videos this run, "
                    f"~{entry['total_words']} words total so far, "
                    f"{entry['dropped_lines']} non-Assamese lines dropped so far, "
                    f"{entry['repeat_collapsed']} repeat-collapsed")

    log(f"DONE this run {name}: {processed_this_run} new videos, "
        f"{len(entry['done_ids'])} total ever, ~{entry['total_words']} words total")


def main():
    requested = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    selected = [s for s in SOURCES if not requested or s["name"] in requested]
    if not selected:
        log(f"No matching sources for {requested!r}; known: {[s['name'] for s in SOURCES]}")
        return

    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for s in selected:
        process_source(s)

    log("youtube_captions run complete.")


if __name__ == "__main__":
    main()
