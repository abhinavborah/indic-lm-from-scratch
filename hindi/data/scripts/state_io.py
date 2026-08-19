"""Shared, lock-protected access to hindi/data/.state.json.

Multiple collection scripts (OCR, scraper) run concurrently and checkpoint to
the same state file under separate top-level namespaces ("ocr", "scrape").
Each call takes an exclusive flock (on a separate, never-replaced .lock file)
for the read-modify-write, and writes are committed via a temp-file + atomic
os.replace() rather than truncate-in-place -- a kill mid-write (this machine
sleeps often and has killed background jobs before) can no longer leave
.state.json empty and wipe every namespace's checkpoint progress.
"""

import fcntl
import json
import os
import tempfile


def _with_lock(state_file, fn):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = state_file.with_suffix(state_file.suffix + ".lock")
    with open(lock_file, "a+") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            raw = state_file.read_text(encoding="utf-8") if state_file.exists() else ""
            data = json.loads(raw) if raw.strip() else {}
            return fn(data)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def load_namespace(state_file, namespace, default):
    return _with_lock(state_file, lambda data: data.get(namespace, default))


def save_namespace(state_file, namespace, value):
    def _save(data):
        data[namespace] = value
        tmp_fd, tmp_path = tempfile.mkstemp(dir=state_file.parent, prefix=".state.", suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_f:
                json.dump(data, tmp_f, indent=2, ensure_ascii=False)
                tmp_f.flush()
                os.fsync(tmp_f.fileno())
            os.replace(tmp_path, state_file)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    _with_lock(state_file, _save)
