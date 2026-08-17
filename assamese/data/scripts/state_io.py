"""Shared, lock-protected access to assamese/data/.state.json.

Multiple collection scripts (OCR, scraper) run concurrently and checkpoint to
the same state file under separate top-level namespaces ("ocr", "scrape").
Each call takes an exclusive flock for the read-modify-write so one script's
checkpoint save can never clobber the other's in-flight namespace.
"""

import fcntl
import json


def _with_lock(state_file, fn):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read()
        data = json.loads(raw) if raw.strip() else {}
        result = fn(f, data)
        fcntl.flock(f, fcntl.LOCK_UN)
        return result


def load_namespace(state_file, namespace, default):
    return _with_lock(state_file, lambda f, data: data.get(namespace, default))


def save_namespace(state_file, namespace, value):
    def _save(f, data):
        data[namespace] = value
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()

    _with_lock(state_file, _save)
