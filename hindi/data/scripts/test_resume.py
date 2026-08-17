#!/usr/bin/env python3
"""Self-test: proves ocr_pipeline's checkpoint/resume actually skips finished work.

Runs against a throwaway state file + output dir (never touches the real
hindi/data/.state.json), so it's safe to run any time without disturbing a
live collection run. Stubs out the network download so the test is fast and
deterministic; the invariant under test is the checkpoint logic itself
(state persisted after each item -> already-"done" items are never
re-downloaded on the next pass), not NCERT's live availability.
"""

import tempfile
from pathlib import Path
from unittest import mock

import ocr_pipeline as ocr


def fake_download_factory(call_log):
    def fake_download(code, tmp_dir):
        call_log.append(code)
        pdf_path = tmp_dir / f"{code}.pdf"
        pdf_path.write_bytes(b"%PDF-fake")
        return pdf_path

    return fake_download


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        state_file = tmp_dir / ".state.json"
        out_dir = tmp_dir / "ocr_out"
        out_dir.mkdir()
        clean_dir = tmp_dir / "ocr_clean"
        clean_dir.mkdir()

        items = [("bookA", "bookA01"), ("bookA", "bookA02"), ("bookB", "bookB01")]
        calls = []

        with mock.patch.object(ocr, "STATE_FILE", state_file), \
             mock.patch.object(ocr, "OUT_DIR", out_dir), \
             mock.patch.object(ocr, "CLEAN_DIR", clean_dir), \
             mock.patch.object(ocr, "download_pdf", side_effect=fake_download_factory(calls)), \
             mock.patch.object(ocr, "extract_pdftotext", return_value="यह एक परीक्षण वाक्य है।"), \
             mock.patch.object(ocr, "log", lambda *a, **k: None):

            # --- "session 1": process the first two items, then interrupt ---
            state = ocr.load_state()
            ocr.process_item(*items[0], state, tmp_dir)
            ocr.process_item(*items[1], state, tmp_dir)
            assert calls == ["bookA01", "bookA02"], f"unexpected downloads: {calls}"
            assert state["bookA01"]["status"] == "done"
            assert state["bookA02"]["status"] == "done"

            # --- "restart": fresh load_state() call, as a real rerun would do ---
            calls.clear()
            state2 = ocr.load_state()
            assert state2["bookA01"]["status"] == "done", "checkpoint did not persist across reload"
            assert state2["bookA02"]["status"] == "done"

            # re-running the full item list must skip the two done items...
            for prefix, code in items:
                ocr.process_item(prefix, code, state2, tmp_dir)
            assert calls == ["bookB01"], (
                f"resume redid finished work or skipped new work: {calls}"
            )
            assert state2["bookB01"]["status"] == "done"

        raw_content = (out_dir / "bookA.txt").read_text(encoding="utf-8")
        clean_content = (clean_dir / "bookA.txt").read_text(encoding="utf-8")
        assert "परीक्षण" in raw_content, "raw text missing from raw output file"
        assert "परीक्षण" in clean_content, "cleaned Devanagari text missing from clean output file"

    print("test_resume: OK -- resume skips completed items, processes new ones, "
          "state persists across reload")


if __name__ == "__main__":
    main()
