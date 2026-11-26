"""The idle watchdog: when to push, and what it types where."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nudge.py"
_spec = importlib.util.spec_from_file_location("nudge", _SCRIPT)
nudge = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(nudge)


def record(ts: float) -> dict:
    return {"ts": ts}


class TestGuard(unittest.TestCase):
    def test_no_history_is_allowed(self):
        self.assertIsNone(nudge.guard([], now=1000.0, min_gap=240.0, max_per_hour=8))

    def test_a_push_within_the_gap_is_refused(self):
        reason = nudge.guard([record(900.0)], now=1000.0, min_gap=240.0, max_per_hour=8)
        self.assertIn("last push", reason)

    def test_the_gap_expires(self):
        self.assertIsNone(nudge.guard([record(700.0)], now=1000.0, min_gap=240.0, max_per_hour=8))

    def test_the_hourly_cap_stops_a_loop(self):
        stamps = [record(1000.0 - 300 * i) for i in range(1, 9)]
        reason = nudge.guard(stamps, now=1000.0, min_gap=240.0, max_per_hour=8)
        self.assertIn("in the last hour", reason)

    def test_old_pushes_do_not_count_toward_the_cap(self):
        stamps = [record(1000.0 - 4000 - 10 * i) for i in range(20)]
        self.assertIsNone(nudge.guard(stamps, now=1000.0, min_gap=240.0, max_per_hour=8))


class TestApplescript(unittest.TestCase):
    def test_it_targets_the_tab_owning_the_tty(self):
        script = nudge.applescript_for("/dev/ttys003", "continue")
        self.assertIn('tty of tab i of window w) is "/dev/ttys003"', script)
        self.assertIn('do script "continue" in tab i of window w', script)

    def test_quotes_and_backslashes_survive(self):
        script = nudge.applescript_for("/dev/ttys001", 'say "hi" \\ done')
        self.assertIn('do script "say \\"hi\\" \\\\ done"', script)


class TestLastActivity(unittest.TestCase):
    def test_missing_files_do_not_count(self):
        self.assertEqual(nudge.last_activity([Path("/nonexistent/nope.log")]), 0.0)

    def test_the_newest_file_wins(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            older = Path(tmp) / "older.log"
            newer = Path(tmp) / "newer.log"
            older.write_text("a")
            newer.write_text("b")
            import os

            os.utime(older, (1000, 1000))
            os.utime(newer, (2000, 2000))
            self.assertEqual(nudge.last_activity([older, newer]), 2000.0)


class TestLogParsing(unittest.TestCase):
    def test_a_missing_log_reads_as_empty(self):
        self.assertEqual(nudge.read_log(Path("/nonexistent/nudge.jsonl")), [])

    def test_broken_lines_are_skipped(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nudge.jsonl"
            path.write_text('{"ts": 1}\nnot json\n{"ts": 2}\n')
            self.assertEqual(nudge.read_log(path), [{"ts": 1}, {"ts": 2}])


if __name__ == "__main__":
    unittest.main()
