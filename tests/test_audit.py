from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hauhau.audit import Audit


class AuditTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "logs" / "audit.jsonl"
        self.audit = Audit(self.path, "session-1")

    def tearDown(self) -> None:
        self.audit.close()
        self._tmp.cleanup()


class TestWrite(AuditTestCase):
    def test_one_json_object_per_line(self) -> None:
        self.audit.log("tool", name="shell", ok=True)
        self.audit.log("tool", name="grep", ok=False)
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            record = json.loads(line)
            self.assertEqual(record["session"], "session-1")
            self.assertEqual(record["event"], "tool")
            self.assertIn("ts", record)

    def test_fields_that_clash_with_the_event_key_are_kept(self) -> None:
        # rag_write logs its own "kind"; the parameter is named "event" so it fits.
        self.audit.log("rag_write", kind="procedural", id=7)
        record = json.loads(self.path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(record["kind"], "procedural")
        self.assertEqual(record["event"], "rag_write")

    def test_awkward_values_are_serialised_not_fatal(self) -> None:
        self.audit.log("odd", path=Path("/tmp/x"), value=float("nan"), blob=b"\x00\x01")
        record = json.loads(self.path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(record["path"], "/tmp/x")
        self.assertIn("blob", record)

    def test_the_file_is_private(self) -> None:
        mode = self.path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)


class TestTail(AuditTestCase):
    def _seed(self, count: int) -> None:
        for index in range(count):
            self.audit.log("e", n=index)

    def test_tail_returns_the_last_n_in_order(self) -> None:
        self._seed(5)
        entries = self.audit.tail(2)
        self.assertEqual([entry["n"] for entry in entries], [3, 4])

    def test_zero_is_one_not_everything(self) -> None:
        # lines[-0:] is the whole file, so /audit 0 dumped the entire log.
        self._seed(5)
        self.assertEqual(len(self.audit.tail(0)), 1)

    def test_a_negative_count_cannot_slice_from_the_front(self) -> None:
        self._seed(5)
        self.assertEqual(len(self.audit.tail(-3)), 1)

    def test_more_than_exists_is_fine(self) -> None:
        self._seed(2)
        self.assertEqual(len(self.audit.tail(50)), 2)

    def test_a_missing_file_is_empty(self) -> None:

        self.assertEqual(Audit(Path(self._tmp.name) / "none.jsonl", "s").tail(3), [])

    def test_corrupt_lines_are_skipped(self) -> None:
        self._seed(2)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("{not json}\n")
        entries = self.audit.tail(5)
        self.assertEqual([entry["n"] for entry in entries], [0, 1])


class TestDisabled(unittest.TestCase):
    def test_a_disabled_audit_writes_nothing_and_closes_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            audit = Audit(path, "s", enabled=False)
            audit.log("tool", name="shell")
            audit.close()
            self.assertFalse(path.exists(), "nothing may be written when audit is off")
            self.assertEqual(audit.tail(5), [])


if __name__ == "__main__":
    unittest.main()
