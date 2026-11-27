"""Reading the Mac's records: parsers, permission failures, and the sync loop."""

from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from hauhau import ingest
from hauhau.ingest import (
    APPLE_EPOCH,
    Fetch,
    SourceError,
    _cursor_epoch,
    fetch_calls,
    fetch_messages,
    iso_to_epoch,
    rows_of,
    sync,
)
from hauhau.journal import Entry, Journal


class TestRowsOf(unittest.TestCase):
    def test_tab_separated_rows_are_parsed(self) -> None:
        out = "1\ta@b.test\tHello\t2026-11-26T10:00:00\n"
        self.assertEqual(rows_of(out, 4), [["1", "a@b.test", "Hello", "2026-11-26T10:00:00"]])

    def test_a_short_row_is_skipped_not_fatal(self) -> None:
        # One odd mail must not cost the other 300.
        out = "1\ta@b.test\tHello\t2026-11-26T10:00:00\n2\tonly-three\tfields\n"
        self.assertEqual(len(rows_of(out, 4)), 1)

    def test_blank_output_is_no_rows(self) -> None:
        self.assertEqual(rows_of("\n\n", 4), [])

    def test_extra_fields_are_dropped(self) -> None:
        self.assertEqual(rows_of("a\tb\tc\td\te\n", 4), [["a", "b", "c", "d"]])


class TestIsoToEpoch(unittest.TestCase):
    def test_isot_string_is_understood(self) -> None:
        stamp = iso_to_epoch("2026-11-26T10:30:00")
        self.assertEqual(time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp)), "2026-11-26 10:30")

    def test_a_trailing_z_means_utc_not_local_time(self) -> None:
        from datetime import UTC, datetime

        expected = datetime(2026, 11, 26, 10, 30, tzinfo=UTC).timestamp()
        self.assertEqual(iso_to_epoch("2026-11-26T10:30:00Z"), expected)

    def test_a_numeric_offset_is_honored(self) -> None:
        from datetime import datetime, timedelta, timezone

        expected = datetime(2026, 11, 26, 10, 30, tzinfo=timezone(timedelta(hours=2))).timestamp()
        self.assertEqual(iso_to_epoch("2026-11-26T10:30:00+02:00"), expected)

    def test_unparseable_text_falls_back_to_now(self) -> None:
        self.assertAlmostEqual(iso_to_epoch("not a date"), time.time(), delta=5)

    def test_empty_text_falls_back_to_now(self) -> None:
        self.assertAlmostEqual(iso_to_epoch(""), time.time(), delta=5)


class TestOpenReadOnly(unittest.TestCase):
    def test_a_missing_database_names_the_path(self) -> None:
        with self.assertRaises(SourceError) as caught:
            ingest._open_ro(Path("/nonexistent/chat.db"))
        self.assertIn("/nonexistent/chat.db", str(caught.exception))

    def test_a_refused_database_explains_full_disk_access(self) -> None:
        # A directory is not a database: sqlite refuses it the same way macOS
        # refuses chat.db without Full Disk Access.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SourceError) as caught:
                ingest._open_ro(Path(tmp))
            self.assertIn("Full Disk Access", str(caught.exception))


def make_chat_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY, date INTEGER, is_from_me INTEGER,
            text TEXT, handle_id INTEGER
        );
        INSERT INTO handle (ROWID, id) VALUES (1, '+46701234567');
        """
    )
    now = time.time()
    def apple(stamp: float) -> int:
        return int((stamp - APPLE_EPOCH) * 1_000_000_000)
    conn.execute(
        "INSERT INTO message VALUES (101, ?, 0, 'on my way', 1)", (apple(now - 60),)
    )
    conn.execute(
        "INSERT INTO message VALUES (102, ?, 1, NULL, 1)", (apple(now - 30),)
    )
    conn.commit()
    conn.close()


class TestFetchMessages(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "chat.db"
        make_chat_db(self.db)
        self._original = ingest.MESSAGES_DB
        ingest.MESSAGES_DB = self.db

    def tearDown(self) -> None:
        ingest.MESSAGES_DB = self._original
        self._tmp.cleanup()

    def test_messages_are_read_with_their_handle(self) -> None:
        fetched = fetch_messages()
        self.assertEqual(len(fetched.entries), 2)
        first = fetched.entries[0]
        self.assertEqual(first.source, "messages")
        self.assertEqual(first.people, ["+46701234567"])
        self.assertEqual(first.title, "on my way")
        self.assertAlmostEqual(first.ts, time.time(), delta=120)

    def test_a_message_without_text_is_still_journalled(self) -> None:
        fetched = fetch_messages()
        self.assertIn("attachment or rich text", fetched.entries[-1].body)

    def test_the_cursor_is_the_newest_timestamp(self) -> None:
        fetched = fetch_messages()
        self.assertTrue(fetched.cursor.startswith("msg:"))
        self.assertIsNotNone(_cursor_epoch(fetched.cursor))

    def test_a_cursor_skips_what_was_already_read(self) -> None:
        fetched = fetch_messages(since=time.time() - 45)
        self.assertEqual(len(fetched.entries), 1, "only the newer message is new")


def make_calls_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE Z_CALLRECORD (
            Z_PK INTEGER PRIMARY KEY, ZDATE REAL, ZDURATION REAL,
            ZADDRESS TEXT, ZORIGINATED INTEGER, ZCALLTYPE INTEGER
        );
        """
    )
    conn.execute(
        "INSERT INTO Z_CALLRECORD VALUES (7, ?, 42.0, '+46701234567', 1, 1)",
        (time.time() - 300 - APPLE_EPOCH,),
    )
    conn.commit()
    conn.close()


class TestFetchCalls(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "CallHistory.storedata"
        make_calls_db(self.db)
        self._original = ingest.CALLS_DB
        ingest.CALLS_DB = self.db

    def tearDown(self) -> None:
        ingest.CALLS_DB = self._original
        self._tmp.cleanup()

    def test_a_call_is_journalled_with_direction_and_length(self) -> None:
        fetched = fetch_calls()
        self.assertEqual(len(fetched.entries), 1)
        entry = fetched.entries[0]
        self.assertEqual(entry.kind, "call")
        self.assertIn("outgoing", entry.title)
        self.assertIn("42s", entry.title)
        self.assertEqual(entry.people, ["+46701234567"])


class TestCursorEpoch(unittest.TestCase):
    def test_a_prefixed_cursor_yields_its_epoch(self) -> None:
        self.assertEqual(_cursor_epoch("msg:1795700000"), 1795700000.0)

    def test_garbage_is_none(self) -> None:
        for value in (None, "", "nonsense", "msg:soon"):
            self.assertIsNone(_cursor_epoch(value))


class TestEnsureApp(unittest.TestCase):
    """A quit app must be started hidden instead of answering -600."""

    def test_it_calls_open_with_the_background_flags(self) -> None:
        calls: list[list[str]] = []
        original = ingest.subprocess.run

        def fake(cmd, **kwargs):
            calls.append(cmd)

            class Done:
                # pgrep reports "not running" so the launch path is taken.
                returncode = 1 if cmd[0] == "pgrep" else 0

            return Done()

        ingest.subprocess.run = fake
        try:
            ingest._ensure_app("Calendar")
        finally:
            ingest.subprocess.run = original
        self.assertEqual(calls[0][:2], ["pgrep", "-qx"])
        self.assertEqual(calls[1][:3], ["open", "-gja", "Calendar"])

    def test_a_running_app_is_not_launched_or_waited_for(self) -> None:
        calls: list[list[str]] = []
        original = ingest.subprocess.run

        def fake(cmd, **kwargs):
            calls.append(cmd)

            class Done:
                returncode = 0

            return Done()

        ingest.subprocess.run = fake
        try:
            with mock.patch.object(ingest.time, "sleep") as sleep:
                ingest._ensure_app("Mail")
        finally:
            ingest.subprocess.run = original
        self.assertEqual([call[0] for call in calls], ["pgrep"], "nothing to launch")
        sleep.assert_not_called()

    def test_a_missing_open_command_is_not_fatal(self) -> None:
        original = ingest.subprocess.run

        def boom(*_args, **_kwargs):
            raise OSError("no open here")

        ingest.subprocess.run = boom
        try:
            ingest._ensure_app("Mail")  # must not raise
        finally:
            ingest.subprocess.run = original


class TestSyncLoop(unittest.TestCase):
    """A broken source must be reported, not crash the other four."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.journal = Journal(Path(self._tmp.name) / "journal.db")
        self._original = dict(ingest.FETCHERS)

    def tearDown(self) -> None:
        ingest.FETCHERS.clear()
        ingest.FETCHERS.update(self._original)
        self.journal.close()
        self._tmp.cleanup()

    def test_a_good_source_is_stored_and_cursored(self) -> None:
        def fake_mail(days=7, limit=300):
            return Fetch(
                [Entry("mail", "mail:1", "email", time.time(), "Hello")], "mail:1795700000"
            )

        ingest.FETCHERS.clear()
        ingest.FETCHERS["mail"] = fake_mail
        report = sync(self.journal)
        self.assertEqual(report["mail"], {"fetched": 1, "added": 1})
        self.assertEqual(self.journal.stats()["total"], 1)
        self.assertEqual(self.journal.cursor("mail"), "mail:1795700000")

    def test_one_failing_source_does_not_stop_the_others(self) -> None:
        def broken(**kwargs):
            raise SourceError("grant Full Disk Access")

        def fine(**kwargs):
            return Fetch([Entry("calendar", "cal:1", "event", time.time(), "Standup")], None)

        ingest.FETCHERS.clear()
        ingest.FETCHERS["messages"] = broken
        ingest.FETCHERS["calendar"] = fine
        report = sync(self.journal)
        self.assertIn("Full Disk Access", report["messages"]["error"])
        self.assertEqual(report["calendar"]["added"], 1)

    def test_an_unexpected_exception_is_reported_too(self) -> None:
        def explode(**kwargs):
            raise KeyError("boom")

        ingest.FETCHERS.clear()
        ingest.FETCHERS["calls"] = explode
        report = sync(self.journal)
        self.assertIn("KeyError", report["calls"]["error"])

    def test_a_second_sync_adds_nothing(self) -> None:
        def fake(**kwargs):
            return Fetch([Entry("mail", "mail:1", "email", time.time(), "Hello")], "mail:1")

        ingest.FETCHERS.clear()
        ingest.FETCHERS["mail"] = fake
        sync(self.journal)
        report = sync(self.journal)
        self.assertEqual(report["mail"]["added"], 0)
        self.assertEqual(self.journal.stats()["total"], 1)

    def test_a_failed_source_leaves_a_note_for_status(self) -> None:
        def broken(**_kwargs):
            raise SourceError("grant Full Disk Access")

        ingest.FETCHERS.clear()
        ingest.FETCHERS["messages"] = broken
        sync(self.journal)
        self.assertFalse(self.journal.states()["messages"]["ok"])

    def test_a_working_source_records_ok(self) -> None:
        def fine(**_kwargs):
            return Fetch([], None)

        ingest.FETCHERS.clear()
        ingest.FETCHERS["mail"] = fine
        sync(self.journal)
        self.assertTrue(self.journal.states()["mail"]["ok"])

    def test_only_the_named_sources_run(self) -> None:
        seen: list[str] = []

        def tracker(name: str):
            def fetch(**kwargs):
                seen.append(name)
                return Fetch([], None)

            return fetch

        ingest.FETCHERS.clear()
        ingest.FETCHERS["mail"] = tracker("mail")
        ingest.FETCHERS["calendar"] = tracker("calendar")
        sync(self.journal, ["calendar"])
        self.assertEqual(seen, ["calendar"])


if __name__ == "__main__":
    unittest.main()
