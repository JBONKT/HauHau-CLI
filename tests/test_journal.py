"""The journal store: idempotent sync, keyword search, and the windows queries."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from hauhau.journal import Entry, Journal, _fts_query

DAY = 86400.0


class JournalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.journal = Journal(Path(self._tmp.name) / "journal.db")
        self.now = time.time()

    def tearDown(self) -> None:
        self.journal.close()
        self._tmp.cleanup()

    def entry(self, **kwargs) -> Entry:
        base = {
            "source": "mail",
            "external_id": "m1",
            "kind": "email",
            "ts": self.now,
            "title": "Invoice from Acme",
            "people": ["billing@acme.test"],
        }
        base.update(kwargs)
        return Entry(**base)


class TestEntries(JournalTestCase):
    def test_a_new_row_is_counted(self) -> None:
        self.assertEqual(self.journal.add([self.entry()]), 1)

    def test_the_same_row_twice_is_counted_once(self) -> None:
        # A sync every 15 minutes must be free once the row exists.
        self.journal.add([self.entry()])
        self.assertEqual(self.journal.add([self.entry()]), 0)
        self.assertEqual(self.journal.stats()["total"], 1)

    def test_an_unknown_source_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.entry(source="telegram")

    def test_an_unknown_kind_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.entry(kind="poem")

    def test_an_entry_without_an_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.entry(external_id="")

    def test_two_sources_may_share_an_id(self) -> None:
        self.journal.add([self.entry()])
        self.journal.add([self.entry(source="messages", kind="message")])
        self.assertEqual(self.journal.stats()["total"], 2)


class TestSearch(JournalTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.journal.add(
            [
                self.entry(external_id="m1", title="Invoice from Acme", ts=self.now - DAY),
                self.entry(
                    external_id="m2",
                    title="Lunch on Friday",
                    people=["dana@example.test"],
                    ts=self.now,
                ),
                self.entry(
                    source="messages", kind="message", external_id="s1", title="on my way",
                    people=["+46701234567"], ts=self.now,
                ),
            ]
        )

    def test_a_word_match_finds_the_row(self) -> None:
        hits = self.journal.search("invoice")
        self.assertEqual(len(hits), 1)
        self.assertIn("Invoice", hits[0].title)

    def test_the_prefix_matches_a_longer_word(self) -> None:
        self.assertTrue(self.journal.search("invoi"))

    def test_a_source_filter_narrows_it(self) -> None:
        self.assertEqual(len(self.journal.search("my way", sources=["messages"])), 1)
        self.assertEqual(len(self.journal.search("my way", sources=["mail"])), 0)

    def test_punctuation_does_not_raise(self) -> None:
        # A raw FTS5 query would treat these as syntax.
        for query in ("AND ( *", "!!!", "-", '"'):
            self.assertIsInstance(self.journal.search(query), list)

    def test_no_match_is_an_empty_list(self) -> None:
        self.assertEqual(self.journal.search("zebra"), [])

    def test_person_finds_both_directions(self) -> None:
        self.assertEqual(len(self.journal.person("dana@example.test")), 1)

    def test_since_excludes_the_old_row(self) -> None:
        hits = self.journal.search("acme", since=self.now - 3600)
        self.assertEqual(hits, [])

    def test_fts_query_quotes_every_token(self) -> None:
        self.assertEqual(_fts_query("a b"), '"a"* OR "b"*')

    def test_fts_query_of_punctuation_is_empty(self) -> None:
        self.assertEqual(_fts_query("!!!" * 3), "")


class TestWindows(JournalTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.journal.add(
            [
                self.entry(external_id="old", ts=self.now - 10 * DAY),
                self.entry(external_id="now", ts=self.now),
                self.entry(
                    source="calendar", kind="event", external_id="c1", ts=self.now + DAY,
                    title="Standup",
                ),
            ]
        )

    def test_between_is_half_open(self) -> None:
        rows = self.journal.between(self.now - 1, self.now + 1)
        self.assertEqual([r.title for r in rows], ["Invoice from Acme"])

    def test_between_can_filter_by_source(self) -> None:
        rows = self.journal.between(self.now - 1, self.now + 2 * DAY, sources=["calendar"])
        self.assertEqual([r.title for r in rows], ["Standup"])

    def test_latest_ignores_the_future(self) -> None:
        # A reminder dated tomorrow must not make "last event" look negative.
        self.assertAlmostEqual(self.journal.latest(self.now).ts, self.now, places=3)

    def test_latest_before_anything_is_none(self) -> None:
        self.assertIsNone(self.journal.latest(self.now - 30 * DAY))

    def test_stats_count_each_source(self) -> None:
        stats = self.journal.stats()
        self.assertEqual(stats["total"], 3)
        self.assertEqual(dict(stats["by_source"]), {"calendar": 1, "mail": 2})


class TestCursors(JournalTestCase):
    def test_a_missing_cursor_is_none(self) -> None:
        self.assertIsNone(self.journal.cursor("mail"))

    def test_a_cursor_round_trips_and_updates(self) -> None:
        self.journal.set_cursor("mail", "mail:100")
        self.assertEqual(self.journal.cursor("mail"), "mail:100")
        self.journal.set_cursor("mail", "mail:200")
        self.assertEqual(self.journal.cursor("mail"), "mail:200")
        self.assertEqual(self.journal.stats()["cursors"], {"mail": "mail:200"})


if __name__ == "__main__":
    unittest.main()
