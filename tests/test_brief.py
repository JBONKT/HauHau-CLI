"""The digest: counts since midnight, what is due, and a readable shape."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from hauhau.brief import compose, counts, midnight
from hauhau.journal import Entry, Journal

DAY = 86400.0


class BriefTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.journal = Journal(Path(self._tmp.name) / "journal.db")
        self.now = time.time()
        self.start = midnight(self.now)

    def tearDown(self) -> None:
        self.journal.close()
        self._tmp.cleanup()

    def add(self, *entries) -> None:
        self.journal.add(list(entries))


class TestMidnight(BriefTestCase):
    def test_midnight_is_today_at_zero(self) -> None:
        shape = time.localtime(midnight(self.now))
        self.assertEqual((shape.tm_hour, shape.tm_min, shape.tm_sec), (0, 0, 0))

    def test_midnight_is_idempotent(self) -> None:
        self.assertEqual(midnight(self.now), midnight(midnight(self.now) + 3600))


class TestCompose(BriefTestCase):
    def test_an_empty_journal_says_so(self) -> None:
        text = compose(self.journal, None, now=self.now)
        self.assertIn("nothing new", text)
        self.assertIn("run a sync", text)

    def test_today_counts_are_summarised(self) -> None:
        self.add(
            Entry("mail", "m1", "email", self.start + 60, "Invoice", people=["a@b.test"]),
            Entry("messages", "s1", "message", self.start + 120, "ok", people=["+46"]),
        )
        text = compose(self.journal, None, now=self.now)
        self.assertIn("1 mail, 1 message", text)

    def test_yesterdays_mail_is_not_reported_as_today(self) -> None:
        self.add(Entry("mail", "m1", "email", self.start - DAY, "Old"))
        text = compose(self.journal, None, now=self.now)
        self.assertIn("nothing new", text)

    def test_a_reminder_due_soon_is_listed(self) -> None:
        self.add(Entry("reminders", "r1", "reminder", self.now + DAY, "Pay the invoice"))
        text = compose(self.journal, None, now=self.now)
        self.assertIn("Pay the invoice", text)

    def test_a_completed_reminder_is_not_listed(self) -> None:
        self.add(
            Entry("reminders", "r1", "reminder", self.now + DAY, "Done already",
                  meta={"completed": True})
        )
        self.assertNotIn("Done already", compose(self.journal, None, now=self.now))

    def test_an_upcoming_event_is_listed(self) -> None:
        self.add(Entry("calendar", "c1", "event", self.now + 3600, "Standup"))
        self.assertIn("Standup", compose(self.journal, None, now=self.now))

    def test_the_future_does_not_make_the_age_negative(self) -> None:
        # A reminder dated tomorrow used to print "last one -24.0h ago".
        self.add(
            Entry("mail", "m1", "email", self.now - 3600, "Recent"),
            Entry("reminders", "r1", "reminder", self.now + 2 * DAY, "Later"),
        )
        text = compose(self.journal, None, now=self.now)
        last = next(line for line in text.splitlines() if "last one" in line)
        self.assertIn("last one 1.0h ago", last)

    def test_counts_only_the_rows_given(self) -> None:
        self.add(
            Entry("mail", "m1", "email", self.now, "a"),
            Entry("mail", "m2", "email", self.now, "b"),
        )
        rows = self.journal.between(self.start, self.now + 1)
        self.assertEqual(counts(rows), {"mail": 2})


if __name__ == "__main__":
    unittest.main()
