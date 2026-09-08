"""Tests for the pipeline's target-week selection (main.resolve_week_from_starts).

The fixture is the real 2026 schedule's per-week first and last kickoff, taken
verbatim from the production `games` table. Only the earliest and latest game in
each week matter to the rule under test -- whether any game in a week has yet to
start -- so the fixture carries those rather than all 887 rows.

Run: python -m pytest tests/ -q   (or: python tests/test_week_resolution.py)
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import resolve_week_from_starts  # noqa: E402
from utils import get_cfb_week  # noqa: E402

# (first kickoff, last kickoff) per real 2026 display week, UTC, from `games`.
REAL_2026_WEEK_BOUNDS = [
    ("2026-08-29 16:00:00", "2026-08-30 02:00:00"),  # wk 0
    ("2026-09-03 22:00:00", "2026-09-07 23:30:00"),  # wk 1  -- Monday-night finish
    ("2026-09-11 00:00:00", "2026-09-13 03:00:00"),  # wk 2
    ("2026-09-17 23:30:00", "2026-09-20 03:00:00"),  # wk 3
    ("2026-09-24 23:30:00", "2026-09-27 02:30:00"),  # wk 4
    ("2026-10-02 00:00:00", "2026-10-04 02:30:00"),  # wk 5
    ("2026-10-07 00:00:00", "2026-10-11 02:30:00"),  # wk 6
    ("2026-10-13 23:00:00", "2026-10-18 02:30:00"),  # wk 7
    ("2026-10-20 23:00:00", "2026-10-25 01:30:00"),  # wk 8
    ("2026-10-27 23:00:00", "2026-11-01 02:30:00"),  # wk 9
    ("2026-11-04 00:00:00", "2026-11-08 03:30:00"),  # wk 10
    ("2026-11-10 05:00:00", "2026-11-15 04:00:00"),  # wk 11 -- latest kickoff of the season
    ("2026-11-17 05:00:00", "2026-11-22 03:30:00"),  # wk 12
    ("2026-11-24 05:00:00", "2026-11-29 02:00:00"),  # wk 13
    ("2026-12-12 20:00:00", "2026-12-12 20:00:00"),  # wk 15 -- Army-Navy, alone
]

STARTS = [datetime.fromisoformat(d) for pair in REAL_2026_WEEK_BOUNDS for d in pair]


def _resolve(now_iso):
    now = datetime.fromisoformat(now_iso)
    candidate = get_cfb_week(today=now.date(), season_start_override=None)
    return candidate, resolve_week_from_starts(candidate, STARTS, now, None)


def test_delayed_tuesday_run_yields_week_1_without_an_override():
    """The case PR #8 had to hand-pin with --week 1. This is what makes that pin redundant."""
    candidate, resolved = _resolve("2026-09-08 07:00:00")
    assert candidate == 2, "the date-derived week is 2 -- that is the bug"
    assert resolved == 1, f"should step back to the completed week 1, got {resolved}"


def test_normal_sunday_run_is_unchanged():
    """HAPPY-PATH BAR: an ordinary Sunday must resolve to exactly the date-derived week."""
    for now_iso, expected in [
        ("2026-09-13 07:00:00", 2),
        ("2026-09-20 07:00:00", 3),
        ("2026-10-04 07:00:00", 5),
        ("2026-11-01 07:00:00", 9),
        ("2026-11-22 07:00:00", 12),
        ("2026-11-29 07:00:00", 13),
    ]:
        candidate, resolved = _resolve(now_iso)
        assert candidate == expected, f"{now_iso}: date-derived {candidate} != {expected}"
        assert resolved == candidate, (
            f"{now_iso}: resolution changed a normal Sunday run from {candidate} to {resolved}"
        )


def test_latest_kickoff_of_the_season_does_not_trip_the_walk_back():
    """Week 11's last game starts 04:00 UTC, 3h before the 07:00 cron.

    This is the case that rules out a completion buffer: with one, this week would
    read as incomplete and regress to week 10.
    """
    candidate, resolved = _resolve("2026-11-15 07:00:00")
    assert candidate == 11
    assert resolved == 11, f"week 11 must not walk back, got {resolved}"


def test_run_before_any_game_has_started_keeps_the_candidate():
    """Nothing has kicked off, so no week qualifies -- the candidate must survive intact
    rather than the scan walking down to some earlier week or below 1."""
    now = datetime.fromisoformat("2026-08-25 07:00:00")
    assert resolve_week_from_starts(0, STARTS, now, None) == 0
    assert resolve_week_from_starts(1, STARTS, now, None) == 1
    assert resolve_week_from_starts(5, STARTS, now, None) == 5


def test_walk_back_is_capped():
    """A stale future-dated row must not drag the published week back indefinitely.

    Here weeks 5, 4 and 3 all look incomplete. With MAX_WALK_BACK_WEEKS=2 the scan gives
    up and keeps the date-derived week rather than silently publishing week 2's ratings.
    """
    now = datetime.fromisoformat("2026-10-04 07:00:00")
    # One not-yet-started row in each of weeks 5, 4 and 3.
    poisoned = STARTS + [
        datetime.fromisoformat("2026-09-19 12:00:00").replace(year=2027),
        datetime.fromisoformat("2026-09-26 12:00:00").replace(year=2027),
        datetime.fromisoformat("2026-10-03 12:00:00").replace(year=2027),
    ]
    assert resolve_week_from_starts(5, poisoned, now, None) == 5


def test_a_week_with_no_games_is_not_treated_as_complete():
    """Bucket 16 holds no regular-season games in 2026. A late-December run must not
    read that emptiness as 'complete' and republish it as a fresh week."""
    now = datetime.fromisoformat("2026-12-27 07:00:00")
    resolved = resolve_week_from_starts(16, STARTS, now, None)
    assert resolved != 16, "an empty week must never resolve as complete"
    assert resolved == 15, f"should fall back to Army-Navy week 15, got {resolved}"


def test_monday_night_kickoff_groups_with_the_preceding_weekend():
    """A Monday 8pm ET kickoff is 00:00 UTC Tuesday. Bucketing on the raw UTC date would
    push it into the next week, undoing the Tuesday->Monday boundary for exactly the
    games that boundary exists for. The real 2025 TCU/North Carolina Labor Day game."""
    from utils import football_day
    labor_day_night = datetime.fromisoformat("2025-09-02 00:00:00")   # Mon Sep 1, 8pm ET
    assert football_day(labor_day_night).isoformat() == "2025-09-01"
    assert get_cfb_week(football_day(labor_day_night), None) == get_cfb_week(
        datetime.fromisoformat("2025-08-30 16:00:00").date(), None
    ), "must land in the same week as the Saturday before it"


def test_tbd_kickoff_placeholders_are_not_dragged_back_a_day():
    """CFBD stores an unscheduled kickoff as midnight ET = 05:00 UTC. 2026 weeks 11-13 each
    open with one. A rollover window wide enough to swallow those would pull whole weeks
    backwards, so the window must stay under 5 hours."""
    from utils import football_day
    placeholder = datetime.fromisoformat("2026-11-17 05:00:00")       # Tue 00:00 ET
    assert football_day(placeholder).isoformat() == "2026-11-17"
    assert get_cfb_week(football_day(placeholder), None) == 12


def test_empty_schedule_falls_back_to_the_candidate():
    now = datetime.fromisoformat("2026-10-04 07:00:00")
    assert resolve_week_from_starts(5, [], now, None) == 5


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
