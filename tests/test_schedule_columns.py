"""Regression tests for the Season Grid's week-column derivation and
conference-championship identification.

Every fixture below is REAL data pulled from the production `games` table
(game_ids and start_dates are verbatim), reduced to the smallest set that
exercises the rule under test. The three bugs these pin down all shipped to
production in the first Season Grid pass:

  1. An empty "Week 15" column, whose only occupant (Army-Navy) is diverted
     into its own dedicated slot.
  2. A confidently-wrong "Wake Forest vs Duke" ACC championship matchup,
     invented by tie-breaking six rivalry-week games that share a start_date.
  3. Hardcoded postseason column labels, which are only correct for a season
     whose championship weekend happens to land on week 14.

Run: python -m pytest tests/ -q   (or: python tests/test_schedule_columns.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifacts.schedule import (  # noqa: E402
    build_canonical_columns,
    identify_army_navy_game,
    identify_conference_championship_games,
)
from utils import get_cfb_week  # noqa: E402


def _game(game_id, season, home, away, start_date, conference="ACC", conference_game=True,
          season_type="regular"):
    """Expand one game into the two team-oriented rows schedule_grid emits."""
    common = dict(game_id=game_id, season=season, season_type=season_type,
                  conference=conference, conference_game=conference_game,
                  start_date=start_date)
    return [
        dict(team=home, opponent=away, home_away="home", **common),
        dict(team=away, opponent=home, home_away="away", **common),
    ]


def _rows(*games):
    out = []
    for g in games:
        out.extend(_game(*g))
    return out


# --- Real ACC 2025: a completed season with a genuine championship game ------
# The Dec 7 game (401777328, Virginia vs Duke) is the real 2025 ACC title game;
# it sits alone in its week bucket, one bucket later than the Nov 29-30 slate.
ACC_2025 = _rows(
    (401754531, 2025, "Wake Forest", "NC State", "2025-09-11 23:30:00"),
    (401754623, 2025, "Georgia Tech", "Clemson", "2025-09-13 16:00:00"),
    (401754611, 2025, "Pittsburgh", "Miami", "2025-11-29 17:00:00"),
    (401754610, 2025, "Duke", "Wake Forest", "2025-11-29 20:30:00"),
    (401754613, 2025, "Virginia", "Virginia Tech", "2025-11-30 00:00:00"),
    (401777328, 2025, "Virginia", "Duke", "2025-12-07 01:00:00"),
)

# --- Real ACC 2026: in progress, no championship game scheduled yet ----------
# Six conference games share the 2026-11-28 05:00 rivalry-week start_date. The
# old max(start_date) rule tie-broke these alphabetically and published
# "Wake Forest vs Duke" as the ACC championship matchup.
ACC_2026 = _rows(
    (401858202, 2026, "Virginia", "NC State", "2026-08-29 19:30:00"),
    (401858206, 2026, "Stanford", "Miami", "2026-09-05 01:00:00"),
    (401858212, 2026, "Florida State", "SMU", "2026-09-07 23:30:00"),
    (401858315, 2026, "California", "Pittsburgh", "2026-11-28 05:00:00"),
    (401858313, 2026, "North Carolina", "NC State", "2026-11-28 05:00:00"),
    (401858317, 2026, "Virginia Tech", "Virginia", "2026-11-28 05:00:00"),
    (401858311, 2026, "Miami", "Boston College", "2026-11-28 05:00:00"),
    (401858312, 2026, "Wake Forest", "Duke", "2026-11-28 05:00:00"),
    (401858316, 2026, "Stanford", "SMU", "2026-11-28 05:00:00"),
)

# Real 2026 Army-Navy, the sole game in its week bucket.
ARMY_NAVY_2026 = _rows(
    (401862844, 2026, "Army", "Navy", "2026-12-12 20:00:00", "American Athletic", True),
)


def test_week_boundary_groups_monday_with_preceding_weekend():
    """The 2026-09-07 Monday FSU/SMU game belongs to week 1, not week 2."""
    from datetime import date
    assert get_cfb_week(date(2026, 9, 5), None) == 1, "Saturday of week 1"
    assert get_cfb_week(date(2026, 9, 6), None) == 1, "Sunday of week 1"
    assert get_cfb_week(date(2026, 9, 7), None) == 1, "Monday night game -- the fix"
    assert get_cfb_week(date(2026, 9, 8), None) == 2, "Tuesday starts week 2"
    assert get_cfb_week(date(2026, 8, 29), None) == 0, "Week 0 Saturday still week 0"


def test_championship_identified_on_a_completed_season():
    """REGRESSION BAR: a real title game must still be found."""
    found = identify_conference_championship_games(ACC_2025, 2025)
    assert found == {"ACC": 401777328}, found


def test_no_championship_invented_mid_season():
    """Six games tied in one bucket is a rivalry slate, not a championship."""
    found = identify_conference_championship_games(ACC_2026, 2026)
    assert found == {}, f"invented a championship matchup: {found}"


def test_single_bucket_conference_identifies_nothing():
    """A conference whose games all sit in one bucket (2024 Pac-12) yields nothing."""
    only_one_bucket = _rows(
        (1, 2024, "Oregon State", "Washington State", "2024-11-30 20:00:00", "Pac-12", True),
    )
    assert identify_conference_championship_games(
        only_one_bucket, 2024, {"Pac-12": "two-team remnant"}
    ) == {}


def test_no_phantom_column_for_a_diverted_army_navy_game():
    rows = ACC_2026 + ARMY_NAVY_2026
    army_navy_id = identify_army_navy_game(rows, 2026)
    assert army_navy_id == 401862844

    champ_ids = set(identify_conference_championship_games(rows, 2026).values())
    columns = build_canonical_columns(rows, 2026, champ_ids, army_navy_id)
    slot_ids = [slot_id for slot_id, _ in columns]

    assert "week-15" not in slot_ids, "Army-Navy's bucket must not mint an empty week column"
    assert "army-navy" in slot_ids, "the game still needs its dedicated slot"


def test_postseason_labels_are_derived_not_hardcoded():
    """2026's championship weekend is week 14; 2025's is week 15."""
    rows_2026 = ACC_2026 + ARMY_NAVY_2026
    army_navy_id = identify_army_navy_game(rows_2026, 2026)
    labels_2026 = dict(build_canonical_columns(rows_2026, 2026, set(), army_navy_id))
    assert labels_2026["conf-championship"] == "Week 14 (CCG)", labels_2026["conf-championship"]
    assert labels_2026["army-navy"] == "Week 15", labels_2026["army-navy"]
    assert labels_2026["cfp-r1-bowls"] == "Bowls"

    champ_2025 = set(identify_conference_championship_games(ACC_2025, 2025).values())
    labels_2025 = dict(build_canonical_columns(ACC_2025, 2025, champ_2025, None))
    assert labels_2025["conf-championship"] == "Week 15 (CCG)", labels_2025["conf-championship"]


def test_identified_championship_does_not_mint_a_week_column():
    champ = set(identify_conference_championship_games(ACC_2025, 2025).values())
    columns = build_canonical_columns(ACC_2025, 2025, champ, None)
    slot_ids = [slot_id for slot_id, _ in columns]
    assert "week-15" not in slot_ids, "the championship game was diverted; its bucket must not persist"
    assert "week-14" in slot_ids, "the Nov 29-30 regular slate keeps its own column"


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


def test_no_army_navy_column_when_the_season_has_no_army_navy_game():
    """A season whose data carries no Army-Navy row at all must not publish the column.

    Nothing but that one game is ever diverted into the army-navy slot, so with no game to
    divert the column is one no team can ever fill. 2025 is the live case and the reason this
    test exists: its regular season ends 2025-12-07 and its postseason opens 2025-12-14, so the
    real 2025-12-13 Army-Navy game is absent from the data entirely. Before the guard in
    build_canonical_columns, 2025 published a completely empty trailing column -- reported by
    the user as "a completely empty week 17 column", and still empty (relabelled "Week 16")
    once championship identification was fixed.

    ACC_2025 is used precisely because it contains no Army-Navy game, the same shape as the
    real season.
    """
    champ = set(identify_conference_championship_games(ACC_2025, 2025).values())
    assert identify_army_navy_game(ACC_2025, 2025) is None, "fixture precondition"

    slot_ids = [slot_id for slot_id, _ in build_canonical_columns(ACC_2025, 2025, champ, None)]

    assert "army-navy" not in slot_ids, (
        "no Army-Navy game exists this season, so the column can never be filled"
    )
    # The rest of the layout is untouched -- this guard removes one column, nothing else.
    assert "conf-championship" in slot_ids
    assert slot_ids[-4:] == [
        "cfp-r1-bowls", "cfp-quarterfinals", "cfp-semifinals", "cfp-national-championship",
    ]


def test_army_navy_column_is_kept_when_the_game_exists():
    """The counterpart guard: the omission is keyed on the game's absence, never on a season.

    Pins that 2026 -- which does carry an Army-Navy row -- is completely unaffected, so the
    change above cannot silently drop the column from a season that needs it.
    """
    rows = ACC_2026 + ARMY_NAVY_2026
    army_navy_id = identify_army_navy_game(rows, 2026)
    assert army_navy_id is not None, "fixture precondition"

    slot_ids = [slot_id for slot_id, _ in build_canonical_columns(rows, 2026, set(), army_navy_id)]
    assert "army-navy" in slot_ids
