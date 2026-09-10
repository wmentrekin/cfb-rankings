"""CHARACTERIZATION tests for artifacts/schedule.py's postseason primitives, pinning their
CURRENT behavior against 2025's real postseason shapes.

These are characterization tests, not specifications. Before this file, `_bucket_cfp_slot` and
`_game_name_for_row` had ZERO test coverage, and `compute_team_records` had no test asserting
postseason games count toward a team's record (also uncovered). Phase 2 of the Season Grid
work is a PLANNED rewrite of postseason bowl-name formatting (sponsor-stripping -- see
docs/season-grid-2025-postseason/requirements.yaml non_goals and plan.yaml risk R3), so these
tests exist to make that future diff legible, not to bless today's output as correct or final.

The `_game_name_for_row` coverage below is deliberately split in two:
  - STABLE CONTRACT tests: assert the shape that must survive phase 2 unchanged (non-null only
    for postseason rows; playoff_bowl_name wins over notes).
  - EXACT STRING tests: pin today's literal output. Phase 2's sponsor-stripping WILL break
    these on purpose -- a failure there is an expected, intentional update, not a regression,
    which is exactly why they are not mixed into the stable-contract section.

No formatting/behavior change is made here. This file only characterizes.

Run: python -m pytest tests/ -q   (or: python tests/test_postseason_slots.py)
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifacts.schedule import CFP_SLOTS, _bucket_cfp_slot, _game_name_for_row  # noqa: E402
from artifacts.schedule_standings import compute_team_records  # noqa: E402

SEASON = 2025

CFP_R1_BOWLS, CFP_QUARTERFINALS, CFP_SEMIFINALS, CFP_NATIONAL_CHAMPIONSHIP = (
    slot_id for slot_id, _label in CFP_SLOTS
)


def _row(
    game_id, team="Team A", opponent="Team B", status="win",
    playoff_round_name=None, playoff_bowl_name=None, playoff_bracket_slot=None,
    playoff_round_order=None, notes=None, season_type="postseason",
    conference="ACC", conference_game=False,
):
    """A schedule_grid-shaped row carrying the postseason columns _bucket_cfp_slot and
    _game_name_for_row read. Defaults to a postseason row since that is what both functions
    are about; season_type is overridden by the regular-season contract tests below."""
    return dict(
        game_id=game_id, season=SEASON, season_type=season_type, team=team, opponent=opponent,
        status=status, conference=conference, conference_game=conference_game,
        playoff_round_name=playoff_round_name, playoff_bowl_name=playoff_bowl_name,
        playoff_bracket_slot=playoff_bracket_slot, playoff_round_order=playoff_round_order,
        notes=notes,
    )


# --- Real 2025 postseason fixtures -------------------------------------------------------
# Round names, bowl names, and bracket-slot ids verbatim from 2025's real postseason
# (per the task handoff): 4 First Round, 4 Quarterfinal, 2 Semifinal, 1 National
# Championship, plus ordinary (non-CFP) bowls sourced from `notes`. playoff_round_order is
# NULL on every row, matching 2025's real data (see test_playoff_round_order_is_never_read).
FIRST_ROUND_ROWS = [
    _row(1, playoff_round_name="First Round",
         playoff_bowl_name="College Football Playoff First Round Game", playoff_bracket_slot="FR1"),
    _row(2, playoff_round_name="First Round",
         playoff_bowl_name="College Football Playoff First Round Game", playoff_bracket_slot="FR2"),
    _row(3, playoff_round_name="First Round",
         playoff_bowl_name="College Football Playoff First Round Game", playoff_bracket_slot="FR3"),
    _row(4, playoff_round_name="First Round",
         playoff_bowl_name="College Football Playoff First Round Game", playoff_bracket_slot="FR4"),
]

QUARTERFINAL_ROWS = [
    _row(5, playoff_round_name="Quarterfinal", playoff_bowl_name="Rose Bowl", playoff_bracket_slot="QF1"),
    _row(6, playoff_round_name="Quarterfinal", playoff_bowl_name="Sugar Bowl", playoff_bracket_slot="QF2"),
    _row(7, playoff_round_name="Quarterfinal", playoff_bowl_name="Cotton Bowl", playoff_bracket_slot="QF3"),
    _row(8, playoff_round_name="Quarterfinal", playoff_bowl_name="Orange Bowl", playoff_bracket_slot="QF4"),
]

SEMIFINAL_ROWS = [
    _row(9, playoff_round_name="Semifinal", playoff_bowl_name="Fiesta Bowl", playoff_bracket_slot="SF1"),
    _row(10, playoff_round_name="Semifinal", playoff_bowl_name="Peach Bowl", playoff_bracket_slot="SF2"),
]

CHAMPIONSHIP_ROW = _row(
    11, playoff_round_name="National Championship",
    playoff_bowl_name="College Football Playoff National Championship Presented by AT&T",
    playoff_bracket_slot="CH",
)

# Ordinary (non-CFP) bowls: no playoff_round_name/playoff_bowl_name at all -- the sponsor
# name lives only in `notes`.
ORDINARY_BOWL_ROWS = [
    _row(12, playoff_round_name=None, playoff_bowl_name=None, notes="Union Home Mortgage Gasparilla Bowl"),
    _row(13, playoff_round_name=None, playoff_bowl_name=None, notes="Bucked Up LA Bowl"),
]


# ===========================================================================
# _bucket_cfp_slot
# ===========================================================================
@pytest.mark.parametrize("row", FIRST_ROUND_ROWS, ids=lambda r: r["playoff_bracket_slot"])
def test_first_round_games_bucket_into_cfp_r1_bowls_slot(row):
    """PINS a deliberate phase-1 shape (requirements.yaml non_goals): the CFP first round is
    NOT split into its own column in this pass -- all four first-round games land in
    cfp-r1-bowls, the same slot as ordinary bowls. Splitting it out, or visually
    distinguishing it within the shared column, is phase-2 scope (open question Q1)."""
    slot_id, _label = _bucket_cfp_slot(row)
    assert slot_id == CFP_R1_BOWLS


@pytest.mark.parametrize("row", ORDINARY_BOWL_ROWS, ids=lambda r: r["notes"])
def test_ordinary_bowls_share_the_same_slot_as_cfp_first_round(row):
    slot_id, _label = _bucket_cfp_slot(row)
    assert slot_id == CFP_R1_BOWLS


@pytest.mark.parametrize("row", QUARTERFINAL_ROWS, ids=lambda r: r["playoff_bracket_slot"])
def test_quarterfinal_games_bucket_into_cfp_quarterfinals_slot(row):
    slot_id, _label = _bucket_cfp_slot(row)
    assert slot_id == CFP_QUARTERFINALS


@pytest.mark.parametrize("row", SEMIFINAL_ROWS, ids=lambda r: r["playoff_bracket_slot"])
def test_semifinal_games_bucket_into_cfp_semifinals_slot(row):
    slot_id, _label = _bucket_cfp_slot(row)
    assert slot_id == CFP_SEMIFINALS


def test_national_championship_game_buckets_into_cfp_national_championship_slot():
    slot_id, _label = _bucket_cfp_slot(CHAMPIONSHIP_ROW)
    assert slot_id == CFP_NATIONAL_CHAMPIONSHIP


def test_playoff_round_order_is_never_read():
    """PINS a deliberate scope decision (plan.yaml non_goals: 'Backfilling playoff_round_order.
    Nothing reads it; _bucket_cfp_slot matches playoff_round_name substrings.'). Two rows
    identical except playoff_round_order (2025's real NULL vs. an arbitrary non-null value)
    must bucket identically."""
    null_order = _row(20, playoff_round_name="Quarterfinal", playoff_bowl_name="Rose Bowl",
                       playoff_bracket_slot="QF1", playoff_round_order=None)
    non_null_order = _row(21, playoff_round_name="Quarterfinal", playoff_bowl_name="Rose Bowl",
                           playoff_bracket_slot="QF1", playoff_round_order=99)
    assert _bucket_cfp_slot(null_order) == _bucket_cfp_slot(non_null_order)


# ===========================================================================
# _game_name_for_row -- STABLE CONTRACT (durable across phase 2)
# ===========================================================================
def test_game_name_is_null_for_a_non_postseason_row_even_with_a_bowl_name_present():
    row = _row(30, season_type="regular", playoff_bowl_name="Rose Bowl", notes="Regular season note")
    assert _game_name_for_row(row) is None


def test_game_name_is_non_null_for_a_postseason_row_with_a_bowl_name():
    row = _row(31, season_type="postseason", playoff_bowl_name="Rose Bowl")
    assert _game_name_for_row(row) is not None


def test_playoff_bowl_name_is_preferred_over_notes():
    row = _row(32, season_type="postseason", playoff_bowl_name="Rose Bowl", notes="Should be ignored")
    assert _game_name_for_row(row) == "Rose Bowl"


def test_notes_is_the_fallback_when_playoff_bowl_name_is_absent():
    row = _row(33, season_type="postseason", playoff_bowl_name=None,
               notes="Union Home Mortgage Gasparilla Bowl")
    assert _game_name_for_row(row) == "Union Home Mortgage Gasparilla Bowl"


def test_game_name_is_null_when_neither_source_is_present():
    row = _row(34, season_type="postseason", playoff_bowl_name=None, notes=None)
    assert _game_name_for_row(row) is None


# ===========================================================================
# _game_name_for_row -- EXACT STRINGS
#
# Phase 2's sponsor-stripping (requirements.yaml non_goals: "decided (strip), but
# implemented in phase 2") will rewrite the strings asserted here. A failure in THIS
# section after that change lands is an expected, intentional update -- update the
# expected string. A failure in the STABLE CONTRACT section above is a regression.
# ===========================================================================
@pytest.mark.parametrize("row,expected", [
    (FIRST_ROUND_ROWS[0], "College Football Playoff First Round Game"),
    (QUARTERFINAL_ROWS[0], "Rose Bowl"),
    (QUARTERFINAL_ROWS[1], "Sugar Bowl"),
    (QUARTERFINAL_ROWS[2], "Cotton Bowl"),
    (QUARTERFINAL_ROWS[3], "Orange Bowl"),
    (SEMIFINAL_ROWS[0], "Fiesta Bowl"),
    (SEMIFINAL_ROWS[1], "Peach Bowl"),
    (CHAMPIONSHIP_ROW, "College Football Playoff National Championship Presented by AT&T"),
], ids=["first-round", "rose", "sugar", "cotton", "orange", "fiesta", "peach", "championship"])
def test_exact_game_name_for_cfp_rounds(row, expected):
    assert _game_name_for_row(row) == expected


@pytest.mark.parametrize("row,expected", [
    (ORDINARY_BOWL_ROWS[0], "Union Home Mortgage Gasparilla Bowl"),
    (ORDINARY_BOWL_ROWS[1], "Bucked Up LA Bowl"),
], ids=["gasparilla", "la-bowl"])
def test_exact_game_name_for_ordinary_bowls_falls_back_to_notes(row, expected):
    assert _game_name_for_row(row) == expected


# ===========================================================================
# compute_team_records -- postseason games count toward the record
# ===========================================================================
def test_postseason_games_count_toward_the_record():
    """PINS the behavior documented in requirements.yaml constraints: compute_team_records
    applies NO season_type filter, so a bowl loss counts toward the overall record exactly
    like a regular-season loss -- the real 2025 shape (9-3 regular + a bowl loss = 9-4, not
    the 9-3 a season_type-filtered tally would produce)."""
    rows = [
        _row(40, team="Georgia Tech", opponent="Vanderbilt", status="win",
             season_type="regular", conference="ACC", conference_game=True),
        _row(41, team="Georgia Tech", opponent="Duke", status="win",
             season_type="regular", conference="ACC", conference_game=True),
        _row(42, team="Georgia Tech", opponent="Vanderbilt", status="loss",
             season_type="postseason", conference="ACC", conference_game=False,
             notes="Union Home Mortgage Gasparilla Bowl"),
    ]
    records = compute_team_records(rows, SEASON)
    assert records["Georgia Tech"]["wins"] == 2
    assert records["Georgia Tech"]["losses"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
