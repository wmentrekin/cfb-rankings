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

from artifacts.bowl_names import short_bowl_name  # noqa: E402
from artifacts.schedule import (  # noqa: E402
    CFP_BYE_STATUS,
    CFP_SLOTS,
    _build_team_weeks,
    _bucket_cfp_slot,
    _game_name_for_row,
    _playoff_round_for_row,
)
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
    """Deliberately a sponsor-FREE notes value, unlike the EXACT STRINGS section below: this
    pins the fallback MECHANISM (notes is read when playoff_bowl_name is absent), not any
    particular sponsored bowl name -- phase 2's sponsor-stripping must not touch this test,
    since there is no sponsor text here to strip."""
    row = _row(33, season_type="postseason", playoff_bowl_name=None,
               notes="Postseason Exhibition Game")
    assert _game_name_for_row(row) == "Postseason Exhibition Game"


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


# ===========================================================================
# T4/AC5 -- first-round CFP bye state (_build_team_weeks' cfp-r1-bowls branch)
# ===========================================================================
def _team_slot_rows_with_quarterfinal(home_away="home", team_seed=None):
    """A team_slot_rows dict shaped like Indiana's real 2025 case: a real row in
    cfp-quarterfinals, no entry at all for cfp-r1-bowls (no real row there).

    team_seed is already team-relative, because schedule_grid flips the seed per perspective
    (migration 0006), exactly as it flips team_score/opp_score. home_away is still set so these
    fixtures stay row-shaped, but nothing reads it to resolve the seed any more."""
    row = _row(50, team="Indiana", opponent="Notre Dame", status="win",
               playoff_round_name="Quarterfinal", playoff_bowl_name="Sugar Bowl",
               playoff_bracket_slot="QF1")
    row["home_away"] = home_away
    row["team_seed"] = team_seed
    return {CFP_QUARTERFINALS: row}


def test_bye_status_present_for_a_team_with_a_quarterfinal_row_but_no_r1_bowls_row():
    """AC5: Indiana's real 2025 shape -- present."""
    team_slot_rows = _team_slot_rows_with_quarterfinal(home_away="home", team_seed=1)
    weeks = _build_team_weeks(CFP_SLOTS, team_slot_rows, None, None, "eligible", {})
    bye_cell = weeks[0]
    assert bye_cell["slot_id"] == CFP_R1_BOWLS
    assert bye_cell["status"] == CFP_BYE_STATUS


def test_bye_status_absent_for_a_team_that_missed_the_playoff():
    """A team with NEITHER a real cfp-quarterfinals row NOR a real cfp-r1-bowls row must keep
    rendering exactly as it does today: the bowl-eligibility placeholder, never the new bye
    status."""
    weeks = _build_team_weeks(CFP_SLOTS, {}, None, None, "eligible", {})
    bowl_cell = weeks[0]
    assert bowl_cell["slot_id"] == CFP_R1_BOWLS
    assert bowl_cell["status"] == "eligible"
    assert bowl_cell["status"] != CFP_BYE_STATUS


def test_bye_status_absent_for_an_ordinary_bowl_team():
    """A team with a REAL cfp-r1-bowls row (an ordinary bowl, or a CFP first-round game) takes
    the real-row path and must never be reclassified as a bye, whatever else that team's
    team_slot_rows dict happens to contain."""
    real_row = _row(51, team="Georgia Tech", opponent="Vanderbilt", status="loss",
                     notes="Union Home Mortgage Gasparilla Bowl")
    real_row["home_away"] = "home"
    team_slot_rows = {CFP_R1_BOWLS: real_row}
    weeks = _build_team_weeks(CFP_SLOTS, team_slot_rows, None, None, "eligible", {})
    bowl_cell = weeks[0]
    assert bowl_cell["slot_id"] == CFP_R1_BOWLS
    assert bowl_cell["status"] != CFP_BYE_STATUS
    assert bowl_cell["game_name"] == "Union Home Mortgage Gasparilla Bowl"
    assert bowl_cell["game_name_short"] == "Gasparilla Bowl"


def test_bye_seed_present_renders_a_seeded_label():
    team_slot_rows = _team_slot_rows_with_quarterfinal(home_away="home", team_seed=1)
    weeks = _build_team_weeks(CFP_SLOTS, team_slot_rows, None, None, "eligible", {})
    bye_cell = weeks[0]
    assert bye_cell["status"] == CFP_BYE_STATUS
    assert bye_cell["game_name"] == "Bye (No. 1)"
    assert bye_cell["game_name_short"] == "Bye (No. 1)"


def test_bye_seed_derived_from_the_away_side_when_the_team_was_the_away_team():
    team_slot_rows = _team_slot_rows_with_quarterfinal(home_away="away", team_seed=4)
    weeks = _build_team_weeks(CFP_SLOTS, team_slot_rows, None, None, "eligible", {})
    assert weeks[0]["game_name"] == "Bye (No. 4)"


def test_bye_seed_absent_degrades_to_a_plain_bye_with_no_game_name():
    """K8: an unpopulated seed field must cost a nicety, not break the feature -- game_name
    stays null (rather than guessing) so the frontend's plain 'CFP Bye' STATUS_TEXT_BY_SLOT
    fallback renders instead of a possibly-wrong seed number."""
    team_slot_rows = _team_slot_rows_with_quarterfinal(home_away="home")
    weeks = _build_team_weeks(CFP_SLOTS, team_slot_rows, None, None, "eligible", {})
    bye_cell = weeks[0]
    assert bye_cell["status"] == CFP_BYE_STATUS
    assert bye_cell["game_name"] is None
    assert bye_cell["game_name_short"] is None


# ===========================================================================
# K5 -- playoff_round on the cell
# ===========================================================================
def test_playoff_round_populated_for_a_cfp_round_game():
    row = _row(60, season_type="postseason", playoff_round_name="Quarterfinal", playoff_bowl_name="Rose Bowl")
    assert _playoff_round_for_row(row) == "Quarterfinal"


def test_playoff_round_null_for_an_ordinary_bowl():
    row = _row(61, season_type="postseason", playoff_round_name=None, notes="Bucked Up LA Bowl")
    assert _playoff_round_for_row(row) is None


def test_playoff_round_null_for_a_non_postseason_row_even_with_a_round_name_present():
    """Mirrors test_game_name_is_null_for_a_non_postseason_row_even_with_a_bowl_name_present --
    same season_type gate, same reason: a stray value on a regular-season row must not leak."""
    row = _row(62, season_type="regular", playoff_round_name="Quarterfinal")
    assert _playoff_round_for_row(row) is None


def test_playoff_round_is_null_on_every_placeholder_and_bye_cell():
    weeks = _build_team_weeks(CFP_SLOTS, {}, None, None, "eligible", {})
    for week in weeks:
        assert week["playoff_round"] is None


# ===========================================================================
# K6 -- short bowl/CFP-round display names (artifacts/bowl_names.py)
# ===========================================================================
@pytest.mark.parametrize("full,expected_short", [
    ("Union Home Mortgage Gasparilla Bowl", "Gasparilla Bowl"),
    ("Bucked Up LA Bowl", "LA Bowl"),
    ("Scooter's Coffee Frisco Bowl", "Frisco Bowl"),
    ("Radiance Technologies Independence Bowl", "Independence Bowl"),
    ("Pop-Tarts Bowl", "Pop-Tarts Bowl"),
    ("Xbox Bowl", "Xbox Bowl"),
    ("Rate Bowl", "Rate Bowl"),
], ids=["gasparilla", "la-bowl", "frisco", "independence", "pop-tarts", "xbox", "rate"])
def test_short_bowl_name_for_2025s_real_ordinary_bowls(full, expected_short):
    assert short_bowl_name(full) == expected_short


@pytest.mark.parametrize("full,expected_short", [
    ("College Football Playoff First Round Game", "CFP First Round"),
    ("Rose Bowl", "Rose Bowl"),
    ("Sugar Bowl", "Sugar Bowl"),
    ("Cotton Bowl", "Cotton Bowl"),
    ("Orange Bowl", "Orange Bowl"),
    ("Fiesta Bowl", "Fiesta Bowl"),
    ("Peach Bowl", "Peach Bowl"),
    ("College Football Playoff National Championship Presented by AT&T", "CFP National Championship"),
], ids=["first-round", "rose", "sugar", "cotton", "orange", "fiesta", "peach", "championship"])
def test_short_bowl_name_for_2025s_real_cfp_rounds(full, expected_short):
    assert short_bowl_name(full) == expected_short


def test_short_bowl_name_shortens_a_future_unseen_sponsor_via_the_structural_rule():
    """K6 durability check: a sponsor never enumerated anywhere in this module still shortens
    correctly, because the rule strips by STRUCTURE (the word immediately before 'Bowl'), not by
    a hardcoded list of known sponsor names -- the whole reason a pure map was rejected."""
    assert short_bowl_name("Some Brand New 2027 Sponsor City Bowl") == "City Bowl"


def test_short_bowl_name_degrades_to_the_full_name_for_an_unmapped_non_bowl_shape():
    """K6: a name that doesn't end in 'Bowl' and isn't in the small override map (e.g. a future
    presenting-sponsor change to a CFP round name) degrades to the FULL name unchanged, never to
    a guess -- the personal-site CSS clamp, not this mapping, is what keeps AC6 true here."""
    unmapped = "College Football Playoff Quarterfinal Presented by SomeFutureSponsor"
    assert short_bowl_name(unmapped) == unmapped


def test_short_bowl_name_passes_through_none():
    assert short_bowl_name(None) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
