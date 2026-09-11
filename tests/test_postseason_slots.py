"""CHARACTERIZATION tests for artifacts/schedule.py's postseason primitives, pinning their
CURRENT behavior against 2025's real postseason shapes.

These are characterization tests, not specifications. Before this file, `_bucket_cfp_slot` and
`_game_name_for_row` had ZERO test coverage, and `compute_team_records` had no test asserting
postseason games count toward a team's record (also uncovered). Phase 2 of the Season Grid
work is a PLANNED rewrite of postseason bowl-name formatting (sponsor-stripping -- see
the season-grid-2025-postseason non_goals and plan risk R3), so these
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

import artifacts.bowl_names as bowl_names  # noqa: E402
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
    assert bowl_cell["game_name_short"] == "Gasparilla"


def test_bye_seed_present_is_carried_as_its_own_field_not_baked_into_game_name():
    """Fix-cycle-1 design correction: the seed is its own nullable cell field (cfp_seed), not
    pre-composed prose in game_name -- baking an integer into a sentence two functions before
    the frontend reads it conflicted with _game_name_for_row's own contract (non-null only for a
    real, determined game) and inverted K5's whole argument against deriving structured meaning
    from a label. game_name/game_name_short stay null here exactly like an ordinary bye cell;
    the frontend composes its own label from status == CFP_BYE_STATUS plus cfp_seed."""
    team_slot_rows = _team_slot_rows_with_quarterfinal(home_away="home", team_seed=1)
    weeks = _build_team_weeks(CFP_SLOTS, team_slot_rows, None, None, "eligible", {})
    bye_cell = weeks[0]
    assert bye_cell["status"] == CFP_BYE_STATUS
    assert bye_cell["cfp_seed"] == 1
    assert bye_cell["game_name"] is None
    assert bye_cell["game_name_short"] is None


def test_bye_seed_absent_degrades_to_a_null_cfp_seed():
    """K8: an unpopulated seed field must cost a nicety, not break the feature -- cfp_seed stays
    null (rather than guessing) so the frontend's plain 'CFP Bye' STATUS_TEXT_BY_SLOT fallback
    renders instead of a possibly-wrong seed number."""
    team_slot_rows = _team_slot_rows_with_quarterfinal(home_away="home")
    weeks = _build_team_weeks(CFP_SLOTS, team_slot_rows, None, None, "eligible", {})
    bye_cell = weeks[0]
    assert bye_cell["status"] == CFP_BYE_STATUS
    assert bye_cell["cfp_seed"] is None
    assert bye_cell["game_name"] is None
    assert bye_cell["game_name_short"] is None


def test_cfp_seed_is_null_on_every_non_bye_cell():
    """cfp_seed exists only to carry a CFP bye team's seed -- null everywhere else, including a
    real game's own cell (K5's playoff_round precedent: a nullable field added to every cell,
    meaningfully populated in exactly one place)."""
    real_row = _row(52, team="Georgia Tech", opponent="Vanderbilt", status="loss",
                     notes="Union Home Mortgage Gasparilla Bowl")
    real_row["home_away"] = "home"
    team_slot_rows = {CFP_R1_BOWLS: real_row}
    weeks = _build_team_weeks(CFP_SLOTS, team_slot_rows, None, None, "eligible", {})
    for week in weeks:
        if week["status"] == CFP_BYE_STATUS:
            continue
        assert week["cfp_seed"] is None, week


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
    # K9: short names now strip the trailing "Bowl" too (see docs/season-grid-standings-fixes),
    # not just the sponsor -- "Gasparilla", not "Gasparilla Bowl".
    ("Union Home Mortgage Gasparilla Bowl", "Gasparilla"),
    ("Bucked Up LA Bowl", "LA"),
    ("Scooter's Coffee Frisco Bowl", "Frisco"),
    ("Radiance Technologies Independence Bowl", "Independence"),
    ("Pop-Tarts Bowl", "Pop-Tarts"),
    ("Xbox Bowl", "Xbox"),
    ("Rate Bowl", "Rate"),
], ids=["gasparilla", "la-bowl", "frisco", "independence", "pop-tarts", "xbox", "rate"])
def test_short_bowl_name_for_2025s_real_ordinary_bowls(full, expected_short):
    assert short_bowl_name(full) == expected_short


@pytest.mark.parametrize("full,expected_short", [
    # Fix-cycle-1: real 2025 bowls whose ROOT is more than one token -- the exact cases a
    # single-trailing-token rule got wrong (see artifacts/bowl_names.py's module docstring for
    # the wrong output each of these used to produce). K9: now also stripped of the trailing
    # "Bowl" itself.
    ("Liberty Mutual Music City Bowl", "Music City"),
    ("Isleta New Mexico Bowl", "New Mexico"),
    ("Lockheed Martin Armed Forces Bowl", "Armed Forces"),
    ("SERVPRO First Responder Bowl", "First Responder"),
    ("SRS Distribution Las Vegas Bowl", "Las Vegas"),
    ("New Orleans Bowl", "New Orleans"),
    ("Bush's Boca Raton Bowl", "Boca Raton"),
    ("Duke's Mayo Bowl", "Duke's Mayo"),  # sponsor-eponymous rebrand, no separate root
    ("Myrtle Beach Bowl", "Myrtle Beach"),  # no title sponsor at all
], ids=["music-city", "new-mexico", "armed-forces", "first-responder", "las-vegas",
        "new-orleans", "boca-raton", "dukes-mayo", "myrtle-beach"])
def test_short_bowl_name_for_2025s_real_multi_token_root_bowls(full, expected_short):
    assert short_bowl_name(full) == expected_short


@pytest.mark.parametrize("full,expected_short", [
    ("College Football Playoff First Round Game", "First Round"),
    ("Rose Bowl", "Rose"),
    ("Sugar Bowl", "Sugar"),
    ("Cotton Bowl", "Cotton"),
    ("Orange Bowl", "Orange"),
    ("Fiesta Bowl", "Fiesta"),
    ("Peach Bowl", "Peach"),
    ("College Football Playoff National Championship Presented by AT&T", "National Championship"),
], ids=["first-round", "rose", "sugar", "cotton", "orange", "fiesta", "peach", "championship"])
def test_short_bowl_name_for_2025s_real_cfp_rounds(full, expected_short):
    assert short_bowl_name(full) == expected_short


# The complete 2025 postseason name universe (R3/K9): all 35 ordinary bowls curated in
# _ROOT_NAMES plus all 8 CFP-related names, input -> expected output, as a single pinned table.
# This is the deliverable that makes the K9 change reviewable -- see
# docs/season-grid-standings-fixes/plan.yaml risk R3. Every entry here also appears, split across
# concerns, in the parametrized tests above/below; this table is what pins the WHOLE 2025 season
# in one place so a future change can't fix one bowl's test while silently breaking another's.
_COMPLETE_2025_TABLE = [
    ("Union Home Mortgage Gasparilla Bowl", "Gasparilla"),
    ("Bucked Up LA Bowl", "LA"),
    ("Scooter's Coffee Frisco Bowl", "Frisco"),
    ("Radiance Technologies Independence Bowl", "Independence"),
    ("Pop-Tarts Bowl", "Pop-Tarts"),
    ("Xbox Bowl", "Xbox"),
    ("Rate Bowl", "Rate"),
    ("Duke's Mayo Bowl", "Duke's Mayo"),
    ("Myrtle Beach Bowl", "Myrtle Beach"),
    ("Liberty Mutual Music City Bowl", "Music City"),
    ("Isleta New Mexico Bowl", "New Mexico"),
    ("Lockheed Martin Armed Forces Bowl", "Armed Forces"),
    ("SERVPRO First Responder Bowl", "First Responder"),
    ("SRS Distribution Las Vegas Bowl", "Las Vegas"),
    ("New Orleans Bowl", "New Orleans"),
    ("Bush's Boca Raton Bowl", "Boca Raton"),
    ("68 Ventures Bowl", "68 Ventures"),
    ("AutoZone Liberty Bowl", "Liberty"),
    ("Bad Boy Mowers Pinstripe Bowl", "Pinstripe"),
    ("Cheez-It Citrus Bowl", "Citrus"),
    ("Famous Idaho Potato Bowl", "Idaho Potato"),
    ("GameAbove Sports Bowl", "GameAbove Sports"),
    ("Go Bowling Military Bowl", "Military"),
    ("IS4S Salute to Veterans Bowl", "Salute to Veterans"),
    ("JLab Birmingham Bowl", "Birmingham"),
    ("Kinder's Texas Bowl", "Texas"),
    ("ReliaQuest Bowl", "ReliaQuest"),
    ("Sheraton Hawaiʻi Bowl", "Hawaiʻi"),
    ("Snoop Dogg Arizona Bowl", "Arizona"),
    ("StaffDNA Cure Bowl", "Cure"),
    ("TaxSlayer Gator Bowl", "Gator"),
    ("Tony the Tiger Sun Bowl", "Sun"),
    ("Trust & Will Holiday Bowl", "Holiday"),
    ("Valero Alamo Bowl", "Alamo"),
    ("Wasabi Fenway Bowl", "Fenway"),
    ("Sugar Bowl", "Sugar"),
    ("Orange Bowl", "Orange"),
    ("Cotton Bowl", "Cotton"),
    ("Rose Bowl", "Rose"),
    ("Peach Bowl", "Peach"),
    ("Fiesta Bowl", "Fiesta"),
    ("College Football Playoff First Round Game", "First Round"),
    ("College Football Playoff National Championship Presented by AT&T", "National Championship"),
]


def test_complete_2025_bowl_and_cfp_name_table_is_exactly_43_entries():
    """Guards the table's own completeness (35 ordinary bowls + 8 CFP names) so a future edit
    that silently drops or duplicates an entry is caught here, not just in the loop below."""
    assert len(_COMPLETE_2025_TABLE) == 43
    assert len({full for full, _ in _COMPLETE_2025_TABLE}) == 43


@pytest.mark.parametrize("full,expected_short", _COMPLETE_2025_TABLE,
                          ids=[full for full, _ in _COMPLETE_2025_TABLE])
def test_short_bowl_name_pins_the_complete_2025_name_table(full, expected_short):
    assert short_bowl_name(full) == expected_short


def test_short_bowl_name_never_inspects_a_sponsor_that_itself_contains_the_word_bowl():
    """Trap (K9): 'Go Bowling Military Bowl' has a SPONSOR containing 'Bowl' ('Go Bowling').
    Matching and stripping operate only on the curated root ('Military Bowl'), never on
    substrings of the raw input, so 'Go Bowling' must never be inspected or altered."""
    assert short_bowl_name("Go Bowling Military Bowl") == "Military"


def test_short_bowl_name_matches_the_okina_in_hawaii_bowl():
    """Trap (K9): 'Hawaiʻi Bowl' contains U+02BB ʻOKINA, not an ASCII apostrophe. This
    root is newly curated, so this is the first time this character reaches the matching logic
    at all -- pin it explicitly rather than relying on it passing incidentally inside the
    complete-table test above."""
    assert short_bowl_name("Sheraton Hawaiʻi Bowl") == "Hawaiʻi"
    assert "ʻ" in short_bowl_name("Sheraton Hawaiʻi Bowl")


def test_short_bowl_name_refuses_to_strip_to_an_empty_or_blank_result(monkeypatch):
    """K9 guard: _strip_trailing_bowl must never turn a resolved root into an empty or
    whitespace-only string. No real curated root is this degenerate (every real root has real
    content before "Bowl"), so this pins the guard directly against a deliberately-constructed
    pathological root via monkeypatch rather than resting on today's data never triggering it."""
    assert bowl_names._strip_trailing_bowl("  Bowl") == "  Bowl"
    assert bowl_names._strip_trailing_bowl(" Bowl Classic") == " Bowl Classic"
    monkeypatch.setattr(bowl_names, "_ROOT_NAMES_BY_LENGTH_DESC", ["  Bowl"])
    assert bowl_names.short_bowl_name("Sponsor   Bowl") == "  Bowl"


def test_short_bowl_name_strips_bowl_classic_before_bowl(monkeypatch):
    """K9: '_strip_trailing_bowl' checks the trailing ' Bowl Classic' suffix before ' Bowl'. No
    real 2025 root has this shape, so this pins it via a monkeypatched curated root -- if the
    ' Bowl Classic' case were dropped (leaving only the plain ' Bowl' strip), a root ending in
    ' Bowl Classic' does not end in ' Bowl' (it ends in 'Classic'), so no strip would fire at
    all and the orphaned 'Classic' qualifier would leak into the displayed name."""
    monkeypatch.setattr(bowl_names, "_ROOT_NAMES_BY_LENGTH_DESC", ["Frisco Bowl Classic"])
    assert bowl_names.short_bowl_name("Sponsor Frisco Bowl Classic") == "Frisco"


def test_short_bowl_name_passes_through_an_unrecognized_bowl_unchanged():
    """Fix-cycle-1: replaces a prior test that asserted a structural rule would shorten this to
    'City Bowl' -- that rule was the bug (it guessed a root from position, not identity, and was
    wrong for every multi-token root; see the module docstring). The curated-list approach must
    NEVER emit a name it hasn't actually recognised, so a bowl that ends in 'Bowl' but matches no
    curated root -- including one that could be mistaken for a substring of a real root like
    'Music City Bowl' -- passes through the FULL name unchanged rather than a truncated guess."""
    assert short_bowl_name("Some Brand New 2027 Sponsor City Bowl") == "Some Brand New 2027 Sponsor City Bowl"


def test_short_bowl_name_prefers_the_longest_matching_curated_root(monkeypatch):
    """Direct test of the longest-suffix-wins tie-break, via a deliberately constructed
    collision -- the REAL curated list has no pair where one root is a suffix of another today,
    so this patches in an artificial one ('Beach Bowl' / 'Myrtle Beach Bowl') rather than
    resting on a coincidence of the current data. The longer, more specific root must win."""
    monkeypatch.setattr(bowl_names, "_ROOT_NAMES_BY_LENGTH_DESC", ["Myrtle Beach Bowl", "Beach Bowl"])
    assert bowl_names.short_bowl_name("Some Sponsor Myrtle Beach Bowl") == "Myrtle Beach"


def test_short_bowl_name_degrades_to_the_full_name_for_an_unmapped_non_bowl_shape():
    """K6: a name that doesn't end in 'Bowl' and isn't in the small override map (e.g. a future
    presenting-sponsor change to a CFP round name) degrades to the FULL name unchanged, never to
    a guess -- the personal-site CSS clamp, not this mapping, is what keeps AC6 true here."""
    unmapped = "College Football Playoff Quarterfinal Presented by SomeFutureSponsor"
    assert short_bowl_name(unmapped) == unmapped


def test_short_bowl_name_requires_a_whitespace_boundary_before_the_root():
    """A name that merely ends with a curated root's characters, with no space actually
    separating it (e.g. a word that happens to end in "la"), must NOT match -- only a genuine,
    word-boundary-separated root should shorten. Without this guard, "Gala Bowl" would
    incorrectly shorten to "LA Bowl" (its raw last 7 characters happen to match)."""
    assert short_bowl_name("Gala Bowl") == "Gala Bowl"


def test_short_bowl_name_passes_through_none():
    assert short_bowl_name(None) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
