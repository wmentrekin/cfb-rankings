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

CFP COLUMN COLLAPSE (Season Grid payload change, playoffs-contract.md): the grid carries ONE
postseason column ("cfp-r1-bowls", labelled "Bowls"). Cases pinned below:
  - _is_cfp_row: a postseason row with a non-empty playoff_round_name is a CFP row; ordinary
    bowls (null/blank round name) and non-postseason rows are not
  - _classify_row_slot: ordinary bowl -> the Bowls slot; every CFP round -> None (never indexed
    into a slot, never mistaken for a regular bowl)
  - _build_team_cfp_seeds: first-round loser, first-round winner who went on, bye team that lost
    its quarterfinal, champion -> one entry each with the right seed; lowest non-null seed wins;
    all-null seeds -> None; non-FBS / non-CFP teams absent
  - _build_team_weeks: playoff team -> exactly one `cfp` cell (no opponent/score/result);
    non-playoff bowl team and ineligible team unchanged
  - build_schedule_payload end to end with a small mixed fixture; no removed slot id anywhere

MUTATION-CHECK LOG (each mutation applied to artifacts/schedule.py, the named test(s) failed,
then reverted):
  (a) _classify_row_slot lets a Quarterfinal row fall through to CFP_SLOTS[0] (bucketed into a
      slot) -> test_every_cfp_round_is_never_indexed_into_a_slot (all 11 rows) failed
  (b) _build_team_cfp_seeds skips rows whose round name contains "quarterfinal" (drops the bye
      team) -> test_cfp_seeds_cover_first_round_loser_winner_bye_team_and_champion,
      test_cfp_seed_is_lowest_non_null_and_none_when_all_null, test_end_to_end_mixed_payload failed
  (c) _cfp_cell emits a non-null opponent -> test_playoff_team_gets_exactly_one_bare_cfp_cell (3 params),
      test_end_to_end_mixed_payload failed

Run: python -m pytest tests/ -q   (or: python tests/test_postseason_slots.py)
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import artifacts.bowl_names as bowl_names  # noqa: E402
from artifacts.bowl_names import short_bowl_name  # noqa: E402
from artifacts.schedule import (  # noqa: E402
    CFP_SLOTS,
    _build_team_cfp_seeds,
    _build_team_weeks,
    _classify_row_slot,
    _game_name_for_row,
    _is_cfp_row,
    _playoff_round_for_row,
    build_canonical_columns,
    build_schedule_payload,
)
from artifacts.schedule_standings import compute_team_records  # noqa: E402

SEASON = 2025

(CFP_R1_BOWLS,) = (slot_id for slot_id, _label in CFP_SLOTS)
REMOVED_SLOT_IDS = {"cfp-quarterfinals", "cfp-semifinals", "cfp-national-championship"}


def _row(
    game_id, team="Team A", opponent="Team B", status="win",
    playoff_round_name=None, playoff_bowl_name=None, playoff_bracket_slot=None,
    playoff_round_order=None, notes=None, season_type="postseason",
    conference="ACC", conference_game=False,
):
    """A schedule_grid-shaped row carrying the postseason columns _classify_row_slot and
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
# _is_cfp_row / _classify_row_slot
# ===========================================================================
ALL_CFP_ROWS = FIRST_ROUND_ROWS + QUARTERFINAL_ROWS + SEMIFINAL_ROWS + [CHAMPIONSHIP_ROW]


def test_columns_end_with_exactly_one_postseason_slot_labelled_bowls():
    assert CFP_SLOTS == [("cfp-r1-bowls", "Bowls")]
    columns = build_canonical_columns([], 2025)
    assert columns[-1] == ("cfp-r1-bowls", "Bowls")
    assert REMOVED_SLOT_IDS.isdisjoint(slot_id for slot_id, _ in columns)


@pytest.mark.parametrize("row", ALL_CFP_ROWS, ids=lambda r: r["playoff_bracket_slot"])
def test_every_cfp_round_is_never_indexed_into_a_slot(row):
    """First Round, Quarterfinal, Semifinal and National Championship rows have no column, so
    they classify to None -- never to the Bowls slot (which would mis-read them as a regular
    bowl) and never to a removed slot id."""
    assert _is_cfp_row(row)
    assert _classify_row_slot(row, set(), None) is None


@pytest.mark.parametrize("row", ORDINARY_BOWL_ROWS, ids=lambda r: r["notes"])
def test_ordinary_bowls_classify_into_the_bowls_slot(row):
    assert not _is_cfp_row(row)
    assert _classify_row_slot(row, set(), None) == ("cfp-r1-bowls", "Bowls")


def test_blank_round_name_is_an_ordinary_bowl_and_regular_rows_are_never_cfp():
    assert not _is_cfp_row(_row(14, playoff_round_name="   ", notes="Pop-Tarts Bowl"))
    assert not _is_cfp_row(_row(15, season_type="regular", playoff_round_name="Quarterfinal"))


def test_playoff_round_order_is_never_read():
    """PINS a deliberate scope decision (plan.yaml non_goals: 'Backfilling playoff_round_order.
    Nothing reads it'). Two rows identical except playoff_round_order must detect identically."""
    null_order = _row(20, playoff_round_name="Quarterfinal", playoff_bowl_name="Rose Bowl",
                       playoff_bracket_slot="QF1", playoff_round_order=None)
    non_null_order = _row(21, playoff_round_name="Quarterfinal", playoff_bowl_name="Rose Bowl",
                           playoff_bracket_slot="QF1", playoff_round_order=99)
    assert _is_cfp_row(null_order) == _is_cfp_row(non_null_order) is True


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
# CFP team cell (_build_team_cfp_seeds + _build_team_weeks' single `cfp` Bowls cell)
# ===========================================================================
FBS = {"Alpha", "Beta", "Gamma", "Delta", "Echo", "Foxtrot"}
BARE_CFP_CELL = {
    "season_type": "postseason", "opponent": None, "opponent_logo_url": None,
    "conditional_opponent": None, "game_name": "College Football Playoff",
    "game_name_short": "CFP", "home_away": None, "neutral_site": False, "status": "cfp",
    "team_score": None, "opp_score": None, "playoff_round": "CFP",
}


def _cfp_row(game_id, team, round_name, seed, status="win", opponent="Zulu"):
    row = _row(game_id, team=team, opponent=opponent, status=status,
               playoff_round_name=round_name, playoff_bracket_slot=f"S{game_id}")
    row["team_seed"] = seed
    return row


CFP_FIXTURE_ROWS = [
    _cfp_row(101, "Alpha", "First Round", 12, status="loss"),           # first-round loser
    _cfp_row(102, "Beta", "First Round", 9, status="win"),              # first-round winner...
    _cfp_row(103, "Beta", "Quarterfinal", 9, status="loss"),            # ...who went on
    _cfp_row(104, "Gamma", "Quarterfinal", 1, status="loss"),           # bye team, lost QF
    _cfp_row(105, "Delta", "Quarterfinal", 2, status="win"),            # champion
    _cfp_row(106, "Delta", "Semifinal", 2, status="win"),
    _cfp_row(107, "Delta", "National Championship", 2, status="win"),
    _row(108, team="Echo", opponent="Zulu", status="win", notes="Pop-Tarts Bowl"),  # plain bowl
]


def test_cfp_seeds_cover_first_round_loser_winner_bye_team_and_champion():
    seeds = _build_team_cfp_seeds(CFP_FIXTURE_ROWS, FBS)
    assert seeds == {"Alpha": 12, "Beta": 9, "Gamma": 1, "Delta": 2}
    assert "Echo" not in seeds and "Foxtrot" not in seeds  # bowl team / no postseason


def test_cfp_seed_is_lowest_non_null_and_none_when_all_null():
    rows = [
        _cfp_row(110, "Alpha", "First Round", None),
        _cfp_row(111, "Alpha", "Quarterfinal", 7),
        _cfp_row(112, "Alpha", "Semifinal", 5),
        _cfp_row(113, "Beta", "First Round", None),
        _cfp_row(114, "Beta", "Quarterfinal", None),
        _cfp_row(115, "Gamma", "Quarterfinal", "n/a"),
    ]
    assert _build_team_cfp_seeds(rows, FBS) == {"Alpha": 5, "Beta": None, "Gamma": None}


def test_cfp_seeds_ignore_non_fbs_teams():
    rows = [_cfp_row(120, "Not FBS", "First Round", 11)]
    assert _build_team_cfp_seeds(rows, FBS) == {}


@pytest.mark.parametrize("seed", [1, 12, None])
def test_playoff_team_gets_exactly_one_bare_cfp_cell(seed):
    weeks = _build_team_weeks(CFP_SLOTS, {}, None, None, "eligible", {},
                              is_cfp_team=True, cfp_seed=seed)
    assert len(weeks) == 1
    assert weeks[0] == {"slot_id": "cfp-r1-bowls", "label": "Bowls", **BARE_CFP_CELL, "cfp_seed": seed}


def test_cfp_cell_key_set_matches_a_placeholder_cell():
    cfp = _build_team_weeks(CFP_SLOTS, {}, None, None, "eligible", {}, is_cfp_team=True)[0]
    placeholder = _build_team_weeks(CFP_SLOTS, {}, None, None, "eligible", {})[0]
    assert set(cfp) == set(placeholder)


def test_non_playoff_bowl_team_is_unchanged():
    real_row = _row(51, team="Georgia Tech", opponent="Vanderbilt", status="loss",
                     notes="Union Home Mortgage Gasparilla Bowl")
    real_row["home_away"] = "home"
    cell = _build_team_weeks(CFP_SLOTS, {CFP_R1_BOWLS: real_row}, None, None, "eligible", {})[0]
    assert cell["status"] == "loss"
    assert cell["opponent"] == "Vanderbilt"
    assert cell["game_name"] == "Union Home Mortgage Gasparilla Bowl"
    assert cell["game_name_short"] == "Gasparilla"
    assert cell["playoff_round"] is None
    assert cell["cfp_seed"] is None


@pytest.mark.parametrize("bowl_status", ["eligible", "ineligible", "possible"])
def test_team_without_a_postseason_game_keeps_its_placeholder(bowl_status):
    cell = _build_team_weeks(CFP_SLOTS, {}, None, None, bowl_status, {})[0]
    assert cell["status"] == bowl_status
    assert cell["game_name"] is None and cell["cfp_seed"] is None and cell["playoff_round"] is None


def _teams_meta():
    return {t: {"conference": "ACC", "division": None, "logos": None} for t in sorted(FBS)}


def _regular_rows():
    rows = []
    for i, team in enumerate(sorted(FBS)):
        rows.append(dict(
            game_id=900 + i, season=SEASON, season_type="regular", team=team, opponent="Zulu",
            conference="ACC", conference_game=False, status="win",
            start_date="2025-09-06T16:00:00Z", home_away="home", neutral_site=False,
            team_score=30, opp_score=10,
        ))
    return rows


def _payload_teams(payload):
    return {t["team"]: t for conf in payload["conferences"] for t in conf["teams"]}


def test_end_to_end_mixed_payload():
    rows = _regular_rows() + [dict(r, season=SEASON, start_date="2025-12-20T16:00:00Z",
                                   home_away="home", neutral_site=False, team_score=20, opp_score=10)
                              for r in CFP_FIXTURE_ROWS]
    payload = build_schedule_payload(rows, _teams_meta(), SEASON)
    teams = _payload_teams(payload)

    expected_seed = {"Alpha": 12, "Beta": 9, "Gamma": 1, "Delta": 2}
    for team, seed in expected_seed.items():
        postseason = [w for w in teams[team]["weeks"] if w["season_type"] == "postseason"]
        assert len(postseason) == 1, team
        assert postseason[0] == {"slot_id": "cfp-r1-bowls", "label": "Bowls",
                                 **BARE_CFP_CELL, "cfp_seed": seed}, team

    echo = teams["Echo"]["weeks"][-1]
    assert echo["slot_id"] == "cfp-r1-bowls"
    assert echo["opponent"] == "Zulu" and echo["game_name"] == "Pop-Tarts Bowl"
    assert echo["playoff_round"] is None and echo["cfp_seed"] is None

    foxtrot = teams["Foxtrot"]["weeks"][-1]
    assert foxtrot["slot_id"] == "cfp-r1-bowls"
    assert foxtrot["status"] != "cfp" and foxtrot["opponent"] is None

    # No cell anywhere carries a removed slot id; every team ends with the Bowls column.
    for team in teams.values():
        slot_ids = [w["slot_id"] for w in team["weeks"]]
        assert REMOVED_SLOT_IDS.isdisjoint(slot_ids)
        assert slot_ids[-1] == "cfp-r1-bowls"
        assert slot_ids.count("cfp-r1-bowls") == 1
    assert REMOVED_SLOT_IDS.isdisjoint(
        w["slot_id"] for t in teams.values() for w in t["weeks"]
    )


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


def test_cfp_cell_playoff_round_is_the_constant_cfp_not_the_row_round_name():
    weeks = _build_team_weeks(CFP_SLOTS, {}, None, None, "eligible", {}, is_cfp_team=True)
    assert weeks[0]["playoff_round"] == "CFP"


# ===========================================================================
# K6 -- short bowl/CFP-round display names (artifacts/bowl_names.py)
# ===========================================================================
@pytest.mark.parametrize("full,expected_short", [
    # Short names strip the trailing "Bowl" too, not just the sponsor -- "Gasparilla", not
    # "Gasparilla Bowl".
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
# This table is what makes the short-name rule reviewable at a glance: the risk it guards is a
# structural rule that reads well on a handful of examples and mangles the rest, which happened
# twice before this existed. Every entry here also appears, split across
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
