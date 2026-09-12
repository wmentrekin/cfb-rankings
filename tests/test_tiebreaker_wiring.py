"""Tests for T4: the seam where artifacts/schedule.py hands a tied group to the tiebreaker engine.

The engine itself is tested in tests/test_tiebreaker_engine.py, the measures in
tests/test_tiebreaker_steps.py, and the config in tests/test_tiebreaker_rules.py. What is tested
HERE is only the wiring:

  - a conference WITH configured rules resolves ties through them, of any group size;
  - a conference WITHOUT configured rules keeps the pre-engine ordering exactly, which is what
    six of the ten FBS conferences still rely on (plan R4/AC7);
  - `tiebreak=None` reproduces the pre-engine behaviour, so every existing hand-built-entry test
    in this suite is still exercising the path it was written for;
  - `resolved_by` reaches the payload and says which step decided;
  - the whole-conference inputs the engine needs are built from the whole conference, not from
    one division or one tied group;
  - a config that cannot be loaded degrades to the pre-engine ordering rather than failing the
    artifact publish.

Run: python -m pytest tests/test_tiebreaker_wiring.py -q
"""
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from artifacts import schedule as schedule_mod  # noqa: E402
from artifacts.schedule import (  # noqa: E402
    _build_tiebreak_inputs,
    _non_fbs_roster,
    _sort_conference_teams,
)

SEASON = 2025

# A conference with primary-source rules, and one without. ALL TEN FBS CONFERENCES ARE NOW
# CONFIGURED, so the unconfigured case can no longer borrow a real conference -- it did twice
# (Mountain West, then Sun Belt) and each transcription broke these tests, which is what the
# assertion below exists to make obvious rather than mysterious.
#
# The pre-engine path is NOT dead code, which is why it is still tested: it is the live path for
# any season outside a conference's configured era -- every season before 2023 for most of them,
# and the whole 2014-2022 divisional era -- and for FBS Independents, which have no conference
# championship and whose conf_record is None.
CONFIGURED = "SEC"
UNCONFIGURED = "Not A Real Conference"


def _entry(team, overall, conf, rank=None):
    wins, losses = overall
    conf_wins, conf_losses = conf
    return {
        "team": team,
        "record": {"wins": wins, "losses": losses},
        "conf_record": {"wins": conf_wins, "losses": conf_losses},
        "rank": rank,
    }


def _row(team, opponent, status, season=SEASON, conference_game=True, season_type="regular",
         game_id=None, team_score=None, opp_score=None):
    return {
        "season": season, "team": team, "opponent": opponent, "status": status,
        "conference_game": conference_game, "season_type": season_type, "game_id": game_id,
        "team_score": team_score, "opp_score": opp_score,
    }


def _game(team_a, score_a, team_b, score_b, **kw):
    a_won = score_a > score_b
    return [
        _row(team_a, team_b, "win" if a_won else "loss", team_score=score_a, opp_score=score_b,
             **kw),
        _row(team_b, team_a, "loss" if a_won else "win", team_score=score_b, opp_score=score_a,
             **kw),
    ]


# ---------------------------------------------------------------------------
# The premise these tests rest on
# ---------------------------------------------------------------------------
def test_the_two_conferences_this_module_assumes_are_as_assumed():
    """CONFIGURED must have rules and UNCONFIGURED must not. If a Group of 5 procedure is
    transcribed later, this fails first and points at the constant to change, instead of some
    downstream test failing for a reason that looks unrelated."""
    configured = _build_tiebreak_inputs(CONFIGURED, [_entry("A", (1, 0), (1, 0))], SEASON, None)
    unconfigured = _build_tiebreak_inputs(
        UNCONFIGURED, [_entry("A", (1, 0), (1, 0))], SEASON, None
    )
    assert configured.rules is not None, f"{CONFIGURED} should have a configured rule set"
    assert unconfigured.rules is None, (
        f"{UNCONFIGURED} now has rules; update this module's UNCONFIGURED constant"
    )


def test_a_configured_conference_still_falls_back_outside_its_era():
    """The realistic unconfigured case now that all ten conferences have rules: a real conference
    asked about a season before its rule set begins. 2014-2022 is the divisional era for most of
    them and no document covers it, so those seasons must still take the pre-engine path."""
    entries = [_entry("A", (1, 0), (1, 0))]
    assert _build_tiebreak_inputs(CONFIGURED, entries, 2015, None).rules is None
    assert _build_tiebreak_inputs(CONFIGURED, entries, SEASON, None).rules is not None


# ---------------------------------------------------------------------------
# A configured conference resolves ties through its own procedure
# ---------------------------------------------------------------------------
def test_configured_conference_resolves_a_two_way_tie_by_head_to_head():
    """Both 3-1 in conference; B beat A. The SEC's step A is head-to-head, so B goes first and
    `resolved_by` names the step that did it."""
    entries = [_entry("A", (8, 1), (3, 1)), _entry("B", (7, 2), (3, 1))]
    rows = _game("B", 27, "A", 20)
    tiebreak = _build_tiebreak_inputs(CONFIGURED, entries, SEASON, None)
    out = _sort_conference_teams(entries, rows, SEASON, None, tiebreak)
    assert [e["team"] for e in out] == ["B", "A"]
    assert out[0]["resolved_by"] == "head_to_head"


def test_configured_conference_resolves_a_three_way_tie_the_pre_engine_path_could_not():
    """THE POINT OF THE WHOLE FEATURE. Three teams tied at 3-1 in a complete round robin among
    themselves: A beat B, B beat C, C beat A is a cycle, so make it decisive instead -- A beat
    both, B beat C. The pre-engine path only ever swapped a group of exactly TWO, so a group of
    three fell through to team name; the SEC's multi-team chain ranks them by intra-group record.
    """
    entries = [
        _entry("Zeta", (8, 1), (3, 1)),        # beats both -> 2-0 in group
        _entry("Yankee", (8, 1), (3, 1)),      # beats Xray   -> 1-1
        _entry("Xray", (8, 1), (3, 1)),        # loses both   -> 0-2
    ]
    rows = (
        _game("Zeta", 30, "Yankee", 10)
        + _game("Zeta", 30, "Xray", 10)
        + _game("Yankee", 30, "Xray", 10)
    )
    tiebreak = _build_tiebreak_inputs(CONFIGURED, entries, SEASON, None)
    out = _sort_conference_teams(entries, rows, SEASON, None, tiebreak)
    assert [e["team"] for e in out] == ["Zeta", "Yankee", "Xray"]
    assert {e["resolved_by"] for e in out} == {"sub_group_record"}
    # Name order would have been Xray, Yankee, Zeta -- the exact inverse of the real result,
    # which is what makes this assertion worth making.


def test_a_team_separated_by_conference_record_alone_has_no_resolved_by():
    """`resolved_by` must say "a tiebreaker step put me here", not be filled in for everyone. A
    team that its conference record separated on its own was never in a tie at all."""
    entries = [_entry("Clear", (9, 0), (5, 0)), _entry("Behind", (5, 4), (2, 3))]
    tiebreak = _build_tiebreak_inputs(CONFIGURED, entries, SEASON, None)
    out = _sort_conference_teams(entries, [], SEASON, None, tiebreak)
    assert [e["team"] for e in out] == ["Clear", "Behind"]
    assert all(e["resolved_by"] is None for e in out)


# ---------------------------------------------------------------------------
# An unconfigured conference is left exactly as it was
# ---------------------------------------------------------------------------
def test_unconfigured_conference_keeps_the_pre_engine_head_to_head_swap():
    """Two teams tied, the second beat the first: the pre-engine swap still fires, so nothing
    regresses for a conference whose rules nobody has supplied."""
    entries = [_entry("A", (8, 1), (3, 1)), _entry("B", (7, 2), (3, 1))]
    rows = _game("B", 27, "A", 20)
    tiebreak = _build_tiebreak_inputs(UNCONFIGURED, entries, SEASON, None)
    out = _sort_conference_teams(entries, rows, SEASON, None, tiebreak)
    assert [e["team"] for e in out] == ["B", "A"]


def test_unconfigured_conference_does_not_guess_at_a_multi_team_procedure():
    """The same three-way tie that the configured conference resolves by intra-group record must
    NOT be resolved that way for a conference with no supplied rules. Guessing at a procedure we
    have not read is the thing this feature exists to stop doing, so the pre-engine ordering
    (placement pct, rank, then name) stands."""
    entries = [
        _entry("Zeta", (8, 1), (3, 1)),
        _entry("Yankee", (8, 1), (3, 1)),
        _entry("Xray", (8, 1), (3, 1)),
    ]
    rows = (
        _game("Zeta", 30, "Yankee", 10)
        + _game("Zeta", 30, "Xray", 10)
        + _game("Yankee", 30, "Xray", 10)
    )
    tiebreak = _build_tiebreak_inputs(UNCONFIGURED, entries, SEASON, None)
    out = _sort_conference_teams(entries, rows, SEASON, None, tiebreak)
    assert [e["team"] for e in out] == ["Xray", "Yankee", "Zeta"], "name order, as before"
    assert all(e["resolved_by"] is None for e in out)


def test_tiebreak_none_reproduces_the_pre_engine_ordering():
    """Every pre-existing test in this suite calls _sort_conference_teams with three positional
    arguments. That path must be byte-identical to what it was, which is what makes those tests
    still meaningful rather than accidentally re-pointed at the engine."""
    def _fixture():
        return [
            _entry("Zeta", (8, 1), (3, 1)),
            _entry("Yankee", (8, 1), (3, 1)),
            _entry("Xray", (8, 1), (3, 1)),
        ]
    rows = (
        _game("Zeta", 30, "Yankee", 10)
        + _game("Zeta", 30, "Xray", 10)
        + _game("Yankee", 30, "Xray", 10)
    )
    bare = [e["team"] for e in _sort_conference_teams(_fixture(), rows, SEASON)]
    unconfigured_entries = _fixture()
    tiebreak = _build_tiebreak_inputs(UNCONFIGURED, unconfigured_entries, SEASON, None)
    via_no_rules = [
        e["team"]
        for e in _sort_conference_teams(unconfigured_entries, rows, SEASON, None, tiebreak)
    ]
    assert bare == via_no_rules == ["Xray", "Yankee", "Zeta"]


# ---------------------------------------------------------------------------
# The whole-conference inputs
# ---------------------------------------------------------------------------
def test_frozen_order_covers_the_whole_conference_and_is_ordered_by_conference_pct():
    """`vs_placed_opponents` walks the conference's order of finish, and
    `opponents_cumulative_conf_pct` looks up OPPONENTS' conference records -- both reach outside
    the tied group, so both inputs must span every member."""
    entries = [
        _entry("Top", (9, 0), (6, 0)),
        _entry("Middle", (6, 3), (3, 3)),
        _entry("Bottom", (2, 7), (0, 6)),
    ]
    tiebreak = _build_tiebreak_inputs(CONFIGURED, entries, SEASON, None)
    assert tiebreak.frozen_order == ["Top", "Middle", "Bottom"]
    assert tiebreak.conf_records == {"Top": (6, 0), "Middle": (3, 3), "Bottom": (0, 6)}


def test_frozen_order_is_deterministic_for_teams_on_equal_percentage():
    """Two teams at the same conference percentage must always land in the same relative place,
    or `vs_placed_opponents` stops being reproducible between runs over identical data."""
    entries = [_entry("Bravo", (5, 4), (2, 2)), _entry("Alpha", (5, 4), (2, 2))]
    first = _build_tiebreak_inputs(CONFIGURED, list(entries), SEASON, None).frozen_order
    second = _build_tiebreak_inputs(CONFIGURED, list(reversed(entries)), SEASON, None).frozen_order
    assert first == second == ["Alpha", "Bravo"]


def test_independents_are_absent_from_conf_records_rather_than_scored_as_zero_zero():
    """An Independent has conf_record None. Recording it as (0, 0) would make it look like a
    conference member who has played nobody, which is a different thing."""
    entries = [_entry("Member", (5, 4), (2, 2))]
    entries.append({
        "team": "Independent", "record": {"wins": 5, "losses": 4}, "conf_record": None,
        "rank": None,
    })
    tiebreak = _build_tiebreak_inputs(CONFIGURED, entries, SEASON, None)
    assert "Independent" not in tiebreak.conf_records
    assert tiebreak.conf_records == {"Member": (2, 2)}


def test_opponent_records_from_outside_the_tied_group_can_decide_a_tie():
    """The discriminating test for whole-conference `conf_records`. Zeta and Alpha are both 1-1
    and never met, and they share no opponents -- so the SEC chain falls through head-to-head and
    the common-opponent steps to step D, the cumulative conference win percentage of each team's
    OWN opponents. Zeta played two 5-0 teams, Alpha two 0-5 teams, so Zeta wins the tie.

    None of those four opponents is in the tied group. If `conf_records` carried only the tied
    teams, step D would find no records for any opponent, fail to separate them, and the chain
    would fall through to the fallback -- which, with identical overall records and no ratings,
    orders by NAME and returns Alpha first. The names are chosen so that the correct answer and
    the broken answer are opposites.
    """
    entries = [
        _entry("Zeta", (5, 5), (1, 1)),
        _entry("Alpha", (5, 5), (1, 1)),
        _entry("Strong1", (9, 0), (5, 0)),
        _entry("Strong2", (9, 0), (5, 0)),
        _entry("Weak1", (0, 9), (0, 5)),
        _entry("Weak2", (0, 9), (0, 5)),
    ]
    rows = (
        _game("Zeta", 28, "Strong1", 21)
        + _game("Strong2", 28, "Zeta", 21)
        + _game("Alpha", 28, "Weak1", 21)
        + _game("Weak2", 28, "Alpha", 21)
    )
    tiebreak = _build_tiebreak_inputs(CONFIGURED, entries, SEASON, None)
    out = _sort_conference_teams(entries, rows, SEASON, None, tiebreak)
    order = [e["team"] for e in out]
    # The four opponents are 5-0 or 0-5 in conference, so they sort above and below the tied
    # pair on conference record alone. What matters here is only Zeta's position RELATIVE to
    # Alpha's.
    assert order.index("Zeta") < order.index("Alpha"), order
    by_team = {e["team"]: e for e in out}
    assert by_team["Zeta"]["resolved_by"] == "opponents_cumulative_conf_pct"
    assert by_team["Alpha"]["resolved_by"] == "opponents_cumulative_conf_pct"


# ---------------------------------------------------------------------------
# Divisions -- the Sun Belt is the only conference of the ten that still has them
# ---------------------------------------------------------------------------
def _div_entry(team, overall, conf, division, rank=None):
    entry = _entry(team, overall, conf, rank)
    entry["division"] = division
    return entry


def test_divisions_are_built_for_every_member_including_a_none():
    """Three of the Sun Belt's steps are division-scoped and all of them decline without a map,
    so the map has to reach the context. A conference that plays no divisions still gets an entry
    per team with value None, so a scoped step can tell "no divisions here" from "this team is
    missing from the map"."""
    entries = [
        _div_entry("East1", (5, 4), (3, 2), "East"),
        _div_entry("West1", (5, 4), (3, 2), "West"),
        _entry("NoDivision", (5, 4), (3, 2)),          # no "division" key at all
    ]
    tiebreak = _build_tiebreak_inputs("Sun Belt", entries, SEASON, None)
    assert tiebreak.divisions == {"East1": "East", "West1": "West", "NoDivision": None}


def test_sun_belt_divisional_record_decides_a_tie_through_the_wiring():
    """End-to-end for the one conference with divisions: two East teams level on ALL conference
    games -- which is how the Sun Belt defines a division champion -- separated by their
    DIVISIONAL records at step 2. If the division map failed to reach the context, that step
    would decline and the chain would fall through to a different answer."""
    entries = [
        _div_entry("Ateam", (7, 2), (2, 1), "East"),
        _div_entry("Bteam", (7, 2), (2, 1), "East"),
        _div_entry("EastFoe1", (4, 5), (1, 2), "East"),
        _div_entry("EastFoe2", (4, 5), (1, 2), "East"),
        _div_entry("WestFoe", (6, 3), (3, 0), "West"),
    ]
    rows = (
        _game("Ateam", 21, "EastFoe1", 14) + _game("Ateam", 21, "EastFoe2", 14)
        + _game("WestFoe", 21, "Ateam", 14)                 # Ateam: 2-0 East, 0-1 cross
        + _game("Bteam", 21, "EastFoe1", 14) + _game("EastFoe2", 21, "Bteam", 14)
        + _game("Bteam", 21, "WestFoe", 14)                 # Bteam: 1-1 East, 1-0 cross
    )
    tiebreak = _build_tiebreak_inputs("Sun Belt", entries, SEASON, None)
    # The Sun Belt sorts each division separately in the real pipeline; mirror that here.
    east = [e for e in entries if e["division"] == "East"]
    out = _sort_conference_teams(east, rows, SEASON, None, tiebreak)
    order = [e["team"] for e in out]
    assert order.index("Ateam") < order.index("Bteam"), order
    by_team = {e["team"]: e for e in out}
    assert by_team["Ateam"]["resolved_by"] == "divisional_record"


# ---------------------------------------------------------------------------
# The non-FBS roster handed to the Big 12's total-wins step
# ---------------------------------------------------------------------------
def test_non_fbs_roster_distinguishes_unavailable_from_empty():
    """None means "roster unavailable, decline the step"; an empty set would mean "loaded, nobody
    qualifies". _fetch_non_fbs_logos substitutes {} when its read FAILS, so an empty mapping here
    is far more likely to be a failed read than a season in which no FBS team played an FCS
    opponent -- and reporting an uncapped win total as a capped one is the worse error."""
    assert _non_fbs_roster(None) is None
    assert _non_fbs_roster({}) is None


def test_non_fbs_roster_keeps_the_school_names():
    roster = _non_fbs_roster({"Mercer": ["u"], "Elon": None})
    assert roster == frozenset({"Mercer", "Elon"})


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------
def test_a_config_that_cannot_be_loaded_degrades_to_the_pre_engine_ordering(monkeypatch, caplog):
    """A malformed or missing rule file must not fail the artifact publish. Every conference
    falls back to the known-good pre-engine ordering, and the failure is logged once."""
    monkeypatch.setattr(schedule_mod, "_TIEBREAKER_CONFIG", None)
    monkeypatch.setattr(schedule_mod, "_TIEBREAKER_CONFIG_FAILED", False)
    monkeypatch.setattr(
        schedule_mod, "load_conference_rules",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no such file")),
    )
    entries = [
        _entry("Zeta", (8, 1), (3, 1)),
        _entry("Yankee", (8, 1), (3, 1)),
        _entry("Xray", (8, 1), (3, 1)),
    ]
    rows = (
        _game("Zeta", 30, "Yankee", 10)
        + _game("Zeta", 30, "Xray", 10)
        + _game("Yankee", 30, "Xray", 10)
    )
    with caplog.at_level(logging.ERROR, logger="cfb_lp"):
        tiebreak = _build_tiebreak_inputs(CONFIGURED, entries, SEASON, None)
        out = _sort_conference_teams(entries, rows, SEASON, None, tiebreak)
    assert tiebreak.rules is None
    assert [e["team"] for e in out] == ["Xray", "Yankee", "Zeta"]
    assert "tiebreaker config" in caplog.text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
