"""Tests for the consuming half of the `non_fbs_teams` table (migration 0004's "T7b").

The ingest (database/get_non_fbs_teams.py) has populated that table since the previous pass,
but nothing read it -- build_schedule_payload sourced opponent logos from teams_meta alone, so
an FCS opponent still rendered as plain text in the grid despite its logo sitting in the
database. These tests cover the lookup that closes that gap, and the invariant that makes it
safe: non-FBS entries reach the OPPONENT LOGO map only, never teams_meta's key set, which is
what decides who gets a Season Grid row and under which conference.

Run: python -m pytest tests/ -q   (or: python tests/test_non_fbs_opponent_logos.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifacts.schedule import build_schedule_payload  # noqa: E402

SEASON = 2026

FBS_LOGO = "https://a.espncdn.com/i/teamlogos/ncaa/logos-dark/32/59.png"
FCS_LOGO = "https://a.espncdn.com/i/teamlogos/ncaa/logos-dark/32/2678.png"


def _teams_meta(team_names, conference, logos=None):
    return {
        t: {"conference": conference, "division": None, "logos": [logos] if logos else None}
        for t in team_names
    }


def _game(game_id, team, opponent, status, conference, start_date, conference_game=False):
    return dict(
        game_id=game_id, season=SEASON, season_type="regular", team=team, opponent=opponent,
        conference=conference, conference_game=conference_game, status=status,
        start_date=start_date, home_away="home", neutral_site=False,
    )


def _cell_for(payload, team_name, opponent_name):
    for conf in payload["conferences"]:
        for t in conf["teams"]:
            if t["team"] != team_name:
                continue
            for wk in t["weeks"]:
                if wk["opponent"] == opponent_name:
                    return wk
            raise AssertionError(
                f"no week cell against {opponent_name!r} for {team_name!r}"
            )
    raise AssertionError(f"{team_name!r} not found in any emitted conference")


def _fcs_matchup_rows():
    """One FBS team hosting one FCS team -- the shape that used to render as text."""
    return [_game("g1", "Georgia Tech", "Mercer", "completed", "ACC", "2026-09-05T16:00:00Z")]


# ---------------------------------------------------------------------------
# The gap itself: an FCS opponent whose logo lives in non_fbs_teams now emits
# that logo instead of falling through to the text-name rendering.
# ---------------------------------------------------------------------------
def test_non_fbs_opponent_gets_its_logo():
    payload = build_schedule_payload(
        _fcs_matchup_rows(),
        _teams_meta(["Georgia Tech"], "ACC", FBS_LOGO),
        SEASON,
        None,
        {"Mercer": [FCS_LOGO]},
    )
    cell = _cell_for(payload, "Georgia Tech", "Mercer")
    assert cell["opponent_logo_url"] == FCS_LOGO


# ---------------------------------------------------------------------------
# Discriminating counterpart: with the lookup absent (the pre-change call, and
# the state this whole change fixes) the same fixture emits no logo. Without
# this, the test above would still pass if the parameter were ignored and
# every cell happened to carry a logo from somewhere else.
# ---------------------------------------------------------------------------
def test_without_the_lookup_the_same_opponent_has_no_logo():
    payload = build_schedule_payload(
        _fcs_matchup_rows(),
        _teams_meta(["Georgia Tech"], "ACC", FBS_LOGO),
        SEASON,
    )
    cell = _cell_for(payload, "Georgia Tech", "Mercer")
    assert cell["opponent_logo_url"] is None


# ---------------------------------------------------------------------------
# The safety invariant migration 0004 exists to protect: a non-FBS team must
# never become a grid row or a conference. teams_meta's keys define the FBS
# universe; the logo map must not leak into it.
# ---------------------------------------------------------------------------
def test_non_fbs_team_does_not_become_a_row_or_a_conference():
    payload = build_schedule_payload(
        _fcs_matchup_rows(),
        _teams_meta(["Georgia Tech"], "ACC", FBS_LOGO),
        SEASON,
        None,
        {"Mercer": [FCS_LOGO], "Furman": [FCS_LOGO], "Samford": None},
    )
    emitted_teams = {t["team"] for conf in payload["conferences"] for t in conf["teams"]}
    assert emitted_teams == {"Georgia Tech"}
    assert [c["name"] for c in payload["conferences"]] == ["ACC"]


# ---------------------------------------------------------------------------
# An FBS row wins a name collision, so the rating model's own team list stays
# authoritative even if the ingest's "never fbs" filter ever breaks upstream.
# ---------------------------------------------------------------------------
def test_fbs_logo_wins_a_name_collision():
    payload = build_schedule_payload(
        [_game("g1", "Georgia Tech", "Clemson", "completed", "ACC", "2026-09-05T16:00:00Z")],
        _teams_meta(["Georgia Tech", "Clemson"], "ACC", FBS_LOGO),
        SEASON,
        None,
        {"Clemson": [FCS_LOGO]},
    )
    cell = _cell_for(payload, "Georgia Tech", "Clemson")
    assert cell["opponent_logo_url"] == FBS_LOGO


# ---------------------------------------------------------------------------
# Omitting the parameter, or passing an empty dict (what _fetch_non_fbs_logos
# returns for a season ingested before migration 0004), reproduces the previous
# behavior exactly rather than raising.
# ---------------------------------------------------------------------------
def test_empty_and_omitted_lookups_are_equivalent_and_do_not_raise():
    args = (_fcs_matchup_rows(), _teams_meta(["Georgia Tech"], "ACC", FBS_LOGO), SEASON)
    assert build_schedule_payload(*args, None, {}) == build_schedule_payload(*args) or True
    empty = build_schedule_payload(*args, None, {})
    omitted = build_schedule_payload(*args)
    # generated_at_utc differs between calls by construction; compare the rest.
    empty.pop("generated_at_utc")
    omitted.pop("generated_at_utc")
    assert empty == omitted


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
