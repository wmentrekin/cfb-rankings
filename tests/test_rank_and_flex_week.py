"""Tests for T5 (per-team `rank` field) and T6 (Pac-12 flex-week `tbd` cells) in
artifacts/schedule.py.

T5 covers build_schedule_payload's new `team_ranks` parameter -- the pure-function half of
_fetch_team_ranks (the DB-facing half is not exercised here, per this module's existing
I/O-boundary split; see test conventions in tests/test_division_grouping.py). `team_ranks` dicts
below are built the same way _fetch_team_ranks builds them: via
artifacts/rankings.py's compute_rank_and_delta(ratings_df, None), reused rather than
reimplemented.

T6 covers FLEX_WEEK_TBD_CONFIG / _flex_week_tbd_slot_ids and their effect on
_build_team_weeks's fallback cell (tbd vs bye), using real 2026 Pac-12 school names and the
real week-12/week-13 date ranges confirmed in tests/test_week_resolution.py's fixture
(week 12: 2026-11-17..2026-11-22, week 13: 2026-11-24..2026-11-29).

Run: python -m pytest tests/ -q   (or: python tests/test_rank_and_flex_week.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from artifacts.rankings import compute_rank_and_delta  # noqa: E402
from artifacts.schedule import build_schedule_payload, _flex_week_tbd_slot_ids  # noqa: E402

SEASON = 2026


def _teams_meta(team_names, conference):
    return {t: {"conference": conference, "division": None, "logos": None} for t in team_names}


def _team_ranks(team_rating_pairs):
    """Mirrors _fetch_team_ranks's own derivation: a ratings df -> compute_rank_and_delta(...,
    None) -> Dict[team -> rank]."""
    df = pd.DataFrame(team_rating_pairs, columns=["team", "rating"])
    ranked = compute_rank_and_delta(df, None)
    return dict(zip(ranked["team"], ranked["rank"].astype(int)))


def _game(game_id, team, opponent, status, conference, start_date, conference_game=True,
          season_type="regular"):
    """One team-oriented schedule_grid row (same convention as
    tests/test_division_grouping.py's _game -- the shape build_schedule_payload consumes)."""
    return dict(
        game_id=game_id, season=SEASON, season_type=season_type, team=team, opponent=opponent,
        conference=conference, conference_game=conference_game, status=status,
        start_date=start_date, home_away="home", neutral_site=False,
    )


def _find_team(payload, team_name):
    for conf in payload["conferences"]:
        for t in conf["teams"]:
            if t["team"] == team_name:
                return t
    raise AssertionError(f"{team_name!r} not found in any emitted conference")


def _find_week_cell(team_entry, slot_id):
    for wk in team_entry["weeks"]:
        if wk["slot_id"] == slot_id:
            return wk
    raise AssertionError(f"slot {slot_id!r} not found for team {team_entry['team']!r}")


# ---------------------------------------------------------------------------
# T5.1: every team gets a rank, contiguous 1..N with no duplicates, ordered by
# rating descending.
# ---------------------------------------------------------------------------
def test_rank_is_contiguous_and_ordered_by_rating_descending():
    teams = ["Alpha", "Bravo", "Charlie"]
    team_ranks = _team_ranks([("Alpha", 10.0), ("Bravo", 30.0), ("Charlie", 20.0)])
    payload = build_schedule_payload([], _teams_meta(teams, "ACC"), SEASON, team_ranks)

    ranks_by_team = {t["team"]: t["rank"] for conf in payload["conferences"] for t in conf["teams"]}
    assert set(ranks_by_team.keys()) == set(teams)
    assert sorted(ranks_by_team.values()) == [1, 2, 3]  # contiguous, no duplicates
    # Highest rating (Bravo, 30.0) is rank 1; lowest (Alpha, 10.0) is rank 3.
    assert ranks_by_team["Bravo"] == 1
    assert ranks_by_team["Charlie"] == 2
    assert ranks_by_team["Alpha"] == 3


# ---------------------------------------------------------------------------
# T5.2: a season with no ratings rows at all emits rank: null for every team
# and does not raise.
# ---------------------------------------------------------------------------
def test_no_ratings_emits_null_rank_for_every_team_without_raising():
    teams = ["Alpha", "Bravo", "Charlie"]
    # This is exactly what _fetch_team_ranks returns when MAX(week) is NULL for the season.
    team_ranks = {}
    payload = build_schedule_payload([], _teams_meta(teams, "ACC"), SEASON, team_ranks)

    ranks_by_team = {t["team"]: t["rank"] for conf in payload["conferences"] for t in conf["teams"]}
    assert set(ranks_by_team.keys()) == set(teams)
    assert all(r is None for r in ranks_by_team.values())

    # Also cover the case where team_ranks is omitted entirely (default None -> {}).
    payload_default = build_schedule_payload([], _teams_meta(teams, "ACC"), SEASON)
    ranks_default = {t["team"]: t["rank"] for conf in payload_default["conferences"] for t in conf["teams"]}
    assert all(r is None for r in ranks_default.values())


# ---------------------------------------------------------------------------
# T5.3: a team present in the grid but absent from ratings emits rank: null
# rather than raising or shifting other teams' ranks.
# ---------------------------------------------------------------------------
def test_team_missing_from_ratings_gets_null_rank_others_unaffected():
    teams = ["Alpha", "Bravo", "Charlie"]  # Charlie is in the grid but has no ratings row.
    team_ranks = _team_ranks([("Alpha", 50.0), ("Bravo", 40.0)])
    payload = build_schedule_payload([], _teams_meta(teams, "ACC"), SEASON, team_ranks)

    ranks_by_team = {t["team"]: t["rank"] for conf in payload["conferences"] for t in conf["teams"]}
    assert ranks_by_team["Alpha"] == 1
    assert ranks_by_team["Bravo"] == 2
    assert ranks_by_team["Charlie"] is None  # not shifted into rank 3, not raised


PAC12_TEAMS = [
    "Boise State", "Colorado State", "Fresno State", "Oregon State",
    "San Diego State", "Texas State", "Utah State", "Washington State",
]


def _flex_week_scenario(announce_flex_game_for=None):
    """Builds a payload where all 8 real 2026 Pac-12 teams have a week-12 conference game and
    NO week-13 row (bucket 13 absent, exactly like a real bye), one Big Ten team has an ordinary
    bye at week 13 (no game, and not in the configured conference), and one ACC team has a real
    week-13 game (so the week-13 canonical column exists in the data at all, matching the real
    2026 shape where other conferences do play that week).

    If `announce_flex_game_for` names a Pac-12 team, that team ALSO gets a real week-13 row
    (conference_game=False, per the verified fact that flex games arrive that way) -- simulating
    the flex game being announced/ingested, to test self-clearing.
    """
    rows = []
    for i, team in enumerate(PAC12_TEAMS):
        rows.append(_game(1000 + i, team, f"Pac12Opp{i}", "win", "Pac-12", "2026-11-21 20:00:00"))
    if announce_flex_game_for is not None:
        rows.append(_game(2000, announce_flex_game_for, "FlexOpponent", "upcoming", "Pac-12",
                           "2026-11-28 20:00:00", conference_game=False))

    rows.append(_game(3000, "Ohio State", "B1GOpp", "win", "Big Ten", "2026-11-21 20:00:00"))
    # Ohio State has no week-13 row at all -> ordinary bye, unaffected by the Pac-12 rule.

    rows.append(_game(4000, "Duke", "ACCOpp", "win", "ACC", "2026-11-28 20:00:00"))
    # Duke has a real week-13 game -> that's what opens the week-13 canonical column.

    teams_meta = {}
    teams_meta.update(_teams_meta(PAC12_TEAMS, "Pac-12"))
    teams_meta.update(_teams_meta(["Ohio State"], "Big Ten"))
    teams_meta.update(_teams_meta(["Duke"], "ACC"))

    return build_schedule_payload(rows, teams_meta, SEASON)


# ---------------------------------------------------------------------------
# T6.1: all teams in the configured conference show `tbd` in the configured
# bucket.
# ---------------------------------------------------------------------------
def test_all_pac12_teams_show_tbd_in_week_13():
    assert _flex_week_tbd_slot_ids(SEASON, "Pac-12") == {"week-13"}

    payload = _flex_week_scenario()
    for team in PAC12_TEAMS:
        cell = _find_week_cell(_find_team(payload, team), "week-13")
        assert cell["status"] == "tbd"
        assert cell["opponent"] is None


# ---------------------------------------------------------------------------
# T6.2: no team in any other conference gains a `tbd` cell.
# ---------------------------------------------------------------------------
def test_no_other_conference_gains_a_tbd_cell():
    assert _flex_week_tbd_slot_ids(SEASON, "Big Ten") == set()
    assert _flex_week_tbd_slot_ids(SEASON, "ACC") == set()

    payload = _flex_week_scenario()

    # Ohio State (Big Ten): no week-13 row -> an ordinary bye, not tbd.
    ohio_cell = _find_week_cell(_find_team(payload, "Ohio State"), "week-13")
    assert ohio_cell["status"] == "bye"

    # Duke (ACC): has a real week-13 game -> renders as that game, not tbd.
    duke_cell = _find_week_cell(_find_team(payload, "Duke"), "week-13")
    assert duke_cell["status"] != "tbd"
    assert duke_cell["opponent"] == "ACCOpp"

    # No cell anywhere outside the Pac-12 is 'tbd'.
    for conf in payload["conferences"]:
        if conf["name"] == "PAC 12":
            continue
        for t in conf["teams"]:
            for wk in t["weeks"]:
                assert wk["status"] != "tbd", f"unexpected tbd cell for {t['team']} at {wk['slot_id']}"


# ---------------------------------------------------------------------------
# T6.3: self-clearing -- when a real game IS present in that bucket, the cell
# renders as that game, not as TBD.
# ---------------------------------------------------------------------------
def test_flex_week_self_clears_once_a_real_game_is_ingested():
    payload = _flex_week_scenario(announce_flex_game_for="Boise State")

    boise_cell = _find_week_cell(_find_team(payload, "Boise State"), "week-13")
    assert boise_cell["status"] != "tbd"
    assert boise_cell["opponent"] == "FlexOpponent"

    # Every other Pac-12 team (flex game still unannounced for them) still shows tbd.
    for team in PAC12_TEAMS:
        if team == "Boise State":
            continue
        cell = _find_week_cell(_find_team(payload, team), "week-13")
        assert cell["status"] == "tbd"


if __name__ == "__main__":
    test_rank_is_contiguous_and_ordered_by_rating_descending()
    test_no_ratings_emits_null_rank_for_every_team_without_raising()
    test_team_missing_from_ratings_gets_null_rank_others_unaffected()
    test_all_pac12_teams_show_tbd_in_week_13()
    test_no_other_conference_gains_a_tbd_cell()
    test_flex_week_self_clears_once_a_real_game_is_ingested()
    print("All tests passed!")
