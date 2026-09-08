"""Tests for T3: carrying `division` through to the Season Grid artifact and
grouping teams by it (artifacts/schedule.py's build_schedule_payload and
_fetch_teams_meta).

Covers:
  1. A Sun Belt-shaped divisional conference emits `division` for every team,
     with all East teams ordered before all West teams and each division
     internally sorted by the existing (unmodified) _sort_conference_teams
     rules -- constructed so a West team with a BETTER conference record than
     an East team must still sort after every East team, proving grouping by
     division (not by record) governs top-level order.
  2. A non-divisional conference emits `division: null` for every team, with
     ordering unchanged from the pre-T3 single-pass sort -- constructed with
     three teams whose relative order depends on the full existing sort logic
     (win% ordering across more than a simple two-team pair), so a grouping
     regression that changed sort scoping would actually flip something.
  3. teams_meta shaped like _fetch_teams_meta's new (with `division`) return
     value flows the field through end-to-end, including the case where a
     team's meta simply has no division at all (non-Sun-Belt conferences,
     which is every team `teams.division` is NULL for today).

Run: python -m pytest tests/ -q   (or: python tests/test_division_grouping.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifacts.schedule import build_schedule_payload  # noqa: E402

SEASON = 2026


def _game(game_id, team, opponent, status, conference, start_date, conference_game=True):
    """One team-oriented schedule_grid row -- the shape build_schedule_payload consumes.
    Only the columns the code under test actually reads are populated (see
    artifacts/schedule.py and artifacts/schedule_standings.py's compute_team_records)."""
    return dict(
        game_id=game_id,
        season=SEASON,
        season_type="regular",
        team=team,
        opponent=opponent,
        conference=conference,
        conference_game=conference_game,
        status=status,
        start_date=start_date,
        home_away="home",
        neutral_site=False,
    )


def _conference_teams(payload, team_names):
    """Find the conferences[] entry containing team_names and return its `teams` list."""
    for conf in payload["conferences"]:
        names = {t["team"] for t in conf["teams"]}
        if names & set(team_names):
            return conf["teams"]
    raise AssertionError(f"none of {team_names} found in any emitted conference")


# ---------------------------------------------------------------------------
# Case 1: Sun Belt-shaped divisional conference
# ---------------------------------------------------------------------------
def test_sunbelt_division_grouping_and_internal_sort():
    rows = []
    # East: App State 2-0 conf (better), Coastal Carolina 0-2 conf (worse)
    rows += [_game(1, "App State", "OppA1", "win", "Sun Belt", "2026-08-29 19:00:00")]
    rows += [_game(2, "App State", "OppA2", "win", "Sun Belt", "2026-09-05 19:00:00")]
    rows += [_game(3, "Coastal Carolina", "OppB1", "loss", "Sun Belt", "2026-08-29 19:00:00")]
    rows += [_game(4, "Coastal Carolina", "OppB2", "loss", "Sun Belt", "2026-09-05 19:00:00")]
    # West: Arkansas State 3-0 conf (BEST record in the whole conference),
    # Troy 0-1 conf (worst). Arkansas State's record beats both East teams',
    # so if division grouping were broken and the whole conference sorted in
    # one pass by record, Arkansas State would land ahead of both East teams.
    rows += [_game(5, "Arkansas State", "OppC1", "win", "Sun Belt", "2026-08-29 19:00:00")]
    rows += [_game(6, "Arkansas State", "OppC2", "win", "Sun Belt", "2026-09-05 19:00:00")]
    rows += [_game(7, "Arkansas State", "OppC3", "win", "Sun Belt", "2026-09-12 19:00:00")]
    rows += [_game(8, "Troy", "OppD1", "loss", "Sun Belt", "2026-08-29 19:00:00")]

    teams_meta = {
        "App State": {"conference": "Sun Belt", "division": "East", "logos": None},
        "Coastal Carolina": {"conference": "Sun Belt", "division": "East", "logos": None},
        "Arkansas State": {"conference": "Sun Belt", "division": "West", "logos": None},
        "Troy": {"conference": "Sun Belt", "division": "West", "logos": None},
    }

    payload = build_schedule_payload(rows, teams_meta, SEASON)
    teams = _conference_teams(payload, teams_meta.keys())

    names_in_order = [t["team"] for t in teams]
    assert names_in_order == ["App State", "Coastal Carolina", "Arkansas State", "Troy"], (
        f"expected all East teams before all West teams (each internally sorted by record), "
        f"got {names_in_order}"
    )

    by_name = {t["team"]: t for t in teams}
    assert by_name["App State"]["division"] == "East"
    assert by_name["Coastal Carolina"]["division"] == "East"
    assert by_name["Arkansas State"]["division"] == "West"
    assert by_name["Troy"]["division"] == "West"

    # Every team entry carries the field, and only these four keys' worth of
    # division values are exercised -- no accidental extra teams.
    assert len(teams) == 4


# ---------------------------------------------------------------------------
# Case 2: non-divisional conference -- ordering must be unchanged, division null
# ---------------------------------------------------------------------------
def test_non_divisional_conference_division_null_and_order_unchanged():
    rows = []
    # Three-way record spread so the ordering genuinely exercises the full
    # existing sort (win% desc, name asc among ties) rather than a trivial
    # two-team case. If a regression grouped these into more than one bucket
    # (e.g. by team identity instead of a real division key), or sorted per
    # team instead of per conference, this order would break.
    rows += [_game(11, "Team-Win", "OppX1", "win", "ACC", "2026-08-29 19:00:00")]
    # Team-Unplayed has zero conference games (0-0 conf record; sentinel 0.5).
    rows += [_game(12, "Team-Unplayed", "OppX2", "win", "ACC", "2026-08-29 19:00:00", conference_game=False)]
    rows += [_game(13, "Team-Loss", "OppX3", "loss", "ACC", "2026-08-29 19:00:00")]

    teams_meta = {
        "Team-Win": {"conference": "ACC", "division": None, "logos": None},
        "Team-Unplayed": {"conference": "ACC", "division": None, "logos": None},
        "Team-Loss": {"conference": "ACC", "division": None, "logos": None},
    }

    payload = build_schedule_payload(rows, teams_meta, SEASON)
    teams = _conference_teams(payload, teams_meta.keys())

    names_in_order = [t["team"] for t in teams]
    assert names_in_order == ["Team-Win", "Team-Unplayed", "Team-Loss"], (
        f"non-divisional conference ordering regressed: got {names_in_order}"
    )
    for t in teams:
        assert t["division"] is None, f"{t['team']} should have division=None, got {t['division']!r}"


# ---------------------------------------------------------------------------
# Case 3: teams_meta shaped like _fetch_teams_meta's new (with division) output
# ---------------------------------------------------------------------------
def test_teams_meta_division_shape_flows_through():
    """Synthetic teams_meta matching _fetch_teams_meta's post-T3 return shape
    (Dict[school -> {"conference", "division", "logos"}]) -- exercising the
    consumer path without hitting a DB, per the task's guidance."""
    rows = [
        _game(21, "Georgia State", "OppY1", "win", "Sun Belt", "2026-08-29 19:00:00"),
        _game(22, "Georgia Southern", "OppY2", "loss", "Sun Belt", "2026-08-29 19:00:00"),
        # A team from a conference where teams.division is NULL in production today.
        _game(23, "Duke", "OppY3", "win", "ACC", "2026-08-29 19:00:00"),
    ]
    teams_meta = {
        "Georgia State": {"conference": "Sun Belt", "division": "East", "logos": ["logo1"]},
        "Georgia Southern": {"conference": "Sun Belt", "division": "East", "logos": None},
        "Duke": {"conference": "ACC", "division": None, "logos": None},
    }

    payload = build_schedule_payload(rows, teams_meta, SEASON)
    all_teams = [t for conf in payload["conferences"] for t in conf["teams"]]
    by_name = {t["team"]: t for t in all_teams}

    assert set(by_name.keys()) == {"Georgia State", "Georgia Southern", "Duke"}
    assert by_name["Georgia State"]["division"] == "East"
    assert by_name["Georgia Southern"]["division"] == "East"
    assert by_name["Duke"]["division"] is None
    # `division` present on every emitted team, additive alongside the pre-existing fields.
    for t in all_teams:
        assert set(["team", "logo_url", "record", "conf_record", "division", "weeks"]) <= set(t.keys())


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
