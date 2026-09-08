"""Regression tests for defects found by the independent review of PR #11.

Each of these covers a gap the existing 50 tests did not, and each was reproduced
before being fixed.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from artifacts.schedule import _sort_conference_teams, build_schedule_payload  # noqa: E402


def _entry(team, w, l, cw=None, cl=None):
    return {"team": team, "record": {"wins": w, "losses": l},
            "conf_record": None if cw is None else {"wins": cw, "losses": cl}}


def test_pandas_null_division_does_not_empty_a_conference():
    """A NULL division must not silently delete every non-divisional conference.

    teams.division is NULL for every team outside a divisional conference. pandas 2.x
    turns that into None, but pandas 3.x turns it into float('nan') for a text column --
    and nan is poison for this code: `nan is None` is False and `nan == nan` is False, so
    the division grouping matches nothing and the conference publishes with an empty
    teams list. No exception, no log, the conference just vanishes.

    The tests that existed when this shipped all built teams_meta from synthetic dicts
    containing real None, so none of them touched the DataFrame path this comes through.
    This one reproduces that path deliberately.
    """
    df = pd.DataFrame({
        "school": ["Duke", "Clemson", "App State", "Troy"],
        "conference": ["ACC", "ACC", "Sun Belt", "Sun Belt"],
        "division": [None, None, "East", "West"],
        "logos": [None, None, None, None],
    })
    # Exactly what _fetch_teams_meta produces, including its null normalisation.
    meta = {
        r["school"]: {
            "conference": None if pd.isnull(r["conference"]) else r["conference"],
            "division": None if pd.isnull(r["division"]) else r["division"],
            "logos": r["logos"],
        }
        for _, r in df.iterrows()
    }
    payload = build_schedule_payload([], meta, 2026)
    by_name = {c["name"]: len(c["teams"]) for c in payload["conferences"]}
    assert by_name.get("ACC") == 2, f"non-divisional conference lost its teams: {by_name}"
    assert by_name.get("SBC") == 2, f"divisional conference lost its teams: {by_name}"


def test_an_unplayed_team_does_not_cancel_a_head_to_head_tiebreak():
    """A 0-0 team must not join a two-way tie and disable the head-to-head swap.

    Giving an unplayed conference record 0.5 puts it level with every 1-1, 2-2 and 3-3
    team. The head-to-head tiebreak only fires on a group of exactly two, so a single
    0-0 team joining two genuinely tied teams grows the group to three and cancels the
    swap -- displaying the loser of that head-to-head game above the winner.
    """
    rows = []
    for team, opp, own, other, status in (("Charlie", "Bravo", 21, 7, "win"),
                                          ("Bravo", "Charlie", 7, 21, "loss")):
        rows.append({"game_id": 1, "season": 2026, "season_type": "regular", "team": team,
                     "opponent": opp, "conference": "ACC", "conference_game": True,
                     "team_score": own, "opp_score": other,
                     "start_date": "2026-10-10 20:00:00", "status": status})

    pair = [_entry("Bravo", 1, 1, 1, 1), _entry("Charlie", 1, 1, 1, 1)]
    assert [e["team"] for e in _sort_conference_teams(list(pair), rows, 2026)][:2] == ["Charlie", "Bravo"]

    with_unplayed = pair + [_entry("Alpha", 0, 0, 0, 0)]
    order = [e["team"] for e in _sort_conference_teams(with_unplayed, rows, 2026)]
    assert order[:2] == ["Charlie", "Bravo"], (
        f"an unplayed team cancelled the head-to-head tiebreak: {order}"
    )


def test_overall_record_sentinel_is_covered():
    """Pins the OVERALL-record sentinel, which no test exercised.

    Mutating artifacts/schedule.py's _overall_pct sentinel from 0.5 back to -1.0 left all
    50 tests green while genuinely changing output. Independents have no conference record,
    so they sort purely on overall percentage -- which is exactly where that sentinel is
    load-bearing and where the gap showed.
    """
    independents = [_entry("Winless", 0, 1), _entry("Unplayed", 0, 0), _entry("Winner", 1, 0)]
    order = [e["team"] for e in _sort_conference_teams(independents, [], 2026)]
    assert order == ["Winner", "Unplayed", "Winless"], (
        f"unplayed overall record is not sorting between a win and a loss: {order}"
    )


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
