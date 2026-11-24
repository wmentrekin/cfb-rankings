"""scripts/rerun_published_weeks.py: the parts that can run without a database or R2."""
import importlib.util
import io
import json
from pathlib import Path

import pandas as pd  # type: ignore
import pytest  # type: ignore

import model.model as weekly_model
import model.prior_model as prior_model
import model.process_data as process_data

_SPEC = importlib.util.spec_from_file_location(
    "rerun_published_weeks", Path(__file__).resolve().parent.parent / "scripts" / "rerun_published_weeks.py")
rerun = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rerun)

TEAMS = ["Alpha", "Bravo", "Charlie", "Delta"]


def _game(week, home, hs, away, as_):
    winner = home if hs > as_ else away
    return {"season": 2026, "week": week, "home_team": home, "home_score": hs, "away_team": away,
            "away_score": as_, "margin": abs(hs - as_), "winner": winner, "alpha": 0.8 if winner == home else 1.2}


GAMES = pd.DataFrame([
    _game(1, "Alpha", 24, "Bravo", 21), _game(1, "Charlie", 17, "Delta", 27),
    _game(2, "Alpha", 35, "Charlie", 14), _game(2, "Bravo", 38, "Delta", 10),
])


@pytest.fixture
def no_database(monkeypatch):
    monkeypatch.setattr(process_data, "get_games_by_year_week",
                        lambda year, week=None: (GAMES[GAMES["week"] <= week] if week else GAMES).copy())
    monkeypatch.setattr(process_data, "get_fbs_teams_by_year", lambda year: pd.DataFrame({"school": TEAMS}))


def test_solve_all_solves_each_week_and_the_prior_season_once(no_database, monkeypatch):
    calls = []
    real = prior_model.get_prior_ratings

    def counting(year, *args, **kwargs):
        calls.append(year)
        return real(year, *args, **kwargs)

    monkeypatch.setattr(weekly_model, "get_prior_ratings", counting)
    before = weekly_model.get_prior_ratings
    solved = rerun.solve_all([(2026, 1), (2026, 2)])

    assert sorted(solved) == [(2026, 1), (2026, 2)]
    assert calls == [2025]  # one prior solve, shared by both weeks
    assert weekly_model.get_prior_ratings is before  # the patch is undone
    week2 = solved[(2026, 2)]
    assert sorted(week2["team"]) == TEAMS  # the FCS stand-in is not a row
    assert set(week2["week"]) == {2} and set(week2["season"]) == {2026}
    assert week2.sort_values("rating", ascending=False).iloc[0]["team"] == "Alpha"
    assert week2["rating"].min() == pytest.approx(0.01)


class _FakeR2:
    def __init__(self, payload):
        self.payload = payload
        self.keys = []

    def get_object(self, Bucket, Key):
        self.keys.append(Key)
        return {"Body": io.BytesIO(json.dumps(self.payload).encode())}


def _df():
    return pd.DataFrame([
        {"team": "Alpha", "rating": 12.345, "season": 2026, "week": 3},
        {"team": "Bravo", "rating": 4.0, "season": 2026, "week": 3},
    ])


def test_verify_accepts_a_matching_file_and_reads_the_zero_padded_key(capsys):
    client = _FakeR2({"rankings": [{"team": "Alpha", "rating": 12.35}, {"team": "Bravo", "rating": 4.0}]})
    rerun.verify_rankings(client, "bucket", 2026, 3, _df())
    assert client.keys == ["rankings/2026/week-03.json"]
    assert "VERIFIED" in capsys.readouterr().out


@pytest.mark.parametrize("payload", [
    {"rankings": [{"team": "Bravo", "rating": 12.35}, {"team": "Alpha", "rating": 4.0}]},  # wrong leader
    {"rankings": [{"team": "Alpha", "rating": 30.1}, {"team": "Bravo", "rating": 4.0}]},    # stale rating
    {"rankings": [{"team": "Alpha", "rating": 12.35}]},                                     # missing team
])
def test_verify_fails_on_a_file_that_does_not_match_this_run(payload):
    with pytest.raises(SystemExit) as stop:
        rerun.verify_rankings(_FakeR2(payload), "bucket", 2026, 3, _df())
    assert stop.value.code == 1


def test_verify_fails_when_one_team_far_down_the_file_is_stale():
    payload = {"rankings": [{"team": "Alpha", "rating": 12.35}, {"team": "Bravo", "rating": 0.01}]}
    with pytest.raises(SystemExit):
        rerun.verify_rankings(_FakeR2(payload), "bucket", 2026, 3, _df())


STARTED = rerun.datetime(2026, 11, 24, 12, 0, tzinfo=rerun.timezone.utc)


def test_fresh_check_accepts_a_file_written_during_the_run_and_rejects_an_older_one():
    rerun.verify_fresh(_FakeR2({"generated_at_utc": "2026-11-24T12:03:00+00:00"}), "b", "schedule/2026/latest.json", STARTED)
    for stale in ("2026-11-23T23:40:00+00:00", None, "not a time"):
        with pytest.raises(SystemExit):
            rerun.verify_fresh(_FakeR2({"generated_at_utc": stale}), "b", "schedule/2026/latest.json", STARTED)


def test_latest_check_fails_when_the_keys_point_at_another_week():
    rerun.verify_latest(_FakeR2({"season": 2026, "week": 12}), "b", 2026, 12)
    with pytest.raises(SystemExit):
        rerun.verify_latest(_FakeR2({"season": 2025, "week": 15}), "b", 2026, 12)


def _wire(monkeypatch, order, newest=(2026, 12)):
    monkeypatch.setenv("R2_BUCKET_NAME", "bucket")
    monkeypatch.setattr(rerun, "get_r2_client", lambda: object())
    monkeypatch.setattr(rerun, "publish_rankings_artifact", lambda s, w: order.append(("rankings", s, w)))
    monkeypatch.setattr(rerun, "verify_rankings", lambda *a: None)
    monkeypatch.setattr(rerun, "verify_fresh", lambda *a: None)
    monkeypatch.setattr(rerun, "verify_latest", lambda c, b, s, w: order.append(("latest", s, w)))
    monkeypatch.setattr(rerun, "publish_schedule_artifact", lambda s: order.append(("grid", s)))
    monkeypatch.setattr(rerun, "publish_playoffs_artifact", lambda s: order.append(("playoffs", s)))
    monkeypatch.setattr(rerun, "newest_week", lambda engine: newest)


def test_publish_stops_before_uploading_when_r2_is_not_configured(monkeypatch):
    published = []
    monkeypatch.setattr(rerun, "get_r2_client", lambda: None)
    monkeypatch.setattr(rerun, "publish_rankings_artifact", lambda s, w: published.append((s, w)))
    with pytest.raises(SystemExit):
        rerun.publish_all(None, {(2026, 3): _df()}, STARTED)
    assert published == []


def test_publish_ends_on_the_newest_week_in_the_database(monkeypatch):
    """A --season 2025 run publishes only 2025 weeks, each of which rewrites the global
    "latest" key. The last publish must be the newest week in the database, then checked."""
    order = []
    _wire(monkeypatch, order)
    rerun.publish_all(None, {(2025, 15): _df(), (2025, 14): _df()}, STARTED)
    assert order == [
        ("rankings", 2025, 14), ("rankings", 2025, 15), ("grid", 2025), ("playoffs", 2025),
        ("rankings", 2026, 12), ("latest", 2026, 12),
    ]


def test_publish_repoints_latest_even_when_a_week_fails_its_check(monkeypatch):
    order = []
    _wire(monkeypatch, order)

    def failing(client, bucket, season, week, df):
        if (season, week) == (2026, 2):
            rerun._fail("stale")

    monkeypatch.setattr(rerun, "verify_rankings", failing)
    with pytest.raises(SystemExit):
        rerun.publish_all(None, {(2026, 1): _df(), (2026, 2): _df(), (2026, 3): _df()}, STARTED)
    assert order == [("rankings", 2026, 1), ("rankings", 2026, 2), ("rankings", 2026, 12), ("latest", 2026, 12)]


def test_playoffs_are_republished_only_for_seasons_that_have_playoff_rules(monkeypatch):
    order = []
    _wire(monkeypatch, order, newest=(2024, 15))
    rerun.publish_all(None, {(2024, 15): _df()}, STARTED)
    assert ("playoffs", 2024) not in order and ("grid", 2024) in order


def test_drift_report_names_added_dropped_and_changed_teams(capsys):
    solved = {(2026, 3): pd.DataFrame([
        {"team": "Alpha", "rating": 5.0, "wins": 3, "losses": 0},
        {"team": "Bravo", "rating": 1.0, "wins": 1, "losses": 2},
        {"team": "Newcomer", "rating": 0.01, "wins": 0, "losses": 3},
    ])}
    stored = {(2026, 3): {"Alpha": (3, 0), "Bravo": (2, 1), "Departed": (0, 3)}}
    rerun.print_drift(solved, stored)
    out = capsys.readouterr().out
    assert "['Newcomer']" in out and "['Departed']" in out and "['Bravo']" in out
    rerun.print_drift({(2026, 3): solved[(2026, 3)].iloc[:1]}, {(2026, 3): {"Alpha": (3, 0)}})
    assert "none: every week" in capsys.readouterr().out
