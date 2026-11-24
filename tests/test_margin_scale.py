"""The margin scale must be built from margins, not from the home/away multiplier.

Regression tests for the bug where both models took the "median margin" from position 5 of a
game tuple (alpha: 0.8 / 1.0 / 1.2) instead of position 4 (margin). See model/margin_scale.py.

Nothing here touches a database: the two loaders in model.process_data are replaced with
functions returning small DataFrames, and the real model code runs on top of them.
"""
import math

import pandas as pd  # type: ignore
import pytest  # type: ignore

import model.model as weekly_model
import model.prior_model as prior_model
import model.process_data as process_data
from model.margin_scale import (
    GAME_ALPHA_INDEX,
    GAME_MARGIN_INDEX,
    compute_k_margin,
    median_margin,
)

TEAMS = ["Alpha", "Bravo", "Charlie", "Delta"]


def _game(week, home, home_score, away, away_score):
    winner = home if home_score > away_score else away
    return {
        "season": 2026, "week": week, "home_team": home, "home_score": home_score,
        "away_team": away, "away_score": away_score, "margin": abs(home_score - away_score),
        "winner": winner, "alpha": 0.8 if winner == home else 1.2,
    }


def _games_df():
    # Margins 3, 10, 21, 28, 35, 14 -> median 17.5. Every alpha is 0.8 or 1.2.
    return pd.DataFrame([
        _game(1, "Alpha", 24, "Bravo", 21),
        _game(1, "Charlie", 17, "Delta", 27),
        _game(2, "Alpha", 35, "Charlie", 14),
        _game(2, "Bravo", 38, "Delta", 10),
        _game(3, "Alpha", 42, "Delta", 7),
        _game(3, "Bravo", 28, "Charlie", 14),
    ])


def _teams_df():
    return pd.DataFrame({"school": TEAMS})


def _process():
    return process_data.process_game_data(_games_df(), _teams_df())


def test_game_tuple_positions_match_the_named_indices():
    """The indices margin_scale relies on are the ones process_game_data really produces."""
    _teams, games, _fcs, _records, _conn = _process()
    first = games[0]  # Alpha 24, Bravo 21 at home
    assert len(first) == 8
    assert first[GAME_MARGIN_INDEX] == 3
    assert first[GAME_ALPHA_INDEX] == pytest.approx(0.8)


def test_median_is_taken_over_margins_not_alpha():
    _teams, games, _fcs, _records, _conn = _process()
    assert median_margin(games) == pytest.approx(17.5)
    # The bug produced a median of about 1 (the alphas), hence k of about 7.
    assert compute_k_margin(games, 7.0) == pytest.approx(7.0 / math.sqrt(17.5))


def test_median_game_demands_exactly_the_target_gap():
    games = [("A", "B", 0, "A", m, 1.0, 1, 2026) for m in (3, 7, 14, 21, 35)]
    k = compute_k_margin(games, 7.0)
    assert k * math.sqrt(14) == pytest.approx(7.0)


def test_no_games_falls_back_to_a_median_of_one():
    assert median_margin([]) == 1.0
    assert compute_k_margin([], 7.0) == pytest.approx(7.0)


def test_missing_and_negative_margins_are_left_out():
    games = [
        ("A", "B", 0, "A", None, 1.0, 1, 2026),
        ("A", "C", 0, "A", float("nan"), 1.0, 1, 2026),
        ("A", "D", 0, "A", -5, 1.0, 1, 2026),
        ("B", "C", 0, "B", 10, 1.0, 1, 2026),
    ]
    assert median_margin(games) == 10.0


def test_all_zero_margins_do_not_divide_by_zero():
    games = [("A", "B", 0, "A", 0, 1.0, 1, 2026)]
    assert math.isfinite(compute_k_margin(games, 7.0))


@pytest.fixture
def no_database(monkeypatch):
    monkeypatch.setattr(process_data, "get_games_by_year_week", lambda year, week=None: _games_df())
    monkeypatch.setattr(process_data, "get_fbs_teams_by_year", lambda year: _teams_df())


@pytest.mark.parametrize("module", [weekly_model, prior_model])
def test_both_models_ask_the_shared_helper_for_the_scale(module, no_database, monkeypatch):
    """Neither model may compute its own scale again: each calls compute_k_margin on its games."""
    seen = []

    def spy(games, target_gap):
        seen.append((len(games), target_gap))
        return compute_k_margin(games, target_gap)

    monkeypatch.setattr(module, "compute_k_margin", spy)
    if module is weekly_model:
        monkeypatch.setattr(weekly_model, "get_prior_ratings", lambda year, params=None: {})
        weekly_model.get_ratings(2026, 8)
    else:
        prior_model.get_prior_ratings(2026)
    assert seen == [(6, 7.0)]


def test_weekly_model_orders_an_unbeaten_team_first(no_database, monkeypatch):
    """End to end on the real solver: the ratings come back and the 3-0 team leads."""
    monkeypatch.setattr(weekly_model, "get_prior_ratings", lambda year, params=None: {})
    ratings, records = weekly_model.get_ratings(2026, 8)
    fbs = [team for team in ratings if team in TEAMS]
    assert fbs[0] == "Alpha"
    assert records["Alpha"][:2] == [3, 0]
    assert ratings["Alpha"] > ratings["Bravo"] > ratings["Delta"]
