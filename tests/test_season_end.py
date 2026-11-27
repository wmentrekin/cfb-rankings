"""The weekly run knows when the rankings are finished (season_end.py and its wiring in main).

The rankings end with conference championship week. Before this, the weekly job rated the
Army-Navy game as a week of its own and republished it every Sunday until someone switched the
schedule off by hand.
"""
import sys
from datetime import date, datetime, timezone
from unittest import mock

import pytest  # type: ignore

import main
import season_end
from season_end import MODE_POSTSEASON, MODE_RANKINGS, SeasonState, choose_mode, default_season, final_rankings_week, season_is_finished

# The stored 2026 regular season from week 12 on, as (week, home, away): a full week 13, three
# title games in week 14, Army-Navy alone in week 15. 2025 has the same shape one week later.
GAMES_2026 = (
    [(12, f"H{i}", f"A{i}") for i in range(5)]
    + [(13, f"H{i}", f"A{i}") for i in range(5)]
    + [(14, "Liberty", "Jacksonville State"), (14, "Boise State", "San Diego State"), (14, "Miami", "Wake Forest")]
    + [(15, "Navy", "Army")]
)


def test_final_week_is_the_last_one_with_a_game_other_than_army_navy():
    assert final_rankings_week(GAMES_2026) == 14
    assert final_rankings_week([(15, "Army", "Navy")] + GAMES_2026[:-1]) == 14  # either order
    assert final_rankings_week([(16, "Navy", "Army"), (15, "A", "B"), (14, "C", "D")]) == 15  # 2025's shape
    assert final_rankings_week([]) is None
    assert final_rankings_week([(15, "Navy", "Army")]) is None


def test_army_navy_sharing_a_week_with_other_games_does_not_hide_that_week():
    """2027: Army-Navy moves to the same weekend as any remaining title games."""
    assert final_rankings_week([(14, "A", "B"), (15, "Navy", "Army"), (15, "C", "D")]) == 15


def test_default_season_belongs_to_the_previous_year_until_july():
    assert default_season(date(2026, 12, 13)) == 2026
    assert default_season(date(2027, 1, 26)) == 2026
    assert default_season(date(2027, 6, 30)) == 2026
    assert default_season(date(2027, 7, 1)) == 2027
    assert default_season(date(2027, 8, 29)) == 2027


def _mode(today, resolved, rated, final=14, explicit=False, season=2026):
    return choose_mode(today, season, resolved, final, rated, explicit)


def test_in_season_runs_are_ordinary_rankings_runs():
    d = _mode(date(2026, 11, 29), 13, rated=False)
    assert (d.mode, d.week) == (MODE_RANKINGS, 13)


def test_championship_week_is_rated_once():
    first = _mode(date(2026, 12, 6), 14, rated=False)
    assert (first.mode, first.week) == (MODE_RANKINGS, 14)
    # The dated selection-day runs on Dec 7 and 8 resolve to the same week: nothing to re-rate.
    again = _mode(date(2026, 12, 7), 14, rated=True)
    assert again.mode == MODE_POSTSEASON


def test_the_army_navy_week_is_never_a_rankings_week():
    """THE regression: Dec 13, the date-derived week is 15 (Army-Navy only)."""
    assert _mode(date(2026, 12, 13), 15, rated=True).mode == MODE_POSTSEASON
    # And every Sunday after, when the week number is pinned at its cap.
    assert _mode(date(2027, 1, 3), 16, rated=True).mode == MODE_POSTSEASON


def test_a_missed_championship_week_is_caught_up_not_skipped():
    """If the Dec 6 run failed, the Dec 13 run rates week 14, not the Army-Navy week."""
    d = _mode(date(2026, 12, 13), 15, rated=False)
    assert (d.mode, d.week) == (MODE_RANKINGS, 14)


def test_an_explicit_week_is_always_a_rankings_run():
    d = _mode(date(2026, 12, 20), 14, rated=True, explicit=True)
    assert (d.mode, d.week) == (MODE_RANKINGS, 14)


def test_before_december_a_short_stored_schedule_cannot_end_the_season():
    """If later weeks failed to load, the last stored week looks like the final one and is
    already rated. Mid-season that must stay an ordinary run."""
    d = _mode(date(2026, 10, 25), 8, rated=True, final=8)
    assert (d.mode, d.week) == (MODE_RANKINGS, 8)


def test_no_stored_games_is_an_ordinary_run():
    assert _mode(date(2026, 12, 13), 15, rated=False, final=None).mode == MODE_RANKINGS


def test_season_is_finished_only_after_a_quiet_spell():
    final = datetime(2027, 1, 26, 0, 30)  # the national championship, in UTC
    starts = [datetime(2026, 12, 12, 20, 0), final]
    assert season_is_finished(datetime(2027, 1, 31, 7, 0), starts) is False  # 5 days on
    assert season_is_finished(datetime(2027, 2, 7, 7, 0), starts) is True    # 12 days on
    assert season_is_finished(datetime(2026, 12, 13, 7, 0), starts) is False  # games still to come
    assert season_is_finished(datetime(2027, 3, 1, 7, 0), []) is False        # nothing loaded


# --- main(): what actually runs in each state ---

class _Clock(datetime):
    fixed = datetime(2026, 12, 13, 7, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.fixed if tz else cls.fixed.replace(tzinfo=None)


def _run(tmp_path, monkeypatch, *, now, resolved_week, state, argv=("main.py",)):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", list(argv))
    _Clock.fixed = now
    calls = []
    mocks = {name: mock.Mock(side_effect=lambda *a, _n=name, **k: calls.append((_n, a, k)) or 0) for name in (
        "load_games_to_db", "load_upcoming_games_to_db", "publish_rankings_artifact",
        "publish_schedule_artifact", "publish_playoffs_artifact", "insert_model_results_to_db")}
    mocks["get_ratings"] = mock.Mock(return_value=None)
    read_state = mock.Mock(side_effect=state) if isinstance(state, Exception) else mock.Mock(return_value=state)
    with mock.patch.object(main, "datetime", _Clock), \
            mock.patch.object(main, "resolve_target_week", return_value=resolved_week), \
            mock.patch.object(season_end, "read_season_state", read_state), \
            mock.patch.object(main, "teams_exist_for_year", return_value=True), \
            mock.patch.object(main, "non_fbs_logo_coverage_is_complete", return_value=True), \
            mock.patch.object(main, "regular_week_has_games", return_value=True), \
            mock.patch.object(main, "apply_game_result_overrides", return_value={"applied": [], "failures": []}), \
            mock.patch.multiple(main, **mocks):
        main.main()
    logs = sorted(tmp_path.glob("run_*.log"))
    return mocks, calls, (logs[-1].read_text() if logs else ""), read_state


def _state(rated, last_game=datetime(2027, 1, 26, 0, 30)):
    return SeasonState(final_week=14, rated_weeks=set(rated), start_dates=(datetime(2026, 12, 5, 20, 0), last_game))


def test_main_after_the_title_games_updates_results_and_publishes_no_rankings(tmp_path, monkeypatch):
    mocks, calls, log, _ = _run(
        tmp_path, monkeypatch, now=datetime(2026, 12, 13, 7, 0, tzinfo=timezone.utc),
        resolved_week=15, state=_state(range(1, 15)))
    mocks["get_ratings"].assert_not_called()
    mocks["insert_model_results_to_db"].assert_not_called()
    mocks["publish_rankings_artifact"].assert_not_called()
    assert [c[0] for c in calls] == [
        "load_upcoming_games_to_db", "load_games_to_db", "publish_schedule_artifact", "publish_playoffs_artifact"]
    assert calls[0][1] == (2026, 14)                         # games after the final week: Army-Navy
    assert calls[1][2] == {"week": None, "season_type": "postseason"}
    assert "Run mode: postseason" in log and "SEASON_COMPLETE" not in log


def test_main_in_january_still_works_on_the_season_that_started_last_year(tmp_path, monkeypatch):
    mocks, calls, log, read_state = _run(
        tmp_path, monkeypatch, now=datetime(2027, 1, 3, 7, 0, tzinfo=timezone.utc),
        resolved_week=16, state=_state(range(1, 15)))
    read_state.assert_called_once_with(2026)
    assert ("publish_playoffs_artifact", (2026,), {}) in calls
    mocks["get_ratings"].assert_not_called()


def test_main_does_nothing_once_the_season_is_over(tmp_path, monkeypatch):
    mocks, calls, log, _ = _run(
        tmp_path, monkeypatch, now=datetime(2027, 3, 7, 7, 0, tzinfo=timezone.utc),
        resolved_week=16, state=_state(range(1, 15)))
    assert calls == []
    mocks["get_ratings"].assert_not_called()
    assert "SEASON_COMPLETE" in log


def test_main_championship_week_is_an_ordinary_run(tmp_path, monkeypatch):
    mocks, calls, log, _ = _run(
        tmp_path, monkeypatch, now=datetime(2026, 12, 6, 7, 0, tzinfo=timezone.utc),
        resolved_week=14, state=_state(range(1, 14)))
    mocks["get_ratings"].assert_called_once_with(2026, 14)
    assert "Run mode: rankings" in log


def test_main_catches_up_a_missed_championship_week(tmp_path, monkeypatch):
    mocks, calls, log, _ = _run(
        tmp_path, monkeypatch, now=datetime(2026, 12, 13, 7, 0, tzinfo=timezone.utc),
        resolved_week=15, state=_state(range(1, 14)))
    mocks["get_ratings"].assert_called_once_with(2026, 14)
    assert "Publishing week 14, not the date-derived week 15" in log


def test_main_stops_in_december_when_it_cannot_tell(tmp_path, monkeypatch):
    mocks, calls, log, _ = _run(
        tmp_path, monkeypatch, now=datetime(2026, 12, 13, 7, 0, tzinfo=timezone.utc),
        resolved_week=15, state=RuntimeError("database unreachable"))
    assert calls == []
    mocks["get_ratings"].assert_not_called()
    assert main.SEASON_END_CHECK_FAILED_MARKER in log


def test_main_carries_on_in_season_when_it_cannot_tell(tmp_path, monkeypatch):
    mocks, calls, log, _ = _run(
        tmp_path, monkeypatch, now=datetime(2026, 10, 25, 7, 0, tzinfo=timezone.utc),
        resolved_week=8, state=RuntimeError("database unreachable"))
    mocks["get_ratings"].assert_called_once_with(2026, 8)
    assert main.SEASON_END_CHECK_FAILED_MARKER not in log


def test_main_with_an_explicit_week_never_asks(tmp_path, monkeypatch):
    mocks, calls, log, read_state = _run(
        tmp_path, monkeypatch, now=datetime(2026, 12, 20, 7, 0, tzinfo=timezone.utc),
        resolved_week=99, state=_state(range(1, 15)), argv=("main.py", "--year", "2026", "--week", "14"))
    read_state.assert_not_called()
    mocks["get_ratings"].assert_called_once_with(2026, 14)
