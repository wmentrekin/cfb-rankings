"""Tests for scripts/backfill_games.py: its safety-critical guards (season/week bounds, the
empty-CFBD-response case, and the post-write read-back) and the safe-default contract --
without `--apply`, load_games_to_db is NEVER called.

The discriminating test here is test_dry_run_still_fetches_and_reports: without it, a script
that fetched nothing at all would also satisfy "dry run writes nothing", for the wrong reason.
A dry run whose whole job is to show you what WOULD be written is worthless if it silently
shows you nothing.

Every CFBD/DB collaborator (`get_games_by_year_week`, `load_games_to_db`, `_engine`,
`_existing_ids`) is mocked, so these run with no network or database, matching the rest of
this suite.

Run: python -m pytest tests/ -q
"""
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pandas as pd  # type: ignore
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "backfill_games.py"

sys.path.insert(0, str(REPO_ROOT))


def _load_script():
    spec = importlib.util.spec_from_file_location("backfill_games", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backfill = _load_script()


def _games_df(ids):
    """Minimal frame carrying only the columns main() actually reads when it prints."""
    return pd.DataFrame([
        {
            "id": gid, "week": 16, "start_date": "2025-12-13 15:00:00",
            "away_team": "Navy", "home_team": "Army", "away_score": 17, "home_score": 10,
            "conference_game": True, "notes": None,
        }
        for gid in ids
    ])


def _run(argv, fetch_return, existing=frozenset(), landed=None):
    """Drive main() with argv, mocking every network/DB collaborator. `landed` defaults to
    `fetch_return`'s ids, i.e. the healthy case where every upserted row reads back."""
    ids = [int(i) for i in fetch_return["id"].tolist()] if not fetch_return.empty else []
    landed_set = set(ids) if landed is None else set(landed)
    calls = {}
    existing_seq = [set(existing), landed_set]

    def fake_existing_ids(engine, game_ids):
        return existing_seq.pop(0) if existing_seq else set()

    with mock.patch.object(sys, "argv", ["backfill_games.py"] + argv), \
            mock.patch.object(backfill, "get_games_by_year_week", return_value=fetch_return) as fetch, \
            mock.patch.object(backfill, "load_games_to_db") as load, \
            mock.patch.object(backfill, "_engine", return_value=mock.Mock()), \
            mock.patch.object(backfill, "_existing_ids", side_effect=fake_existing_ids):
        code = backfill.main()
        calls["fetch"] = fetch
        calls["load"] = load
    return code, calls


# --- Bounds guards: a typo must fail loudly, before any network call. -----------------
@pytest.mark.parametrize("season", [2013, 2101], ids=["below", "above"])
def test_out_of_range_season_exits_nonzero_without_fetching(season):
    with mock.patch.object(sys, "argv", ["backfill_games.py", "--season", str(season)]), \
            mock.patch.object(backfill, "get_games_by_year_week") as fetch:
        assert backfill.main() == 1
    fetch.assert_not_called()


@pytest.mark.parametrize("week", [-1, 21], ids=["negative", "too-high"])
def test_out_of_range_week_exits_nonzero_without_fetching(week):
    argv = ["backfill_games.py", "--season", "2025", "--week", str(week)]
    with mock.patch.object(sys, "argv", argv), \
            mock.patch.object(backfill, "get_games_by_year_week") as fetch:
        assert backfill.main() == 1
    fetch.assert_not_called()


# --- The failure mode most easily mistaken for success. -------------------------------
def test_empty_cfbd_response_exits_nonzero_and_writes_nothing():
    """CFBD legitimately returns [] for a week with no games, so this is not an exception --
    but it IS indistinguishable from a successful backfill if the script exits 0 and says
    little. It must fail the job instead."""
    code, calls = _run(["--season", "2025", "--week", "16"], pd.DataFrame())
    assert code == 1
    calls["load"].assert_not_called()


# --- Safe-default contract. ------------------------------------------------------------
def test_dry_run_never_calls_load_games_to_db():
    code, calls = _run(["--season", "2025", "--week", "16"], _games_df([401752900]))
    assert code == 0
    calls["load"].assert_not_called()


def test_dry_run_still_fetches_and_reports(capsys):
    """The discriminating half: a dry run that fetched nothing would trivially satisfy
    "writes nothing" while being useless. It must still query CFBD and classify each row."""
    code, calls = _run(["--season", "2025", "--week", "16"], _games_df([401752900]))
    assert code == 0
    calls["fetch"].assert_called_once_with(2025, 16, "regular")
    out = capsys.readouterr().out
    assert "NEW" in out and "401752900" in out
    assert "DRY RUN" in out


def test_apply_calls_load_games_to_db_with_the_same_slice():
    code, calls = _run(["--season", "2025", "--week", "16", "--apply"], _games_df([401752900]))
    assert code == 0
    calls["load"].assert_called_once_with(2025, 16, "regular")


def test_apply_emits_the_marker_the_workflow_greps_for(capsys):
    """The workflow fails the job unless BACKFILL_APPLIED appears in the log. If this marker
    is ever renamed here without updating .github/workflows/backfill-games.yml, every real
    apply starts failing -- so the string is pinned on both sides."""
    code, _ = _run(["--season", "2025", "--week", "16", "--apply"], _games_df([401752900]))
    assert code == 0
    assert "BACKFILL_APPLIED" in capsys.readouterr().out


def test_apply_fails_when_a_row_does_not_read_back(capsys):
    """load_games_to_db returns None and raises only on a DB/HTTP error, so a clean return
    does not prove the rows landed. A row missing from the read-back must fail the job."""
    code, _ = _run(
        ["--season", "2025", "--week", "16", "--apply"],
        _games_df([401752900, 401752901]),
        landed=[401752900],          # the second row never landed
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "401752901" in out
    assert "BACKFILL_APPLIED" not in out, "the success marker must not be emitted on a failed read-back"


def test_existing_row_is_reported_as_existing_not_new(capsys):
    code, _ = _run(["--season", "2025", "--week", "16"], _games_df([401752900]),
                   existing={401752900})
    assert code == 0
    out = capsys.readouterr().out
    assert "EXISTING" in out
    assert "0 new, 1 already present." in out


def test_existing_ids_short_circuits_on_empty_input():
    """`id = ANY(:ids)` with an empty list is fine, but the short-circuit means an empty
    fetch never opens a connection at all -- assert it does not touch the engine."""
    engine = mock.Mock()
    assert backfill._existing_ids(engine, []) == set()
    engine.connect.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
