"""Tests for scripts/publish_season_schedule_artifact.py: its CLI argument handling, and --
most importantly -- the dry-run contract: `--dry-run` never calls publish_schedule_artifact
(the function that performs the R2 upload).

Every DB/R2-touching collaborator (`_build_payload`, `publish_schedule_artifact`) is mocked,
so these tests run with no database or network access, matching the rest of this suite. The
discriminating half of the dry-run contract is test_real_run_calls_publish_schedule_artifact
below: without it, a script that NEVER called publish_schedule_artifact (dry-run or not) would
also pass the dry-run test, for the wrong reason.

Run: python -m pytest tests/ -q   (or: python tests/test_season_schedule_publish.py)
"""
import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "publish_season_schedule_artifact.py"


def _load_script_module():
    """scripts/ has no __init__.py (it's a bag of standalone one-off scripts, not a package),
    so the module is loaded by file path rather than a normal `import scripts.foo`."""
    spec = importlib.util.spec_from_file_location("publish_season_schedule_artifact", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publish_script = _load_script_module()


def _fake_payload(season=2025):
    """A minimal, realistically-shaped payload -- enough for the script's --out write and
    summary printing to exercise real code, without touching build_schedule_payload itself
    (that's covered by test_schedule_columns.py / test_non_fbs_opponent_logos.py)."""
    return {
        "season": season,
        "generated_at_utc": "2026-09-10T00:00:00+00:00",
        "conferences": [
            {
                "name": "ACC",
                "teams": [
                    {
                        "team": "Georgia Tech",
                        "logo_url": None,
                        "rank": 12,
                        "record": {"wins": 9, "losses": 4},
                        "conf_record": {"wins": 6, "losses": 2},
                        "division": None,
                        "weeks": [
                            {
                                "slot_id": "week-1", "label": "Week 1", "season_type": "regular",
                                "opponent": "Clemson", "opponent_logo_url": None,
                                "conditional_opponent": None, "game_name": None,
                                "home_away": "home", "neutral_site": False,
                                "status": "win", "team_score": 24, "opp_score": 21,
                            },
                        ],
                    },
                ],
            },
        ],
    }


def _run_main(argv, build_payload_return=None):
    """Invokes publish_script.main() with sys.argv patched and both DB/R2-touching
    collaborators mocked. Returns (build_mock, publish_mock)."""
    if build_payload_return is None:
        build_payload_return = _fake_payload()
    with mock.patch.object(publish_script, "_build_payload", return_value=build_payload_return) as build_mock, \
         mock.patch.object(publish_script, "publish_schedule_artifact") as publish_mock, \
         mock.patch.object(sys, "argv", ["publish_season_schedule_artifact.py"] + argv):
        publish_script.main()
    return build_mock, publish_mock


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------
def test_missing_season_is_a_hard_error():
    """--season is required; argparse exits (code 2) rather than running with no season."""
    with mock.patch.object(sys, "argv", ["publish_season_schedule_artifact.py", "--dry-run"]):
        with pytest.raises(SystemExit):
            publish_script.main()


def test_non_integer_season_is_a_hard_error():
    with mock.patch.object(sys, "argv", ["publish_season_schedule_artifact.py", "--season", "not-a-year"]):
        with pytest.raises(SystemExit):
            publish_script.main()


def test_season_is_parsed_as_int_not_str():
    build_mock, _ = _run_main(["--season", "2025", "--dry-run"])
    build_mock.assert_called_once_with(2025)
    assert build_mock.call_args.args[0] != "2025"


def test_out_is_optional_and_dry_run_still_succeeds_without_it():
    """No --out at all: the script must not require a path to run a dry run."""
    build_mock, publish_mock = _run_main(["--season", "2025", "--dry-run"])
    build_mock.assert_called_once_with(2025)
    publish_mock.assert_not_called()


def test_out_writes_the_built_payload_as_json(tmp_path):
    out_path = tmp_path / "payload.json"
    payload = _fake_payload(season=2025)
    _run_main(["--season", "2025", "--dry-run", "--out", str(out_path)], build_payload_return=payload)
    assert json.loads(out_path.read_text()) == payload


# ---------------------------------------------------------------------------
# The dry-run contract: no upload happens.
#
# "Upload" happens exclusively inside publish_schedule_artifact (it owns the R2 client, the
# bucket, and the put_object calls) -- the script itself never touches R2 directly. So
# asserting that name was never called, at the script's own module boundary, is the precise,
# non-loose version of "nothing was uploaded" rather than an indirect proxy for it.
# ---------------------------------------------------------------------------
def test_dry_run_never_calls_publish_schedule_artifact(tmp_path):
    out_path = tmp_path / "payload.json"
    build_mock, publish_mock = _run_main(
        ["--season", "2025", "--dry-run", "--out", str(out_path)]
    )
    publish_mock.assert_not_called()
    build_mock.assert_called_once_with(2025)
    # --out is still honored on a dry run -- the payload is inspectable even though nothing
    # was uploaded.
    assert out_path.exists()


# ---------------------------------------------------------------------------
# Discriminating counterpart to the test above: without --dry-run, publish_schedule_artifact
# IS called, with the parsed season. Without this test, a script that dropped the call to
# publish_schedule_artifact entirely (dry-run or not) would still pass the dry-run test above,
# for the wrong reason -- it would just be broken in the opposite direction.
# ---------------------------------------------------------------------------
def test_real_run_calls_publish_schedule_artifact():
    build_mock, publish_mock = _run_main(["--season", "2025"])
    publish_mock.assert_called_once_with(2025)
    build_mock.assert_called_once_with(2025)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
