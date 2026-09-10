"""Tests for scripts/publish_season_schedule_artifact.py: its CLI argument handling, its
safety-critical guards (season range, non-empty payload, and the R2 round-trip verification),
and -- most importantly -- the safe-default contract: without `--publish`, nothing is ever
uploaded, and `--publish` alone is not treated as success unless the round trip confirms it.

Every DB/R2-touching collaborator (`_build_payload`, `publish_schedule_artifact`,
`_verify_published`, and `artifacts.r2.get_r2_client` where exercised directly) is mocked, so
these tests run with no database or network access, matching the rest of this suite. The
discriminating half of the safe-default contract is
test_real_run_calls_publish_schedule_artifact_and_verifies_round_trip below: without it, a
script that NEVER called publish_schedule_artifact (regardless of `--publish`) would also pass
the safe-default test, for the wrong reason.

Run: python -m pytest tests/ -q   (or: python tests/test_season_schedule_publish.py)
"""
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "publish_season_schedule_artifact.py"

sys.path.insert(0, str(REPO_ROOT))

from artifacts.schedule import build_schedule_payload  # noqa: E402


def _load_script_module():
    """scripts/ has no __init__.py (it's a bag of standalone one-off scripts, not a package),
    so the module is loaded by file path rather than a normal `import scripts.foo`."""
    spec = importlib.util.spec_from_file_location("publish_season_schedule_artifact", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publish_script = _load_script_module()


def _fake_payload(season=2025, conferences=None):
    """A minimal, realistically-shaped payload -- enough for the script's --out write and
    summary printing to exercise real code, without touching build_schedule_payload itself
    (that's covered by test_schedule_columns.py / test_non_fbs_opponent_logos.py)."""
    if conferences is None:
        conferences = [
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
        ]
    return {
        "season": season,
        "generated_at_utc": "2026-09-10T00:00:00+00:00",
        "conferences": conferences,
    }


def _run_main(argv, build_payload_return=None):
    """Invokes publish_script.main() with sys.argv patched and every DB/R2-touching
    collaborator mocked (_build_payload, publish_schedule_artifact, _verify_published).
    Returns (build_mock, publish_mock, verify_mock)."""
    if build_payload_return is None:
        build_payload_return = _fake_payload()
    with mock.patch.object(publish_script, "_build_payload", return_value=build_payload_return) as build_mock, \
         mock.patch.object(publish_script, "publish_schedule_artifact") as publish_mock, \
         mock.patch.object(publish_script, "_verify_published") as verify_mock, \
         mock.patch.object(sys, "argv", ["publish_season_schedule_artifact.py"] + argv):
        publish_script.main()
    return build_mock, publish_mock, verify_mock


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------
def test_missing_season_is_a_hard_error():
    """--season is required; argparse exits (code 2) rather than running with no season."""
    with mock.patch.object(sys, "argv", ["publish_season_schedule_artifact.py"]):
        with pytest.raises(SystemExit):
            publish_script.main()


def test_non_integer_season_is_a_hard_error():
    with mock.patch.object(sys, "argv", ["publish_season_schedule_artifact.py", "--season", "not-a-year"]):
        with pytest.raises(SystemExit):
            publish_script.main()


def test_season_is_parsed_as_int_not_str():
    build_mock, _, _ = _run_main(["--season", "2025"])
    build_mock.assert_called_once_with(2025)
    assert build_mock.call_args.args[0] != "2025"


def test_out_is_optional_and_default_safe_path_still_succeeds_without_it():
    """No --out and no --publish at all: the script must not require either to run."""
    build_mock, publish_mock, verify_mock = _run_main(["--season", "2025"])
    build_mock.assert_called_once_with(2025)
    publish_mock.assert_not_called()
    verify_mock.assert_not_called()


def test_out_writes_the_built_payload_as_json(tmp_path):
    out_path = tmp_path / "payload.json"
    payload = _fake_payload(season=2025)
    _run_main(["--season", "2025", "--out", str(out_path)], build_payload_return=payload)
    assert json.loads(out_path.read_text()) == payload


# ---------------------------------------------------------------------------
# Season range validation (BLOCKER 3): --season -1, 0, or 99999 all parse fine as plain ints
# (argparse's type=int has no range of its own) and, left unchecked, would build and
# "publish" an empty-but-successful-looking artifact. This must be rejected BEFORE any
# DB/R2 work -- so _build_payload is asserted never called, not just that main() exits.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("season", [-1, 0, 99999], ids=["negative", "zero", "far-future"])
def test_season_out_of_range_is_a_hard_error_before_any_db_work(season):
    with mock.patch.object(publish_script, "_build_payload") as build_mock, \
         mock.patch.object(sys, "argv", ["publish_season_schedule_artifact.py", "--season", str(season)]):
        with pytest.raises(SystemExit):
            publish_script.main()
    build_mock.assert_not_called()


def test_season_at_the_boundaries_is_accepted():
    """Discriminating counterpart to the out-of-range test above: the boundary values
    themselves (MIN_SEASON, and this-year-plus-one) must NOT be rejected -- otherwise a
    validator that rejected every season would also pass the out-of-range test, for the
    wrong reason."""
    import datetime
    max_season = datetime.datetime.now(datetime.timezone.utc).year + 1
    for season in (publish_script.MIN_SEASON, max_season):
        build_mock, publish_mock, verify_mock = _run_main(
            ["--season", str(season)], build_payload_return=_fake_payload(season=season)
        )
        build_mock.assert_called_once_with(season)


# ---------------------------------------------------------------------------
# Non-empty-payload validation (BLOCKER 3, other half): a built payload with zero
# conferences must refuse to proceed, rather than reporting success on an empty artifact.
# ---------------------------------------------------------------------------
def test_empty_payload_refuses_to_publish():
    empty = _fake_payload(conferences=[])
    with mock.patch.object(publish_script, "_build_payload", return_value=empty), \
         mock.patch.object(publish_script, "publish_schedule_artifact") as publish_mock, \
         mock.patch.object(publish_script, "_verify_published") as verify_mock, \
         mock.patch.object(sys, "argv", ["publish_season_schedule_artifact.py", "--season", "2025", "--publish"]):
        with pytest.raises(SystemExit):
            publish_script.main()
    publish_mock.assert_not_called()
    verify_mock.assert_not_called()


# ---------------------------------------------------------------------------
# The safe-default contract: without --publish, nothing is ever uploaded.
#
# "Upload" happens exclusively inside publish_schedule_artifact (it owns the R2 client, the
# bucket, and the put_object calls) -- the script itself never touches R2 directly except via
# _verify_published's read-back, which only runs on the --publish path. So asserting both
# names were never called, at the script's own module boundary, is the precise, non-loose
# version of "nothing was uploaded" rather than an indirect proxy for it.
# ---------------------------------------------------------------------------
def test_default_safe_path_never_calls_publish_schedule_artifact(tmp_path):
    out_path = tmp_path / "payload.json"
    build_mock, publish_mock, verify_mock = _run_main(
        ["--season", "2025", "--out", str(out_path)]
    )
    publish_mock.assert_not_called()
    verify_mock.assert_not_called()
    build_mock.assert_called_once_with(2025)
    # --out is still honored on the safe default path -- the payload is inspectable even
    # though nothing was uploaded.
    assert out_path.exists()


# ---------------------------------------------------------------------------
# Discriminating counterpart to the test above: with --publish, publish_schedule_artifact IS
# called, and its result is verified via a round trip. Without this test, a script that
# NEVER called publish_schedule_artifact (regardless of --publish) would still pass the
# safe-default test above, for the wrong reason -- it would just be broken in the opposite
# direction (BLOCKER 1: "success" that never actually happened).
# ---------------------------------------------------------------------------
def test_real_run_calls_publish_schedule_artifact_and_verifies_round_trip():
    build_mock, publish_mock, verify_mock = _run_main(["--season", "2025", "--publish"])
    publish_mock.assert_called_once_with(2025)
    verify_mock.assert_called_once_with(2025, build_mock.return_value)


# ---------------------------------------------------------------------------
# _verify_published: the round-trip check itself (BLOCKER 1 / BLOCKER 2). Exercised
# directly here (not through main()) so each failure mode -- mismatch, missing object, R2
# unconfigured -- is pinned independently of the CLI plumbing.
# ---------------------------------------------------------------------------
def _client_returning(payload_dict):
    client = mock.MagicMock()
    body = mock.MagicMock()
    body.read.return_value = json.dumps(payload_dict).encode("utf-8")
    client.get_object.return_value = {"Body": body}
    return client


def test_verify_published_passes_when_round_trip_matches_ignoring_generated_at_utc():
    built = _fake_payload()
    published = dict(built, generated_at_utc="a-completely-different-timestamp")
    with mock.patch("artifacts.r2.get_r2_client", return_value=_client_returning(published)), \
         mock.patch.dict(os.environ, {"R2_BUCKET_NAME": "test-bucket"}):
        publish_script._verify_published(2025, built)  # must not raise/exit


def test_verify_published_fails_on_content_mismatch():
    built = _fake_payload()
    published = _fake_payload(conferences=[])  # a real mismatch, not just generated_at_utc
    with mock.patch("artifacts.r2.get_r2_client", return_value=_client_returning(published)), \
         mock.patch.dict(os.environ, {"R2_BUCKET_NAME": "test-bucket"}):
        with pytest.raises(SystemExit):
            publish_script._verify_published(2025, built)


def test_verify_published_fails_when_the_object_is_missing():
    client = mock.MagicMock()
    client.get_object.side_effect = Exception("NoSuchKey")
    with mock.patch("artifacts.r2.get_r2_client", return_value=client), \
         mock.patch.dict(os.environ, {"R2_BUCKET_NAME": "test-bucket"}):
        with pytest.raises(SystemExit):
            publish_script._verify_published(2025, _fake_payload())


def test_verify_published_fails_when_r2_is_not_configured():
    # R2_BUCKET_NAME is deliberately SET here (unlike the two tests above) so this isolates
    # the "get_r2_client() returned None" guard specifically -- without setting it, the
    # subsequent "bucket not set" guard would also independently exit non-zero and this test
    # would pass even if the client-is-None guard were removed entirely, for the wrong reason.
    with mock.patch("artifacts.r2.get_r2_client", return_value=None), \
         mock.patch.dict(os.environ, {"R2_BUCKET_NAME": "test-bucket"}):
        with pytest.raises(SystemExit):
            publish_script._verify_published(2025, _fake_payload())


# ---------------------------------------------------------------------------
# _build_payload itself (not mocked): pins that it really is the K6 composition -- the same
# four _fetch_* helpers plus build_schedule_payload that publish_schedule_artifact uses
# internally -- rather than something that has silently diverged from it. Every other test
# in this file mocks _build_payload away, so without this one nothing exercises its body.
# ---------------------------------------------------------------------------
def test_build_payload_composes_fetch_helpers_and_build_schedule_payload():
    fake_rows = [
        dict(game_id="g1", season=2025, season_type="regular", team="Georgia Tech",
             opponent="Clemson", conference="ACC", conference_game=True, status="win",
             start_date="2025-09-05T16:00:00Z", home_away="home", neutral_site=False,
             team_score=24, opp_score=21, playoff_round_name=None, playoff_bowl_name=None,
             notes=None),
        dict(game_id="g1", season=2025, season_type="regular", team="Clemson",
             opponent="Georgia Tech", conference="ACC", conference_game=True, status="loss",
             start_date="2025-09-05T16:00:00Z", home_away="away", neutral_site=False,
             team_score=21, opp_score=24, playoff_round_name=None, playoff_bowl_name=None,
             notes=None),
    ]
    fake_teams_meta = {
        "Georgia Tech": {"conference": "ACC", "division": None, "logos": None},
        "Clemson": {"conference": "ACC", "division": None, "logos": None},
    }
    fake_team_ranks = {"Georgia Tech": 12, "Clemson": 5}
    fake_non_fbs_logos = {}

    with mock.patch.object(publish_script, "_engine", return_value=mock.MagicMock()), \
         mock.patch.object(publish_script, "_fetch_schedule_grid_rows", return_value=fake_rows), \
         mock.patch.object(publish_script, "_fetch_teams_meta", return_value=fake_teams_meta), \
         mock.patch.object(publish_script, "_fetch_team_ranks", return_value=fake_team_ranks), \
         mock.patch.object(publish_script, "_fetch_non_fbs_logos", return_value=fake_non_fbs_logos):
        result = publish_script._build_payload(2025)

    expected = build_schedule_payload(fake_rows, fake_teams_meta, 2025, fake_team_ranks, fake_non_fbs_logos)
    # generated_at_utc differs between the two calls by wall-clock construction; compare the
    # rest, which is everything the composition is actually responsible for getting right.
    result.pop("generated_at_utc")
    expected.pop("generated_at_utc")
    assert result == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
