"""Publisher, index, script and workflow for the playoffs artifact -- no DB, no R2, no network.

Fake R2 client in the style of tests/test_season_schedule_publish.py; every DB-touching
collaborator is mocked. Also pins the success marker shared by the publisher's log line and the
workflow's grep, as tests/test_ingest_resilience.py does for INGEST_FAILURE.
"""
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from artifacts import playoffs  # noqa: E402

SCRIPT_PATH = REPO / "scripts" / "publish_playoffs_artifact.py"
WORKFLOW_PATH = REPO / ".github" / "workflows" / "publish-playoffs-artifact.yml"


def _load_script():
    spec = importlib.util.spec_from_file_location("publish_playoffs_artifact", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


def _payload(season=2026):
    return {
        "season": season, "kind": "projected", "as_of_week": 12, "generated_at_utc": "2026-11-23T00:00:00+00:00",
        "teams": [{"seed": 1, "team": "X"}], "rounds": [], "games": [{"slot": f"G{i}"} for i in range(11)], "first_out": [],
    }


class FakeClient:
    def __init__(self, keys=()):
        self.keys = list(keys)
        self.puts = {}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.puts[Key] = json.loads(Body)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        client = self

        class _Paginator:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in client.keys + list(client.puts) if k.startswith(Prefix)]}

        return _Paginator()


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
def test_index_is_rebuilt_from_listing():
    client = FakeClient(keys=["playoffs/2025/latest.json", "playoffs/2026/latest.json", "playoffs/index.json",
                              "schedule/2027/latest.json", "playoffs/2026/other.json"])
    index = playoffs.build_playoffs_index(client, "b")
    assert index["seasons"] == [2026, 2025] and index["latest_season"] == 2026
    assert set(index) == {"generated_at_utc", "latest_season", "seasons"}


def test_index_none_when_nothing_published():
    assert playoffs.build_playoffs_index(FakeClient(keys=["schedule/2026/latest.json"]), "b") is None


def test_index_none_on_listing_failure():
    broken = mock.Mock()
    broken.get_paginator.side_effect = RuntimeError("boom")
    assert playoffs.build_playoffs_index(broken, "b") is None


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------
def _publish(year, payload, client, caplog, bucket="bucket"):
    with mock.patch.object(playoffs, "load_playoffs_payload", return_value=payload), \
         mock.patch.object(playoffs, "get_r2_client", return_value=client), \
         mock.patch.object(playoffs, "load_dotenv"), \
         mock.patch.dict("os.environ", {"R2_BUCKET_NAME": bucket} if bucket else {}, clear=not bucket), \
         caplog.at_level(logging.INFO, logger="cfb_lp"):
        playoffs.publish_playoffs_artifact(year)


def test_publish_uploads_payload_and_index(caplog):
    client = FakeClient()
    _publish(2026, _payload(), client, caplog)
    assert client.puts["playoffs/2026/latest.json"] == _payload()
    assert client.puts["playoffs/index.json"]["seasons"] == [2026]
    assert "Published playoffs artifact to R2 key playoffs/2026/latest.json" in caplog.text


def test_unsupported_season_skipped_without_touching_anything(caplog):
    client = FakeClient()
    with mock.patch.object(playoffs, "load_playoffs_payload") as load:
        with mock.patch.object(playoffs, "get_r2_client", return_value=client):
            playoffs.publish_playoffs_artifact(2027)
    load.assert_not_called()
    assert client.puts == {}


def test_none_payload_publishes_nothing(caplog):
    client = FakeClient()
    _publish(2026, None, client, caplog)
    assert client.puts == {}


def test_missing_bucket_publishes_nothing(caplog):
    client = FakeClient()
    _publish(2026, _payload(), client, caplog, bucket=None)
    assert client.puts == {}


def test_publisher_never_raises():
    with mock.patch.object(playoffs, "load_playoffs_payload", side_effect=RuntimeError("db down")):
        playoffs.publish_playoffs_artifact(2026)  # must not raise


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------
def _run_script(argv, payload=None):
    payload = _payload() if payload is None else payload
    with mock.patch.object(script, "load_playoffs_payload", return_value=payload) as load, \
         mock.patch.object(script, "publish_playoffs_artifact") as publish, \
         mock.patch.object(script, "_verify_published") as verify, \
         mock.patch.object(sys, "argv", ["publish_playoffs_artifact.py"] + argv):
        script.main()
    return load, publish, verify


def test_default_is_read_only():
    load, publish, verify = _run_script(["--season", "2026"])
    load.assert_called_once_with(2026)
    publish.assert_not_called()
    verify.assert_not_called()


def test_publish_flag_publishes_and_verifies():
    _, publish, verify = _run_script(["--season", "2026", "--publish"])
    publish.assert_called_once_with(2026)
    verify.assert_called_once()


def test_out_writes_payload(tmp_path):
    out = tmp_path / "p.json"
    _run_script(["--season", "2026", "--out", str(out)])
    assert json.loads(out.read_text()) == _payload()


@pytest.mark.parametrize("season", ["2024", "2027", "0"])
def test_unsupported_season_exits_before_any_work(season):
    with mock.patch.object(script, "load_playoffs_payload") as load, \
         mock.patch.object(sys, "argv", ["publish_playoffs_artifact.py", "--season", season]):
        with pytest.raises(SystemExit) as exc:
            script.main()
    assert exc.value.code == 1
    load.assert_not_called()


def test_empty_payload_exits_nonzero_and_never_publishes():
    with mock.patch.object(script, "load_playoffs_payload", return_value=None), \
         mock.patch.object(script, "publish_playoffs_artifact") as publish, \
         mock.patch.object(sys, "argv", ["publish_playoffs_artifact.py", "--season", "2026", "--publish"]):
        with pytest.raises(SystemExit) as exc:
            script.main()
    assert exc.value.code == 1
    publish.assert_not_called()


def test_verify_detects_mismatch_and_ignores_generated_at(monkeypatch):
    published = dict(_payload(), generated_at_utc="other")

    class _Client:
        def __init__(self, body):
            self.body = body

        def get_object(self, Bucket, Key):
            assert Key == "playoffs/2026/latest.json"
            return {"Body": mock.Mock(read=lambda: json.dumps(self.body).encode())}

    monkeypatch.setenv("R2_BUCKET_NAME", "b")
    with mock.patch("artifacts.r2.get_r2_client", return_value=_Client(published)):
        script._verify_published(2026, _payload())  # equal ignoring generated_at_utc
    with mock.patch("artifacts.r2.get_r2_client", return_value=_Client(dict(published, kind="actual"))):
        with pytest.raises(SystemExit):
            script._verify_published(2026, _payload())


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
def test_workflow_parses_and_is_manual_dry_run_by_default():
    # PyYAML is not a project dependency (and none may be added): the structural parse runs where
    # it happens to be installed, the text checks below always run.
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load(WORKFLOW_PATH.read_text())
    triggers = wf.get("on", wf.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    assert triggers["workflow_dispatch"]["inputs"]["dry_run"]["default"] is True
    assert wf["concurrency"]["group"] == "weekly-rankings" and wf["concurrency"]["cancel-in-progress"] is False
    steps = wf["jobs"]["publish_playoffs_artifact"]["steps"]
    uploads = [s["with"]["name"] for s in steps if "upload-artifact" in s.get("uses", "")]
    assert uploads == ["playoffs-payload-season-${{ github.event.inputs.season }}",
                       "playoffs-publish-log-season-${{ github.event.inputs.season }}"]


def test_workflow_text_shape():
    src = WORKFLOW_PATH.read_text()
    assert "workflow_dispatch:" in src and "group: weekly-rankings" in src and "cancel-in-progress: false" in src
    assert re.search(r"dry_run:\n(?:.*\n)*?\s+default: true", src)
    assert src.count("actions/upload-artifact@v4") == 2


def test_workflow_fail_safe_polarity_and_script_name():
    src = WORKFLOW_PATH.read_text()
    assert 'if [ "$DRY_RUN" = "false" ]; then' in src and 'if [ "$DRY_RUN" = "true" ]' not in src
    assert "scripts/publish_playoffs_artifact.py" in src


def test_workflow_marker_matches_publisher_log_line():
    marker = playoffs.PUBLISHED_LOG_MESSAGE % "playoffs/${SEASON}/latest.json"
    assert marker == "Published playoffs artifact to R2 key playoffs/${SEASON}/latest.json"
    assert f'MARKER="{marker}"' in WORKFLOW_PATH.read_text()


def test_main_hook_follows_schedule_publish_and_respects_staging():
    src = (REPO / "main.py").read_text()
    schedule_at = src.index("publish_schedule_artifact(args.year)")
    playoffs_at = src.index("publish_playoffs_artifact(args.year)")
    assert schedule_at < playoffs_at
    block = src[src.index("# PUBLISH PLAYOFFS ARTIFACT"):]
    assert "if not args.staging:" in block and "try:" in block and "except Exception" in block
