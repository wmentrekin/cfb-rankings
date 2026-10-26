"""A transient CFBD failure must not turn into a silently stale, mislabelled publish.

BACKGROUND: on 2026-10-25 CFBD's gateway returned a 504 (after 120s) for the week 8 /games
request. The bare requests.get had no timeout or retry, main.py's generic handler logged "they
may already be loaded" and carried on, and the run rated the previous week's games, labelled the
result week 8, and published it -- every step green. See database/cfbd_http.py and
main._log_ingest_fetch_failure.

Three layers are pinned here, each against the real thing rather than a mock of it:
  1. cfbd_get against a local HTTP server that actually returns 504s / hangs / returns 401.
  2. main() with the ingest patched to fail: it must stop before the model and publish.
  3. Static guards: no bare requests.get in database/, and the INGEST_FAILURE token main.py
     emits is the literal string both workflows grep for.

MUTATION-CHECK LOG: (a) restoring `requests.get` in get_games.py failed the static guard;
(b) restoring the old `logger.warning(...)`/continue in main's regular-season handler failed
test_regular_season_ingest_failure_stops_before_model_and_publish; (c) removing 504 from
RETRY_STATUSES failed test_retries_a_504_then_succeeds; (d) removing the timeout
argument failed test_a_hung_server_is_abandoned_not_awaited. Each was reverted afterwards.
"""
import ast
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest  # type: ignore
import requests  # type: ignore

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import main  # noqa: E402
from database import cfbd_http  # noqa: E402


# ---------------------------------------------------------------------------
# 1. cfbd_get against a real local server
# ---------------------------------------------------------------------------
class _Server:
    """Serves a scripted sequence of (status, body, delay) responses, then repeats the last."""

    def __init__(self, script):
        self.script = list(script)
        self.hits = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 -- http.server API
                index = min(outer.hits, len(outer.script) - 1)
                outer.hits += 1
                status, body, delay = outer.script[index]
                if delay:
                    time.sleep(delay)
                payload = body.encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client gave up on a delayed response, which is the point

            def log_message(self, *args):  # silence
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/games"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    monkeypatch.setattr(cfbd_http, "RETRY_BACKOFF", 0)


def test_retries_a_504_then_succeeds():
    """The 2026-10-25 failure shape: gateway errors first, then a healthy response."""
    with _Server([(504, "{}", 0), (504, "{}", 0), (200, '[{"id": 1}]', 0)]) as server:
        response = cfbd_http.cfbd_get(server.url, headers={"Authorization": "Bearer x"})
    assert response.json() == [{"id": 1}]
    assert server.hits == 3


def test_exhausted_retries_raise_the_same_httperror_callers_always_got():
    with _Server([(504, "{}", 0)]) as server:
        with pytest.raises(requests.HTTPError):
            cfbd_http.cfbd_get(server.url)
    assert server.hits == cfbd_http.RETRY_TOTAL + 1


def test_a_deterministic_4xx_is_not_retried():
    with _Server([(401, "{}", 0)]) as server:
        with pytest.raises(requests.HTTPError):
            cfbd_http.cfbd_get(server.url)
    assert server.hits == 1, "a bad key is not transient; retrying it only delays the failure"


def test_a_hung_server_is_abandoned_not_awaited(monkeypatch):
    """The original bug waited the gateway's full 120s because nothing bounded the read."""
    monkeypatch.setattr(cfbd_http, "READ_TIMEOUT", 0.2)
    monkeypatch.setattr(cfbd_http, "RETRY_TOTAL", 1)
    with _Server([(200, "[]", 2.0)]) as server:
        started = time.monotonic()
        with pytest.raises(requests.RequestException):
            cfbd_http.cfbd_get(server.url)
        # Measured inside the block: leaving it waits for the slow handler, which says nothing
        # about how long the CLIENT waited.
        elapsed = time.monotonic() - started
    assert elapsed < 1.5, f"the client waited {elapsed:.1f}s on a server that took 2s to answer"


# ---------------------------------------------------------------------------
# 2. main() when the ingest fails
# ---------------------------------------------------------------------------
def _run_main(tmp_path, monkeypatch, load_games, *, week=8):
    """Run main() with every collaborator stubbed except the ingest handling under test.

    Returns the mocks that must (or must not) have been reached, and the run's log text.
    """
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["main.py", "--year", "2026", "--week", str(week)])
    mocks = {
        "get_ratings": mock.Mock(return_value=None),  # None -> main logs and exits cleanly
        "publish_rankings_artifact": mock.Mock(),
        "publish_schedule_artifact": mock.Mock(),
    }
    with mock.patch.object(main, "teams_exist_for_year", return_value=True), \
            mock.patch.object(main, "non_fbs_logo_coverage_is_complete", return_value=True), \
            mock.patch.object(main, "load_games_to_db", side_effect=load_games), \
            mock.patch.object(main, "apply_game_result_overrides",
                              return_value={"applied": [], "failures": []}), \
            mock.patch.object(main, "get_ratings", mocks["get_ratings"]), \
            mock.patch.object(main, "publish_rankings_artifact", mocks["publish_rankings_artifact"]), \
            mock.patch.object(main, "publish_schedule_artifact", mocks["publish_schedule_artifact"]):
        main.main()
    log = (tmp_path / f"run_2026_w{week}.log").read_text()
    return mocks, log


def _http_504():
    return requests.HTTPError("504 Server Error: Gateway Timeout for url: https://api.collegefootballdata.com/games")


def test_regular_season_ingest_failure_stops_before_model_and_publish(tmp_path, monkeypatch):
    """THE regression: week 8's /games call failed; the run must not rate and publish week 7's data."""
    def load_games(year, week=None, season_type="regular"):
        raise _http_504()

    mocks, log = _run_main(tmp_path, monkeypatch, load_games)

    mocks["get_ratings"].assert_not_called()
    mocks["publish_rankings_artifact"].assert_not_called()
    mocks["publish_schedule_artifact"].assert_not_called()
    assert "INGEST_FAILURE" in log
    assert "Regular-season" in log
    assert "Run finished successfully" not in log


def test_postseason_ingest_failure_is_flagged_but_does_not_block_the_week(tmp_path, monkeypatch):
    """The postseason slate does not gate the regular-season week, so it alarms without stopping."""
    def load_games(year, week=None, season_type="regular"):
        if season_type == "postseason":
            raise _http_504()

    mocks, log = _run_main(tmp_path, monkeypatch, load_games)

    mocks["get_ratings"].assert_called_once()
    assert "INGEST_FAILURE" in log
    assert "Postseason" in log


def test_healthy_ingest_emits_no_failure_token_and_reaches_the_model(tmp_path, monkeypatch):
    mocks, log = _run_main(tmp_path, monkeypatch, lambda *a, **k: None)

    mocks["get_ratings"].assert_called_once()
    assert "INGEST_FAILURE" not in log


# ---------------------------------------------------------------------------
# 3. Static guards
# ---------------------------------------------------------------------------
def test_no_database_module_calls_requests_get_directly():
    """A bare requests.get has no timeout and no retry -- the original bug. Use cfbd_get."""
    offenders = []
    for path in sorted((REPO / "database").glob("*.py")):
        if path.name == "cfbd_http.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"get", "post", "request"}
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "requests"):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"bare requests calls (no timeout/retry): {offenders}"


def test_ingest_failure_token_matches_between_main_and_workflows():
    token = main.INGEST_FAILURE_MARKER
    assert token == "INGEST_FAILURE"
    for name in ("weekly-update.yml", "test-week-run.yml"):
        src = (REPO / ".github" / "workflows" / name).read_text()
        assert re.search(rf'grep -q "{token}" output\.log', src), (
            f"{name} must grep output.log for the literal {token!r} main.py emits; without it a "
            "failed ingest goes green behind `|| true`"
        )
