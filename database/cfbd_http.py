"""Shared HTTP access to the CollegeFootballData API: a bounded wait and bounded retries.

WHY THIS EXISTS: all three CFBD fetchers used a bare `requests.get(url, headers=...)`, which
has NO timeout and NO retry. On 2026-10-25 CFBD's gateway returned a 504 after 120 seconds for
the week 8 /games request. With nothing to retry it, the single failure propagated to main.py's
ingest handler, which logged a warning and carried on -- so the run rated the previous week's
games, labelled the result week 8, and published it. Every step reported success.

A gateway timeout from a public API is transient far more often than not, so the fix at this
layer is the boring one: don't wait forever, and try a few more times before giving up. What
happens when the retries are exhausted is main.py's decision, not this module's -- an exhausted
call raises exactly what a bare requests.get(...).raise_for_status() always raised, so callers
need no new handling.

Only idempotent GETs are retried, and only for failures that say "try again": connection
errors, read timeouts, and 429/500/502/503/504. A 401 (bad key) or 404 is deterministic and
fails on the first attempt, as before.
"""
import requests  # type: ignore
from requests.adapters import HTTPAdapter  # type: ignore
from urllib3.util.retry import Retry  # type: ignore

# (connect, read) seconds. A healthy week of games returns in a few seconds; the failure this
# guards against sat for the gateway's full 120s. 60s read leaves ample headroom for a slow but
# healthy response while ensuring a stuck one is abandoned and retried rather than awaited.
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60

# Retries after the first attempt, i.e. up to 5 attempts in total. Backoff sleeps
# factor * 2**(n-1) seconds before retry n (urllib3 skips the first), so the sleeps total about
# 28s with the default factor of 2; the worst case for one call is therefore a few minutes,
# well inside the workflow's 60-minute job timeout. Honours a Retry-After header when CFBD
# sends one with a 429/503.
RETRY_TOTAL = 4
RETRY_BACKOFF = 2.0
RETRY_STATUSES = (429, 500, 502, 503, 504)


def _build_session() -> requests.Session:
    # Constants are read here, at call time, so a test (or an operator) can override them.
    retry = Retry(
        total=RETRY_TOTAL,
        connect=RETRY_TOTAL,
        read=RETRY_TOTAL,
        status=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=RETRY_STATUSES,
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        # Hand back the final response instead of raising MaxRetryError, so the caller's
        # raise_for_status() produces the same requests.HTTPError it always did.
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def cfbd_get(url, headers=None, params=None) -> requests.Response:
    """GET a CFBD endpoint with a timeout and retries; raise on a final HTTP error status.

    Returns the successful Response. Raises requests.HTTPError for an error status that
    survived the retries (or was never retryable), and requests.ConnectionError /
    requests.Timeout if the network failure outlasted them -- all subclasses of
    requests.RequestException, which is what main.py's ingest handlers already catch.
    """
    with _build_session() as session:
        response = session.get(url, headers=headers, params=params, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    response.raise_for_status()
    return response
