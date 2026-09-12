"""
One-off: publish one season's Season Grid artifact, and do nothing else.

BACKGROUND
----------
There is currently no narrow way to republish a past season's schedule artifact. The two
existing entry points both do far more than that:

  - `main.py --year N` re-ingests that season from CFBD, runs the rating model, and
    overwrites the season's `ratings` table rows and its rankings artifact along the way --
    and it resolves the pipeline week from TODAY's date, which is meaningless for a finished
    season (see main.py's resolve_week_from_starts).
  - `artifacts.schedule.publish_schedule_artifact(season)` on its own is the correct narrow
    primitive -- it only reads schedule_grid/teams/ratings/non_fbs_teams and writes R2
    (schedule/{season}/latest.json + schedule/index.json); no ingest, no model, no ratings
    write. But it is a plain function, not reachable from the command line, and -- critically
    -- it NEVER RAISES: every one of its six bail-outs (no teams, no rows, R2 unconfigured, no
    bucket configured, upload_json returning False, or an unexpected exception) is logged and
    swallowed, so a caller that only checks the process exit code cannot tell a real, matching
    publish from a silently skipped or failed one.

This script is that missing entry point: `--season N` builds one season's Season Grid
payload and, only if `--publish` is passed, publishes it and then PROVES the publish landed
by reading the object back from R2 and diffing it against what this run built (ignoring
generated_at_utc) -- see _verify_published(). Without `--publish` (the default, safe path --
opt IN to publishing, matching scripts/repair_unplayed_game_winners.py's `--apply`
precedent), nothing is ever uploaded. `--season` is bounds-checked and the built payload must
have at least one conference before either path proceeds, so a typo'd or not-yet-ingested
season fails loudly instead of quietly reporting success on an empty artifact.

BLAST RADIUS
------------
Default (no `--publish`): read-only against the database (see CREDENTIALS REQUIRED below).
Writes nothing anywhere except the optional `--out` file on local disk.

`--publish`: overwrites the R2 keys `schedule/{season}/latest.json` and `schedule/index.json`
in place -- an atomic per-object PUT, not a delete-then-write. No other R2 key and no database
table is touched; every DB call in this path (both here and inside publish_schedule_artifact)
is a SELECT. The one additional call this path makes beyond publish_schedule_artifact itself
is a single read-only `get_object` on the key it just wrote, to verify the round trip.

CREDENTIALS REQUIRED
---------------------
  default (no --publish):  DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME only
                            (read-only queries against schedule_grid, teams, ratings,
                            non_fbs_teams).
  --publish:                the same DB_* set, plus R2_ACCOUNT_ID / R2_ACCESS_KEY_ID /
                            R2_SECRET_ACCESS_KEY / R2_BUCKET_NAME -- both
                            publish_schedule_artifact's own requirement (see
                            artifacts/r2.py's get_r2_client()) and this script's own
                            round-trip read-back.

Usage:
    python scripts/publish_season_schedule_artifact.py --season 2025 --out payload.json
    python scripts/publish_season_schedule_artifact.py --season 2025 --out payload.json --publish
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Running this file directly only puts scripts/ on sys.path, not the repo root, so
# file directly only puts scripts/ on sys.path, not the repo root, so `artifacts` isn't
# importable without it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # type: ignore  # noqa: E402
from sqlalchemy import create_engine  # type: ignore  # noqa: E402

from artifacts.schedule import (  # noqa: E402
    CFP_SLOTS,
    build_schedule_payload,
    publish_schedule_artifact,
    _fetch_non_fbs_logos,
    _fetch_schedule_grid_rows,
    _fetch_team_ranks,
    _fetch_teams_meta,
)

# The four _fetch_* imports above are underscore-prefixed internals of artifacts/schedule.py,
# not a stable public API -- importing them here is a deliberate, narrow coupling, not an
# oversight. plan.yaml's K6 settled that no seam belongs in schedule.py for this:
# build_schedule_payload is already public and already separate from publish_schedule_artifact,
# so composing these four helpers plus build_schedule_payload is how this script reproduces,
# for the default path and for --out, exactly the payload publish_schedule_artifact builds
# internally -- without duplicating or diverging from its query logic. Do not edit schedule.py
# to avoid this coupling; escalate instead (see the handoff).

# CFBD's practical minimum -- this pipeline has never ingested anything earlier -- through
# next calendar year (so next season's early game additions can be republished ahead of its
# own kickoff). Just enough to catch a typo'd or garbage --season (-1, 0, 99999) BEFORE any
# DB/R2 work: those all parse fine as plain ints (argparse's type=int has no range of its
# own) and, left unchecked, build and "publish" an empty-but-successful-looking artifact.
MIN_SEASON = 2000


def _engine():
    load_dotenv()
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )


def _validate_season(season: int) -> None:
    """Exits non-zero for a season outside the sane range, before any DB/R2 work runs."""
    max_season = datetime.now(timezone.utc).year + 1
    if not (MIN_SEASON <= season <= max_season):
        print(
            f"FAIL: --season {season} is out of range ({MIN_SEASON}-{max_season}); refusing to run.",
            file=sys.stderr,
        )
        sys.exit(1)


def _validate_payload_has_conferences(payload: dict) -> None:
    """Exits non-zero when the built payload has zero conferences -- an in-range season with
    no ingested schedule_grid/teams data (wrong season, or not yet ingested) would otherwise
    report success identically to a real, populated one."""
    if not payload.get("conferences"):
        print(
            f"FAIL: built payload for season={payload.get('season')} has zero conferences; "
            f"refusing to publish an empty artifact.",
            file=sys.stderr,
        )
        sys.exit(1)


def _build_payload(season: int) -> dict:
    """Read-only. Composes the same fetch + build calls publish_schedule_artifact makes
    internally, so this always reflects exactly what that function would upload."""
    engine = _engine()
    try:
        rows = _fetch_schedule_grid_rows(engine, season)
        teams_meta = _fetch_teams_meta(engine, season)
        team_ranks = _fetch_team_ranks(engine, season)
        # Supplementary and non-fatal, matching publish_schedule_artifact's own handling: a
        # missing/empty non_fbs_teams table must not stop payload construction.
        try:
            non_fbs_logos = _fetch_non_fbs_logos(engine, season)
        except Exception as e:
            print(f"WARNING: could not read non_fbs_teams for season={season}; FCS opponents "
                  f"will render as text. Exception: {e}")
            non_fbs_logos = {}
    finally:
        engine.dispose()

    return build_schedule_payload(rows, teams_meta, season, team_ranks, non_fbs_logos)


def _verify_published(season: int, expected_payload: dict) -> None:
    """Round-trip verification: read schedule/{season}/latest.json back from R2 right after
    publish_schedule_artifact() runs, and diff it against the payload this run built (ignoring
    generated_at_utc, which legitimately differs between the two build times). This is the
    only check anywhere in this path that can distinguish a real, matching upload from one
    that was silently skipped or failed -- publish_schedule_artifact() itself never raises and
    always returns None regardless of which of its six bail-outs it hit. Exits non-zero on any
    mismatch, missing object, or read failure; prints a VERIFIED line on success.
    """
    from artifacts.r2 import get_r2_client  # local import: only exercised on the --publish path

    client = get_r2_client()
    if client is None:
        print("FAIL: R2 not configured; cannot verify the publish round-tripped.", file=sys.stderr)
        sys.exit(1)
    bucket = os.getenv("R2_BUCKET_NAME")
    if not bucket:
        print("FAIL: R2_BUCKET_NAME not set; cannot verify the publish round-tripped.", file=sys.stderr)
        sys.exit(1)

    key = f"schedule/{season}/latest.json"
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        published = json.loads(obj["Body"].read())
    except Exception as e:
        print(f"FAIL: could not read back {key} from R2 after publish: {e}", file=sys.stderr)
        sys.exit(1)

    expected = {k: v for k, v in expected_payload.items() if k != "generated_at_utc"}
    published_cmp = {k: v for k, v in published.items() if k != "generated_at_utc"}
    if expected != published_cmp:
        print(
            f"FAIL: published {key} does not match the payload this run built "
            f"(ignoring generated_at_utc) -- the publish did not round-trip.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"VERIFIED: {key} round-tripped from R2 and matches the payload this run built.")


def _print_summary(payload: dict) -> None:
    conferences = payload.get("conferences", [])
    team_count = sum(len(c["teams"]) for c in conferences)
    first_team_weeks = conferences[0]["teams"][0]["weeks"] if conferences and conferences[0]["teams"] else []
    cfp_slot_ids = {slot_id for slot_id, _label in CFP_SLOTS}

    # Resolved postseason CELLS (a real game -- non-null opponent -- sitting in a cfp-* slot),
    # not just column presence: build_canonical_columns() always emits all four CFP slot ids
    # for ANY non-empty season regardless of whether the postseason has actually been played
    # (they're placeholder/bye cells otherwise), so counting slot ids alone would read as "the
    # postseason is populated" when it may be nothing of the kind. Counted across every team,
    # not just the first, since that's what "populated" actually means.
    resolved_postseason_cells = sum(
        1
        for conf in conferences
        for team in conf["teams"]
        for wk in team["weeks"]
        if wk["slot_id"] in cfp_slot_ids and wk.get("opponent") is not None
    )

    print(f"season: {payload.get('season')}")
    print(f"conferences: {len(conferences)}")
    print(f"teams: {team_count}")
    print(f"columns (first team): {len(first_team_weeks)}")
    print(f"resolved postseason cells: {resolved_postseason_cells}")


def main() -> None:
    # Without this, artifacts/schedule.py's logger (named "cfb_lp") has no configured handler
    # when this script is run standalone (only main.py's own setup_logging() configures one),
    # so at the interpreter's default WARNING level, publish_schedule_artifact()'s own
    # logger.info("Published schedule artifact to R2 key %s", ...) confirmation -- the one
    # line a human or a workflow log-grep needs to tell a real publish from a skipped one --
    # is silently dropped. INFO here surfaces that line (and every other info/warning this
    # path logs) on stdout.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--season", type=int, required=True, help="Season year to publish, e.g. 2025.")
    parser.add_argument(
        "--publish", action="store_true",
        help="Actually publish to R2 and verify the round trip. Without this flag (the "
             "default, safe path), the payload is built and optionally written to --out, "
             "but nothing is uploaded.",
    )
    parser.add_argument("--out", type=str, default=None, help="Optional path to write the built payload as JSON.")
    args = parser.parse_args()

    _validate_season(args.season)

    payload = _build_payload(args.season)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote payload to {args.out}")

    _validate_payload_has_conferences(payload)

    if not args.publish:
        print(f"DRY RUN -- season={args.season}: payload built{' and written' if args.out else ''}; "
              f"nothing uploaded to R2. Pass --publish to actually publish.")
    else:
        print(f"Publishing season={args.season} to R2...")
        publish_schedule_artifact(args.season)
        print(f"publish_schedule_artifact({args.season}) returned; verifying the round trip...")
        _verify_published(args.season, payload)

    _print_summary(payload)


if __name__ == "__main__":
    main()
