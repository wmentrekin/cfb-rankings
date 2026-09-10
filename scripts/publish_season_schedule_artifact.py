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
    write. But it is a plain function, not reachable from the command line, and it has no
    dry-run mode of its own.

This script is that missing entry point: `--season N` builds one season's Season Grid
payload and, unless `--dry-run` is passed, publishes it. Nothing else in the pipeline runs.

BLAST RADIUS
------------
`--dry-run`: read-only against the database (see CREDENTIALS REQUIRED below). Writes
nothing anywhere except the optional `--out` file on local disk.

Without `--dry-run`: overwrites the R2 keys `schedule/{season}/latest.json` and
`schedule/index.json` in place -- an atomic per-object PUT, not a delete-then-write. No
other R2 key and no database table is touched; every DB call in this path (both here and
inside publish_schedule_artifact) is a SELECT.

CREDENTIALS REQUIRED
---------------------
  --dry-run:      DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME only (read-only
                   queries against schedule_grid, teams, ratings, non_fbs_teams).
  without it:     the same DB_* set, plus R2_ACCOUNT_ID / R2_ACCESS_KEY_ID /
                   R2_SECRET_ACCESS_KEY / R2_BUCKET_NAME -- publish_schedule_artifact's own
                   requirement (see artifacts/r2.py's get_r2_client()).

Usage:
    python scripts/publish_season_schedule_artifact.py --season 2025 --dry-run --out payload.json
    python scripts/publish_season_schedule_artifact.py --season 2025 --out payload.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# See scripts/backfill_2025_postseason.py for why this bootstrap is needed -- running this
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
# for --dry-run and for --out, exactly the payload publish_schedule_artifact builds internally
# -- without duplicating or diverging from its query logic. Do not edit schedule.py to avoid
# this coupling; escalate instead (see the handoff).


def _engine():
    load_dotenv()
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )


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


def _print_summary(payload: dict) -> None:
    conferences = payload.get("conferences", [])
    team_count = sum(len(c["teams"]) for c in conferences)
    first_team_weeks = conferences[0]["teams"][0]["weeks"] if conferences and conferences[0]["teams"] else []
    cfp_slot_ids = {slot_id for slot_id, _label in CFP_SLOTS}
    postseason_slot_ids_present = sorted(
        wk["slot_id"] for wk in first_team_weeks if wk["slot_id"] in cfp_slot_ids
    )

    print(f"season: {payload.get('season')}")
    print(f"conferences: {len(conferences)}")
    print(f"teams: {team_count}")
    print(f"columns (first team): {len(first_team_weeks)}")
    print(f"postseason slot ids present: {postseason_slot_ids_present}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--season", type=int, required=True, help="Season year to publish, e.g. 2025.")
    parser.add_argument("--dry-run", action="store_true", help="Build and write the payload; upload nothing to R2.")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write the built payload as JSON.")
    args = parser.parse_args()

    payload = _build_payload(args.season)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote payload to {args.out}")

    if args.dry_run:
        print(f"DRY RUN -- season={args.season}: payload built{' and written' if args.out else ''}; "
              f"nothing uploaded to R2.")
    else:
        print(f"Publishing season={args.season} to R2...")
        publish_schedule_artifact(args.season)
        print(f"Done. publish_schedule_artifact({args.season}) completed "
              f"(see its own logs above for the R2 upload result).")

    _print_summary(payload)


if __name__ == "__main__":
    main()
