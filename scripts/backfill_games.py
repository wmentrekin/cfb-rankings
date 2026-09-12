"""
Ingest one (season, season_type, week) slice of CFBD games into `games`, and do nothing else.

BACKGROUND
----------
There is no narrow way to fetch a single missing slice of a finished season. The two existing
entry points both do far more:

  - `main.py --year N --week W` re-ingests, then runs the rating model and overwrites that
    season's `ratings` rows and its rankings artifact. For a finished season that is a large,
    unwanted blast radius: it rewrites what the live Rankings tab serves for every team, to
    fix a gap that may be a single game.
  - `database.get_games.load_games_to_db(...)` is the correct narrow primitive -- it fetches
    CFBD and upserts `games`, touching nothing else -- but it is a plain function with no
    command-line entry point, and it reports nothing about WHAT it wrote. A caller cannot
    tell "fetched 1 new game" from "fetched 0 games and silently did nothing", which is the
    exact failure mode that lets a backfill look successful while changing nothing.

This script is that missing entry point. It fetches the slice, diffs it against what is
already in `games`, prints every row it found and classifies each as NEW or EXISTING, and
only writes when `--apply` is passed. After writing it re-reads the affected ids back out of
the database and fails loudly if any row it claimed to upsert is not actually there.

This supersedes a one-off 2025-postseason backfill script that used the same
load_games_to_db() upsert path but was hardcoded to one season/season_type, with no dry run,
no diff and no read-back. That script has been deleted; use this one.

WHY A DRY RUN MATTERS HERE
--------------------------
The upsert is ON CONFLICT DO UPDATE on `id`, so re-running is safe for rows whose CFBD
content is unchanged -- but it is NOT a no-op when CFBD has since revised a game. Fetching a
whole finished season to recover one missing row would silently re-write every other row in
that season with whatever CFBD reports today. The dry run is how you see that before it
happens: it prints the NEW/EXISTING split so you can scope `--week` down to only what is
actually missing.

MANUAL RESULT OVERRIDES ARE NOT RE-APPLIED BY THIS SCRIPT.
`database/game_result_overrides.json` is re-asserted by main.py AFTER its ingest calls, so
the weekly pipeline cannot lose an override to a re-ingest. This script has no such step --
it is ingest only. If you backfill a slice that contains an overridden game, that game
reverts to CFBD's reported result until the next main.py run re-asserts the override. Check
game_result_overrides.json against your slice before passing --apply.

BLAST RADIUS
------------
Default (no `--apply`): one CFBD GET and read-only SELECTs. Writes nothing anywhere.

`--apply`: upserts the fetched rows into the `games` table by `id`. No other table is
touched, nothing is deleted, and no R2 key is written -- republishing the affected season's
artifact is a separate, deliberate step (scripts/publish_season_schedule_artifact.py).

CREDENTIALS REQUIRED
--------------------
API_KEY (CFBD) plus DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME. Run it wherever those are
set -- locally with a populated .env, or via .github/workflows/backfill-games.yml, which
supplies the same GitHub Actions secrets the scheduled pipeline uses.

Usage:
    python scripts/backfill_games.py --season 2025 --week 16              # dry run
    python scripts/backfill_games.py --season 2025 --week 16 --apply      # writes
    python scripts/backfill_games.py --season 2025 --season-type postseason
"""

import argparse
import os
import sys
from pathlib import Path

# Running this file directly only puts scripts/ on sys.path, not the repo root, so
# `database` isn't importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, text  # type: ignore  # noqa: E402

from database.get_games import get_games_by_year_week, load_games_to_db  # noqa: E402

# Matches the bounds used by scripts/publish_season_schedule_artifact.py: a typo'd season
# should fail loudly here rather than quietly fetch an empty slate and report success.
MIN_SEASON = 2014
MAX_SEASON = 2100

VALID_SEASON_TYPES = ("regular", "postseason")


def _engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    return create_engine(url)


def _existing_ids(engine, game_ids):
    """The subset of `game_ids` already present in `games`. Empty input short-circuits without
    a query -- an `IN ()` with no elements is a syntax error in Postgres, not an empty set."""
    if not game_ids:
        return set()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id FROM games WHERE id = ANY(:ids)"),
            {"ids": list(game_ids)},
        ).fetchall()
    return {r[0] for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--week", type=int, default=None,
        help="CFBD week. Omit to fetch the WHOLE season_type -- read the dry run's NEW/EXISTING "
             "split before applying that, since it rewrites every existing row from CFBD.",
    )
    parser.add_argument("--season-type", default="regular", choices=VALID_SEASON_TYPES)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually upsert. Without it nothing is written (opt IN, matching "
             "publish_season_schedule_artifact.py's --publish).",
    )
    args = parser.parse_args()

    if not (MIN_SEASON <= args.season <= MAX_SEASON):
        print(f"ERROR: --season {args.season} outside [{MIN_SEASON}, {MAX_SEASON}].")
        return 1
    if args.week is not None and not (0 <= args.week <= 20):
        print(f"ERROR: --week {args.week} outside [0, 20].")
        return 1

    slice_desc = (
        f"season={args.season} season_type={args.season_type} "
        f"week={args.week if args.week is not None else 'ALL'}"
    )
    print(f"Fetching {slice_desc} from CFBD...")
    games_df = get_games_by_year_week(args.season, args.week, args.season_type)

    if games_df.empty:
        # Not an error on its own -- a season/week with no games is a legitimate CFBD answer
        # -- but it IS the outcome most likely to be mistaken for success, so say so plainly
        # and exit non-zero so a CI job surfaces it instead of going green on nothing.
        print(f"CFBD returned NO games for {slice_desc}. Nothing to back-fill.")
        return 1

    engine = _engine()
    fetched_ids = [int(i) for i in games_df["id"].tolist()]
    existing = _existing_ids(engine, fetched_ids)

    print(f"\nCFBD returned {len(games_df)} game(s) for {slice_desc}:\n")
    new_count = 0
    for _, row in games_df.iterrows():
        gid = int(row["id"])
        is_new = gid not in existing
        new_count += is_new
        print(
            f"  [{'NEW     ' if is_new else 'EXISTING'}] id={gid} week={row['week']} "
            f"{str(row['start_date'])[:16]}  {row['away_team']} @ {row['home_team']}  "
            f"{row['away_score']}-{row['home_score']}  conf_game={row['conference_game']}  "
            f"notes={row['notes']!r}"
        )
    print(f"\n{new_count} new, {len(fetched_ids) - new_count} already present.")

    if not args.apply:
        print(
            f"\nDRY RUN -- {slice_desc}: nothing written. Pass --apply to upsert these rows."
        )
        return 0

    print(f"\nApplying: upserting {len(fetched_ids)} row(s) into `games`...")
    load_games_to_db(args.season, args.week, args.season_type)

    # load_games_to_db returns None and raises only on a DB/HTTP error, so a clean return
    # does not by itself prove the rows are in the table. Read them back.
    landed = _existing_ids(engine, fetched_ids)
    missing = sorted(set(fetched_ids) - landed)
    if missing:
        print(f"ERROR: {len(missing)} row(s) are NOT in `games` after the upsert: {missing}")
        return 1

    print(f"Verified: all {len(fetched_ids)} row(s) present in `games` after upsert.")
    print(
        f"BACKFILL_APPLIED {slice_desc} fetched={len(fetched_ids)} new={new_count}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
