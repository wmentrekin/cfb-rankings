"""One-off repair: clear `winner` on games that were never played.

BACKGROUND
----------
`database/get_games.py` used to compute a game's winner with a plain score
comparison. CFBD returns an unplayed game as 0-0, and `0 > 0` is False, so every
future or cancelled game was written to `games.winner` as an AWAY-TEAM WIN. The
source bug is fixed -- get_games.py now writes NULL when both scores are 0 -- but
that fix only applies to rows re-fetched afterwards. Rows already in the table
stay wrong until they are either re-fetched or repaired.

BLAST RADIUS (measured 2026-09-08, production)
----------------------------------------------
  season 2022:    6 rows    season 2023:   18 rows
  season 2024:   14 rows    season 2026:  793 rows
  season 2025:    0 rows

The 2026 rows are simply the rest of the season's schedule, and self-heal week by
week as the pipeline re-fetches them. The 2022-2024 rows do NOT self-heal: they
are real cancelled or postponed games (e.g. App State at Liberty, 2024-09-28,
called off for Hurricane Helene) that have been counted as away-team wins in
every historical rating run since.

The Season Grid is unaffected -- `schedule_grid` derives status from the scores
and start_date, never from `winner`. The rating model IS affected: process_data.py
reads `winner` from `rankings_games`.

REPAIR RULE
-----------
Set winner = NULL wherever home_score = 0 AND away_score = 0 -- byte-for-byte the
same condition get_games.py now applies at write time, so this brings existing
rows in line with the fixed code rather than inventing a new rule. A genuine 0-0
final is not possible in modern college football (overtime has settled ties since
1996; the last 0-0 FBS game was 1983), and the surrounding code already treats
0-0 as "not played" regardless.

Fully reversible: `winner` is derivable from the scores alone.

USAGE
-----
    python scripts/repair_unplayed_game_winners.py            # dry run, prints counts
    python scripts/repair_unplayed_game_winners.py --apply    # performs the update
"""
import argparse
import os

import pandas as pd  # type: ignore
from dotenv import load_dotenv  # type: ignore
from sqlalchemy import create_engine, text  # type: ignore

SELECT_AFFECTED = """
SELECT season, COUNT(*) AS rows, MIN(start_date) AS earliest, MAX(start_date) AS latest
FROM games
WHERE home_score = 0 AND away_score = 0 AND winner IS NOT NULL
GROUP BY season
ORDER BY season;
"""

SAMPLE_AFFECTED = """
SELECT season, week, home_team, away_team, winner, start_date
FROM games
WHERE home_score = 0 AND away_score = 0 AND winner IS NOT NULL AND season < 2026
ORDER BY season, start_date
LIMIT 20;
"""

REPAIR = """
UPDATE games
SET winner = NULL
WHERE home_score = 0 AND away_score = 0 AND winner IS NOT NULL;
"""


def _engine():
    load_dotenv()
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Perform the update (default: dry run).")
    args = parser.parse_args()

    engine = _engine()
    try:
        affected = pd.read_sql_query(SELECT_AFFECTED, engine)
        if affected.empty:
            print("Nothing to repair: no 0-0 game carries a winner.")
            return

        print("Rows to repair, by season:")
        print(affected.to_string(index=False))
        total = int(affected["rows"].sum())

        historical = pd.read_sql_query(SAMPLE_AFFECTED, engine)
        if not historical.empty:
            print("\nPre-2026 rows (these are real cancelled/postponed games, and have been "
                  "miscounted in every rating run for their season):")
            print(historical.to_string(index=False))

        if not args.apply:
            print(f"\nDRY RUN -- {total} row(s) would be repaired. Re-run with --apply to perform it.")
            return

        with engine.begin() as conn:
            result = conn.execute(text(REPAIR))
        print(f"\nRepaired {result.rowcount} row(s).")

        remaining = pd.read_sql_query(SELECT_AFFECTED, engine)
        if remaining.empty:
            print("Verified: no 0-0 game carries a winner any more.")
        else:
            print("WARNING -- rows still affected after the update:")
            print(remaining.to_string(index=False))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
