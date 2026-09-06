"""
One-time backfill: 2025 postseason games (bowls + CFP).

Part of the schedule-grid feature (T2). Loads season_type='postseason' games
for the 2025 season into the `games` table via the existing, already-
idempotent load_games_to_db() upsert path (ON CONFLICT DO UPDATE on id) --
safe to re-run if it fails partway through or needs to be run again.

season=2026's postseason is NOT backfilled here on purpose: it effectively
backfills itself the first time main.py's new recurring postseason call
(added alongside this script in the same change) runs against production,
since 2026's postseason games play out over the following months and the
weekly pipeline run re-fetches season_type='postseason' every time.

Requires real credentials this sandbox does not have -- run this wherever
CFBD_API_KEY (referenced as API_KEY by database/get_games.py) and
DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME are set, e.g. locally with a
populated .env, or as a one-off CI job using the same GitHub Actions
secrets the scheduled pipeline uses.

Usage:
    python scripts/backfill_2025_postseason.py
    uv run python scripts/backfill_2025_postseason.py
"""

import sys
from pathlib import Path

# Running this file directly (`python scripts/backfill_2025_postseason.py`) only puts
# scripts/ on sys.path, not the repo root, so `database` isn't importable -- unlike
# main.py, which lives at the repo root itself. Bootstrap the repo root onto sys.path
# so this script works regardless of the current working directory it's invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.get_games import load_games_to_db

SEASON = 2025


def main():
    print(f"Backfilling season_type='postseason' games for season={SEASON}...")
    load_games_to_db(SEASON, week=None, season_type='postseason')
    print(f"Done. Postseason games for season={SEASON} loaded into DB (upsert completed).")


if __name__ == "__main__":
    main()
