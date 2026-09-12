"""
One-off cleanup: remove the premature 2026 Week 1 rankings artifact from R2
and restore rankings/latest.json + rankings/index.json to the correct state.

BACKGROUND: weekly-update.yml was manually triggered (workflow_dispatch) on
2026-09-06, before Week 1's Sun/Mon games had finished -- bypassing the
workflow's own gate, which only suppresses the *scheduled* cron trigger, not
manual dispatch, specifically to delay Week 1 until 2026-09-08. That run also
hit a since-fixed bug (unplayed games scored as away-team wins), corrupting
the published rankings. The `ratings` DB rows for season=2026/week=1 have
already been deleted directly. This script cleans up the R2 side, which
needs real R2 credentials this sandbox does not have.

What this does:
  1. Deletes rankings/2026/week-01.json and rankings/2026/latest.json --
     there should be NO 2026 rankings data published until the real Tuesday
     2026-09-08 run.
  2. Re-publishes 2025 week 15 (the last real published week before this
     incident) via the existing publish_rankings_artifact() -- idempotent,
     re-derives from the unchanged 2025 week-15 ratings already in the DB,
     so 2025/week-15.json and 2025/latest.json come out byte-identical to
     before. The important effect is that this overwrites the GLOBAL
     rankings/latest.json key back to 2025 week 15's payload (it was
     pointing at the corrupted 2026 week 1 payload), and rebuilds
     rankings/index.json fresh from the bucket's actual contents -- which,
     once step 1's deletes have landed, no longer include 2026 at all.

Requires real credentials this sandbox does not have -- run this wherever
CFBD_API_KEY/DB_*/R2_* are set (e.g. locally with a populated .env).

Usage:
    python scripts/cleanup_premature_2026_week1_rankings.py
    uv run python scripts/cleanup_premature_2026_week1_rankings.py
"""

import os
import sys
from pathlib import Path

# Running this file directly only puts scripts/ on sys.path, not the repo root, so
# running this file directly only puts scripts/ on sys.path, not the repo
# root, so `artifacts` isn't importable without it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # type: ignore

from artifacts.r2 import get_r2_client, publish_rankings_artifact

KEYS_TO_DELETE = [
    "rankings/2026/week-01.json",
    "rankings/2026/latest.json",
]

# The last real published week before the incident (see ratings table:
# season=2025 week=15 was the most recent non-2026 entry, and 2026 had no
# other week besides the corrupted week=1).
RESTORE_YEAR = 2025
RESTORE_WEEK = 15


def main():
    load_dotenv()
    client = get_r2_client()
    if client is None:
        print("R2 not configured (R2_ACCOUNT_ID/R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY); aborting.")
        return

    bucket = os.getenv("R2_BUCKET_NAME")
    if not bucket:
        print("R2_BUCKET_NAME not configured; aborting.")
        return

    for key in KEYS_TO_DELETE:
        try:
            client.delete_object(Bucket=bucket, Key=key)
            print(f"Deleted {key}")
        except Exception as e:
            print(f"Failed to delete {key}: {e}")

    print(f"Restoring rankings/latest.json and rankings/index.json to season={RESTORE_YEAR} week={RESTORE_WEEK}...")
    publish_rankings_artifact(RESTORE_YEAR, RESTORE_WEEK)
    print("Done. Verify: rankings/2026/ should not exist in R2, and rankings/latest.json"
          f" + rankings/index.json should reflect season={RESTORE_YEAR} week={RESTORE_WEEK}.")


if __name__ == "__main__":
    main()
