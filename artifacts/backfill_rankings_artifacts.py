import argparse

from artifacts.r2 import publish_rankings_artifact
from utils import setup_logging


def main():
    """
    Backfill rankings artifacts to R2 for every week of a season already present in `ratings`.
    Reuses publish_rankings_artifact per week -- no model re-run, just reads existing DB rows.
    """
    parser = argparse.ArgumentParser(description="Backfill rankings artifacts to R2 for a season.")
    parser.add_argument("--year", type=int, required=True, help="Season year to backfill.")
    parser.add_argument("--start-week", type=int, default=1, help="First week to backfill (default 1).")
    parser.add_argument("--end-week", type=int, required=True, help="Last week to backfill (inclusive).")
    args = parser.parse_args()

    logger = setup_logging(args.year, args.start_week)
    logger.info("Backfilling rankings artifacts for year=%s weeks=%s-%s", args.year, args.start_week, args.end_week)

    for week in range(args.start_week, args.end_week + 1):
        logger.info("Publishing artifact for year=%s week=%s", args.year, week)
        publish_rankings_artifact(args.year, week)

    logger.info("Backfill finished for year=%s weeks=%s-%s", args.year, args.start_week, args.end_week)


if __name__ == "__main__":
    main()
