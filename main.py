import os
import sys
import inspect
import argparse
from model.model import get_ratings
from database.model_to_db import ratings_to_df, insert_model_results_to_db
from database.get_games import load_games_to_db
from utils import get_cfb_week, setup_logging
import pandas as pd #type: ignore
from datetime import datetime, date, timedelta

def main():
    
    # PARSE ARGS
    parser = argparse.ArgumentParser(description="Run CFB QP model and prepare data for DB upload.")
    parser.add_argument('--year', type=int, default=None, help='Season year (e.g., 2024). If not provided, current year is used.')
    parser.add_argument('--week', type=int, default=None, help='Week number (optional). If omitted, computed from date.')
    parser.add_argument('--staging', action='store_true', help='Write results to staging table (rankings_test).')
    parser.add_argument('--season-start', type=str, default=None, help='Optional YYYY-MM-DD to override season start date for week calc.')
    args = parser.parse_args()

    # DETERMINE YEAR AND WEEK
    if args.year is None:
        args.year = datetime.now().year
    season_start_override = None
    if args.season_start:
        try:
            y, m, d = [int(x) for x in args.season_start.split("-")]
            season_start_override = date(y, m, d)
        except Exception:
            print("Invalid --season-start format; expected YYYY-MM-DD. Ignoring override.")
            season_start_override = None
    if args.week is None:
        args.week = get_cfb_week(season_start_override=season_start_override)
        print(f"No --week provided: computed week={args.week} based on date.")

    # SETUP LOGGING
    logger = setup_logging(args.year, args.week)
    logger.info("Starting model run: year=%s week=%s staging=%s", args.year, args.week, args.staging)

    # PULL DATA
    try:
        logger.info("Loading games into DB for year=%s week=%s", args.year, args.week)
        load_games_to_db(args.year, args.week)
    except Exception as e:
        logger.warning("Games loading raised an exception (they may already be loaded). Continuing. Exception: %s", e)

    # RUN MODEL
    logger.info("Running model.get_ratings(year=%s, week=%s)", args.year, args.week)
    results = None
    try:
        results = get_ratings(args.year, args.week)
    except Exception as ex:
        logger.exception("Model raised an exception: %s", ex)
        logger.error("Exiting due to model error.")
        return
    if results is None:
        logger.error("Model returned None. Exiting.")
        return

    # PROCESS RESULTS
    try:
        ratings, records = results[0], results[1]
    except Exception:
        logger.exception("Unexpected return signature from get_ratings(). Expected (ratings, records).")
        return
    try:
        ratings_df = ratings_to_df(ratings, records, args.year, args.week)
    except Exception as ex:
        logger.exception("Failed to convert model results to DataFrame: %s", ex)
        return

    logger.info("Model returned %d rating rows; sample:", len(ratings_df))
    logger.info("\n%s", ratings_df.head(20).to_string(index=False))

    # INSERT TO DB
    try:
        insert_model_results_to_db(ratings_df, staging = args.staging)
        logger.info("Insert complete.")
    except Exception as ex:
        logger.exception("Insert failed: %s", ex)
        return

    logger.info("Run finished successfully for year=%s week=%s", args.year, args.week)

if __name__ == "__main__":
    main()