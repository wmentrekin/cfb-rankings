import argparse
from typing import Dict, Optional
from model.model import get_ratings
from database.model_to_db import ratings_to_df, insert_model_results_to_db
from database.get_games import load_games_to_db
from database.get_teams import load_teams_to_db
from database.get_non_fbs_teams import load_non_fbs_teams_to_db
from artifacts.r2 import publish_rankings_artifact
from artifacts.schedule import publish_schedule_artifact
from utils import football_day, get_cfb_week, setup_logging
import pandas as pd #type: ignore
from datetime import datetime, date, timezone
import os
from dotenv import load_dotenv # type: ignore
from sqlalchemy import create_engine # type: ignore

def teams_exist_for_year(year):
    """
    Check whether the teams table already has any rows for the given season.
    Args:
        year (int): Season year to check.
    Returns:
        bool: True if at least one teams row exists for that season.
    """
    load_dotenv()
    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    engine = create_engine(db_url)
    count_df = pd.read_sql_query(f"SELECT COUNT(*) AS n FROM teams WHERE season = {year};", engine)
    engine.dispose()
    return int(count_df['n'].iloc[0]) > 0

# A run delayed past more than this many week boundaries is an operational anomaly, not
# something to silently resolve: walking back arbitrarily far lets a single stale
# future-dated row drag the published week backwards. Beyond the cap the run keeps the
# date-derived week and says so loudly; --week is the deliberate override for a genuinely
# long delay.
MAX_WALK_BACK_WEEKS = 2


def _naive_utc(value) -> Optional[datetime]:
    """Coerce one start_date to a naive-UTC datetime, or None if it cannot be read.

    `games.start_date` is naive UTC today, but a NULL (NaT) or a tz-aware value must not
    take down the run -- week selection is a convenience, never a reason to publish nothing.
    """
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if stamp is pd.NaT or pd.isna(stamp):
        return None
    moment = stamp.to_pydatetime()
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def resolve_week_from_starts(candidate: int, start_dates, now: datetime, season_start_override=None) -> int:
    """The pure decision behind resolve_target_week(), split out so it is testable without a DB.

    Returns the greatest week at or below `candidate` that BOTH has games and has no game
    still waiting to kick off. Requiring "has games" matters: without it an empty bucket --
    e.g. week 16 in a season whose regular slate ends at 15 -- reads as vacuously complete
    and the run republishes it as though it were new.

    `now` must be naive UTC, matching `games.start_date`.
    """
    games_by_week: Dict[int, int] = {}
    unstarted_by_week: Dict[int, int] = {}
    for raw in start_dates:
        start = _naive_utc(raw)
        if start is None:
            continue
        bucket = get_cfb_week(football_day(start), season_start_override)
        games_by_week[bucket] = games_by_week.get(bucket, 0) + 1
        if start > now:
            unstarted_by_week[bucket] = unstarted_by_week.get(bucket, 0) + 1

    floor = max(1, candidate - MAX_WALK_BACK_WEEKS)
    for week in range(candidate, floor - 1, -1):
        if games_by_week.get(week, 0) == 0:
            print(f"resolve_target_week: week {week} has no games; looking further back.")
            continue
        if unstarted_by_week.get(week, 0) > 0:
            print(
                f"resolve_target_week: week {week} still has {unstarted_by_week[week]} of "
                f"{games_by_week[week]} game(s) yet to kick off; looking further back."
            )
            continue
        if week != candidate:
            print(f"resolve_target_week: resolved to week {week} (date-derived week was {candidate}).")
        return week

    print(
        f"resolve_target_week: no completed week found within {MAX_WALK_BACK_WEEKS} week(s) of "
        f"the date-derived week {candidate}; keeping {candidate}. Pass --week to override."
    )
    return candidate


def resolve_target_week(year: int, today: date, season_start_override=None) -> int:
    """The week this run should publish: the last one whose games have all kicked off.

    get_cfb_week(today) answers "which week is it now", which is only the same question as
    "which week should we publish" because the cron happens to fire on a Sunday. A run that
    slips past a week boundary mislabels the rankings -- the delayed 2026 Week 1 run (Tue
    Sep 8, after a Mon Sep 7 game) computed week 2 and would have published Week 1's slate
    as Week 2, which is why that one run was hand-pinned with --week 1.

    Never raises: week selection is a convenience, and a failure here must not stop the run.
    Any problem falls back to the date-derived week, which is today's behavior.
    """
    candidate = get_cfb_week(today=today, season_start_override=season_start_override)
    if candidate < 1:
        return candidate

    try:
        load_dotenv()
        db_url = (
            f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
            "?sslmode=require"
        )
        engine = create_engine(db_url)
        try:
            games_df = pd.read_sql_query(
                f"SELECT start_date FROM games WHERE season = {int(year)} AND season_type = 'regular';",
                engine,
            )
        finally:
            engine.dispose()

        if games_df.empty:
            print(f"resolve_target_week: no {year} games loaded yet; using date-derived week={candidate}.")
            return candidate

        # Naive UTC to match games.start_date -- datetime.now() is naive LOCAL time, which is
        # only correct by accident on a UTC CI runner.
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        return resolve_week_from_starts(
            candidate, list(games_df["start_date"]), now_utc, season_start_override
        )
    except Exception as exc:  # noqa: BLE001 -- never let week selection break the run
        print(f"resolve_target_week: week selection failed ({exc}); falling back to week={candidate}.")
        return candidate


def main():
    """
    Main function to run the model and handle data loading and saving.
    Args:
        year (int): Year of the season
        week (int, optional): Week number. Defaults to None.
        staging (bool, optional): Whether to write to staging table. Defaults to False.
        season_start (str, optional): YYYY-MM-DD to override season start date for week calculation
    Returns:
        None
    Raises:
        Various exceptions for data loading, model execution, and database insertion
    Notes:
        - Uses argparse for command line arguments
        - Logs progress and errors
        - Loads game data into the database before running the model
        - Converts model results to DataFrame and inserts into database
    """
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
        today = datetime.now(timezone.utc).date()
        args.week = resolve_target_week(args.year, today, season_start_override)
        print(f"No --week provided: computed week={args.week} (last week whose games have all started).")

    # SETUP LOGGING
    logger = setup_logging(args.year, args.week)
    logger.info("Starting model run: year=%s week=%s staging=%s", args.year, args.week, args.staging)

    # GUARD AGAINST WEEK-0 (OR EARLIER) RUNS
    # The system doesn't produce meaningful ratings before Week 1 games have been played (week 0
    # is the pre-season placeholder get_cfb_week() returns before the season's first Sunday).
    # Applies identically whether --week was omitted (auto-computed) or passed explicitly as 0/negative.
    if args.week < 1:
        logger.error(
            "Refusing to run for year=%s week=%s: the system doesn't produce meaningful ratings "
            "before Week 1 games have been played. Exiting before any data load, model run, DB "
            "write, or artifact publish.",
            args.year, args.week,
        )
        return

    # ENSURE TEAMS LOADED
    try:
        if teams_exist_for_year(args.year):
            logger.info("Teams already present in DB for year=%s; skipping team load.", args.year)
        else:
            logger.info("No teams found in DB for year=%s; loading FBS teams before game ingestion.", args.year)
            load_teams_to_db(args.year)
            logger.info("Teams for year=%s loaded into DB.", args.year)
    except Exception as e:
        logger.exception("Team loading check/load failed for year=%s: %s", args.year, e)
        logger.error("Exiting: cannot proceed with game ingestion without a teams table for year=%s.", args.year)
        return

    # LOAD NON-FBS TEAM LOGO DATA (supplementary, not a model input)
    # `non_fbs_teams` is a separate table from `teams` -- it does NOT feed the
    # model's team list or the games/schedule-grid ingestion, so a failure
    # here must never block the pipeline. Always attempted (not gated behind
    # an existence check like the FBS teams block above) since the upsert on
    # (season, school) is idempotent and cheap, mirroring how postseason
    # games are always re-fetched below.
    try:
        logger.info("Loading non-FBS Division-I team logo data for year=%s", args.year)
        non_fbs_result = load_non_fbs_teams_to_db(args.year)
        logger.info(
            "Non-FBS teams for year=%s: %s stored, %s with a non-empty logos array.",
            args.year, non_fbs_result.get('stored'), non_fbs_result.get('with_logos'),
        )
    except Exception as e:
        logger.warning("Non-FBS team logo loading raised an exception. Continuing. Exception: %s", e)

    # PULL DATA
    try:
        logger.info("Loading games into DB for year=%s week=%s", args.year, args.week)
        load_games_to_db(args.year, args.week)
    except Exception as e:
        logger.warning("Games loading raised an exception (they may already be loaded). Continuing. Exception: %s", e)

    # PULL POSTSEASON DATA
    # Postseason week numbers are a different, colliding numbering scheme from
    # the regular-season pipeline-week cursor (args.week) -- always fetch the
    # whole postseason slate (week=None) rather than reusing args.week, which
    # would silently fetch the wrong/empty postseason games every run. Cheap
    # (a few dozen games) and safe given load_games_to_db's upsert idempotency.
    try:
        logger.info("Loading postseason games into DB for year=%s", args.year)
        load_games_to_db(args.year, week=None, season_type='postseason')
    except Exception as e:
        logger.warning("Postseason games loading raised an exception (they may already be loaded). Continuing. Exception: %s", e)

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

    # PUBLISH RANKINGS ARTIFACT
    if not args.staging:
        try:
            publish_rankings_artifact(args.year, args.week)
        except Exception as ex:
            logger.exception("Rankings artifact publish step failed unexpectedly: %s", ex)
    else:
        logger.info("Skipping rankings artifact publish because --staging was set.")

    # PUBLISH SCHEDULE (SEASON GRID) ARTIFACT
    # Season-scoped, not per-week -- no `week` argument. Own try/except so a schedule-publish
    # failure never affects (or is affected by) the rankings publish above.
    if not args.staging:
        try:
            publish_schedule_artifact(args.year)
        except Exception as ex:
            logger.exception("Schedule artifact publish step failed unexpectedly: %s", ex)
    else:
        logger.info("Skipping schedule artifact publish because --staging was set.")

    logger.info("Run finished successfully for year=%s week=%s", args.year, args.week)

if __name__ == "__main__":
    main()