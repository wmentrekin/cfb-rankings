import os
import sys
from model.model import get_ratings
from database.model_to_db import ratings_to_df, insert_model_results_to_db
from database.get_games import load_games_to_db
import pandas as pd #type: ignore
import logging
from datetime import datetime, date, timedelta

# A kickoff at 8pm ET on a Monday is 00:00 UTC on TUESDAY. Bucketing on the raw UTC date
# therefore pushes exactly the late Monday-night games -- the ones the Tuesday->Monday
# boundary exists to capture -- into the following week anyway, leaving that fix half-done.
# Rolling the day over a few hours after midnight UTC fixes it.
#
# The window is 4 hours, and the exact value matters. CFBD stores a game whose kickoff time
# is not yet set as midnight ET, which is 05:00 UTC -- 2026 weeks 11, 12 and 13 each open
# with such a row. A 6-hour rollover pulls those placeholders back onto the preceding Monday
# and drags whole weeks with them; 4 hours leaves them on their real day while still
# capturing a genuine Monday-night kickoff (8-10pm ET = 00:00-02:00 UTC Tuesday).
#
# Only MONDAY's classification actually affects a bucket, since Tuesday->Monday weeks put
# Saturday and Sunday in the same week either way -- so a late Saturday west-coast game
# (04:00 UTC Sunday) landing on "Sunday" is harmless.
FOOTBALL_DAY_ROLLOVER_HOURS = 4


def football_day(start: datetime) -> date:
    """The calendar day a kickoff belongs to, rolling over at 06:00 UTC rather than midnight."""
    return (start - timedelta(hours=FOOTBALL_DAY_ROLLOVER_HOURS)).date()


def get_cfb_week(today: date, season_start_override: date) -> int:
    """
    Calculate the current CFB week number based on today's date and an optional season start override.
    Args:
        today (date): Current date. If None, uses today's date. Defaults to None.
        season_start_override (date, optional): If provided, uses this date as the season start instead of calculating the first Sunday after August 24th. Defaults to None.
    Returns:
        week_num (int): Current CFB week number (0-16)
    """
    if today is None:
        today = datetime.now().date()

    if season_start_override is None:
        year = today.year
        # Weeks run Tuesday -> Monday, not Monday -> Sunday. College football weeks
        # straddle the calendar week: Week 0 is a lone Saturday, Week 1 has run
        # Thursday through Monday (2026: Sep 3 - Sep 7), and later weeks run
        # Tuesday/Thursday through Saturday. Anchoring on Monday put a Monday-night
        # game in the FOLLOWING week, split from the weekend it belongs to.
        # Anchor on the Monday on/before Aug 24, then step forward one day to the
        # Tuesday. The +1 day is the whole fix: it moves Mondays back into the week
        # of the weekend that precedes them and leaves every other weekday alone.
        anchor_monday = date(year, 8, 24)
        anchor_monday -= timedelta(days=anchor_monday.weekday())
        season_start = anchor_monday + timedelta(days=1)
    else:
        season_start = season_start_override

    days_since_start = (today - season_start).days
    if days_since_start < 0:
        week_num = 0
    else:
        week_num = days_since_start // 7

    return max(0, min(16, week_num))

def setup_logging(year: int, week: int) -> logging.Logger:
    """Create logger that prints to stdout and writes to a file.
    Args:
        year (int): Year of the season
        week (int): Week number
    Returns:
        logger (logging.Logger): Configured logger instance
    """
    logger = logging.getLogger("cfb_lp")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # file handler
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"run_{year}_w{week}.log")
    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("Logger initialized, writing to %s", filename)
    return logger