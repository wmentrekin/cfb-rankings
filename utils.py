import os
import sys
from model.model import get_ratings
from database.model_to_db import ratings_to_df, insert_model_results_to_db
from database.get_games import load_games_to_db
import pandas as pd #type: ignore
import logging
from datetime import datetime, date, timedelta

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
        aug25 = date(year, 8, 24) # default: first sunday after week 0
        season_start = aug25 - timedelta(days=aug25.weekday())
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