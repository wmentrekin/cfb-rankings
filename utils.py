import os
import sys
import inspect
import argparse
from model.model import get_ratings
from database.model_to_db import ratings_to_df, insert_model_results_to_db
from database.get_games import load_games_to_db
import pandas as pd #type: ignore
import logging
from datetime import datetime, date, timedelta

def get_cfb_week(today: date, season_start_override: date) -> int:
    """
    Compute CFB week number from a given date.
    Default rule: week 0 starts on `season_start_override` or a reasonable default.
    Adjust season_start_override each year to the actual first week date you want.
    Returns integer week in [0, 16] (cap).
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
    """Create logger that prints to stdout and writes to a file."""
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