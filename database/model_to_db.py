import pandas as pd #type: ignore
from typing import Dict, List, Tuple, Any
from sqlalchemy import create_engine #type: ignore
import os
from dotenv import load_dotenv #type: ignore

def ratings_to_df(ratings: Dict[str, float], records: Dict[str, Any], season: int, week: int) -> pd.DataFrame:
    """
    Convert ratings and records to a DataFrame for database upload.
    Args:
        ratings: Dict of team name to rating.
        records: Dict of team name to (wins, losses).
        season: int, season year
        week: int, week number
    Returns:
        DataFrame with columns: team, rating, wins, losses, record, season, week
    """
    data = []
    for team in ratings.keys():
        if team != "fcs":
            wins = records[team][0]
            losses = records[team][1]
            ties = records[team][2]
            rating = ratings[team]
            # Convert numpy array to float if needed
            if hasattr(rating, "item"):
                rating = float(rating.item())
            else:
                rating = float(rating)
            if week is None:
                week = 0
            data.append({
                'team': team,
                'rating': rating,
                'wins': wins,
                'losses': losses,
                'ties': ties,
                'season': season,
                'week': week
            })
        else:
            print("Skipping fcs team")
    return pd.DataFrame(data)

def insert_model_results_to_db(ratings_df: pd.DataFrame):
    """
    Insert model results DataFrames into database tables.
    Args:
        ratings_df: DataFrame of ratings.
        fcs_losses_df: DataFrame of FCS losses.
        records_df: DataFrame of records.
        table_prefix: Prefix for table names (default "model_").
    """
    load_dotenv()
    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    engine = create_engine(db_url)
    ratings_df.to_sql(f"ratings", engine, if_exists="append", index=False)
    engine.dispose()
