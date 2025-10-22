import requests # type: ignore
import pandas as pd # type: ignore
from dotenv import load_dotenv # type: ignore
import os
from sqlalchemy import create_engine, Table, MetaData # type:ignore
from sqlalchemy.dialects.postgresql import insert # type:ignore

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.collegefootballdata.com"

def get_games_by_year_week(year, week=None, season_type='regular'):
    """
    Fetches game data from the College Football Data API for a given year and optional week.
    Args:
        year (int): Year of the season
        week (int, optional): Week number. Defaults to None.
        season_type (str, optional): Type of season ('regular', 'postseason', etc.). Defaults to 'regular'.
    Returns:
        pd.DataFrame: DataFrame containing game data
    """
    api_key = API_KEY
    url = f"{BASE_URL}/games"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "year": year,
        "seasonType": season_type,
    }
    if week:
        params["week"] = week
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    games_df = pd.DataFrame(response.json())
    games_df = games_df[["id","season","week","seasonType", "startDate","homeTeam","homePoints","awayTeam","awayPoints","neutralSite","conferenceGame","venue","venueId","homeConference","awayConference"]]
    games_df["id"] = pd.to_numeric(games_df["id"], errors="coerce").fillna(0).astype("Int64")
    games_df["venueId"] = pd.to_numeric(games_df["venueId"], errors="coerce").fillna(0).astype(int)
    games_df["homePoints"] = pd.to_numeric(games_df["homePoints"], errors="coerce").fillna(0).astype("Int64")
    games_df["awayPoints"] = pd.to_numeric(games_df["awayPoints"], errors="coerce").fillna(0).astype("Int64")
    games_df["margin"]  = abs(games_df["homePoints"] - games_df["awayPoints"])
    games_df["winner"] = games_df.apply(lambda row: row["homeTeam"] if row["homePoints"] > row["awayPoints"] else row["awayTeam"], axis=1)
    games_df["alpha"] = games_df.apply(lambda row: 1 if row["neutralSite"] else (0.8 if row["homeTeam"] == row["winner"] else 1.2), axis=1)
    
    games_df = games_df.rename(columns={
        'seasonType': 'season_type',
        'startDate': 'start_date',
        'homeTeam': 'home_team',
        'homePoints': 'home_score',
        'awayTeam': 'away_team',
        'awayPoints': 'away_score',
        'neutralSite': 'neutral_site',
        'conferenceGame': 'conference_game',
        'venueId': 'venueid',
        'homeConference': 'home_conference',
        'awayConference': 'away_conference',
    })

    games_df['home_score'] = games_df['home_score'].astype('Int64')
    games_df['away_score'] = games_df['away_score'].astype('Int64')
    games_df['margin'] = games_df['margin'].astype('Int64')
    games_df['venueid'] = games_df['venueid'].astype('Int64')
    games_df['alpha'] = games_df['alpha'].astype(float)
    games_df['neutral_site'] = games_df['neutral_site'].astype(bool)
    games_df['conference_game'] = games_df['conference_game'].astype(bool)
    games_df['start_date'] = pd.to_datetime(games_df['start_date'])

    # Get FBS teams from teams table
    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    engine = create_engine(db_url)
    fbs_teams = pd.read_sql_query(f"SELECT school FROM teams WHERE season = {year}", engine)
    fbs_team_list = set(fbs_teams['school'].tolist())

    # Filter games_df to only FBS teams
    games_df = games_df[
        games_df['home_team'].isin(fbs_team_list) | games_df['away_team'].isin(fbs_team_list)
    ]

    return games_df

def load_games_to_db(year, week=None, season_type='regular'):
    """
    Loads game data into the database for a given year and optional week.
    Uses upsert (insert/update) to handle existing records.
    
    Args:
        year (int): Year of the season
        week (int, optional): Week number. Defaults to None.
        season_type (str, optional): Type of season ('regular', 'postseason', etc.). Defaults to 'regular'.
    """
    games_df = get_games_by_year_week(year, week, season_type)
    games_df = games_df.where(pd.notnull(games_df), None)

    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    engine = create_engine(db_url)
    
    metadata = MetaData()
    table = Table('games', metadata, autoload_with=engine)
    
    with engine.begin() as conn:
        for _, row in games_df.iterrows():
            stmt = insert(table).values(**row.to_dict())
            update_dict = {c: getattr(stmt.excluded, c) for c in games_df.columns if c != 'id'}
            stmt = stmt.on_conflict_do_update(
                index_elements=['id'],
                set_=update_dict
            )
            conn.execute(stmt)
    
    engine.dispose()
    print(f"Games for year {year} week {week} loaded into DB (upsert completed).")