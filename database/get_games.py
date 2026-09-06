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

    games_data = response.json()

    # CFBD legitimately returns an empty list for a season/season_type with no
    # games yet (e.g. a future season's postseason before conference championship
    # week has happened) -- pd.DataFrame([]) has zero columns, so the column
    # selection below would raise KeyError on every such call. Short-circuit with
    # an empty DataFrame carrying the correct final columns so load_games_to_db's
    # empty-iterrows loop is a clean no-op instead of a raised (if caught)
    # exception on every weekly pipeline run until real data appears.
    if not games_data:
        return pd.DataFrame(columns=[
            "id", "season", "week", "season_type", "start_date", "home_team", "home_score",
            "away_team", "away_score", "neutral_site", "conference_game", "venue", "venueid",
            "home_conference", "away_conference", "margin", "winner", "alpha", "notes",
            "playoff_round_name", "playoff_round_order", "playoff_bracket_slot", "playoff_bowl_name",
        ])

    # Defensive extraction of the nested `playoff` object (present only on
    # CFP-affiliated games -- most games, including all regular-season and
    # non-CFP postseason games, have no `playoff` object at all). Never
    # raise on a missing object or a missing/renamed field within it --
    # default to None so the vast majority of rows simply carry nulls here.
    for game in games_data:
        playoff = game.get("playoff") or {}
        if not isinstance(playoff, dict):
            playoff = {}
        game["playoffRoundName"] = playoff.get("round_name") or playoff.get("roundName")
        game["playoffRoundOrder"] = playoff.get("round_order") or playoff.get("roundOrder")
        game["playoffBracketSlot"] = playoff.get("bracket_slot") or playoff.get("bracketSlot")
        game["playoffBowlName"] = playoff.get("bowl_name") or playoff.get("bowlName")

    games_df = pd.DataFrame(games_data)
    games_df = games_df[["id","season","week","seasonType", "startDate","homeTeam","homePoints","awayTeam","awayPoints","neutralSite","conferenceGame","venue","venueId","homeConference","awayConference","notes","playoffRoundName","playoffRoundOrder","playoffBracketSlot","playoffBowlName"]]
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
        'playoffRoundName': 'playoff_round_name',
        'playoffRoundOrder': 'playoff_round_order',
        'playoffBracketSlot': 'playoff_bracket_slot',
        'playoffBowlName': 'playoff_bowl_name',
    })

    games_df['home_score'] = games_df['home_score'].astype('Int64')
    games_df['away_score'] = games_df['away_score'].astype('Int64')
    games_df['margin'] = games_df['margin'].astype('Int64')
    games_df['venueid'] = games_df['venueid'].astype('Int64')
    games_df['alpha'] = games_df['alpha'].astype(float)
    games_df['neutral_site'] = games_df['neutral_site'].astype(bool)
    games_df['conference_game'] = games_df['conference_game'].astype(bool)
    games_df['start_date'] = pd.to_datetime(games_df['start_date'])
    # playoff_round_order: nullable int, no fillna -- absent for the vast
    # majority of rows (non-playoff games), and that null is the correct
    # value, not a default to paper over.
    games_df['playoff_round_order'] = pd.to_numeric(games_df['playoff_round_order'], errors="coerce").astype("Int64")
    # playoff_bracket_slot is documented by CFBD as a STRING field (not
    # numeric) -- do not cast it. A numeric coercion here would silently
    # null out any real value that isn't purely digits (e.g. a bracket
    # label), which is exactly the kind of silent data loss this column
    # exists to avoid.

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