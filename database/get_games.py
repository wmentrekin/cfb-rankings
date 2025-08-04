import requests
import pandas as pd # type: ignore
from dotenv import load_dotenv # type: ignore
import os
from sqlalchemy import create_engine # type:ignore

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.collegefootballdata.com"

def get_games_by_year_week(year, week=None, season_type='regular'):
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

    return games_df

def load_games_to_db(year, week=None, season_type='regular'):
    games_df = get_games_by_year_week(year, week, season_type)
    games_df = games_df.where(pd.notnull(games_df), None)

    # Use SQLAlchemy engine for DB connection
    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    engine = create_engine(db_url)
    with engine.begin() as conn:
        # Use pandas to_sql for bulk insert (if you want to use raw SQL, you can use conn.execute)
        # Here is an example using raw SQL for upsert:
        rows = [
            (
                row["id"],
                row["season"],
                row["week"],
                row["seasonType"],
                row["startDate"],
                row["homeTeam"],
                row["homePoints"],
                row["awayTeam"],
                row["awayPoints"],
                row["neutralSite"],
                row["conferenceGame"],
                row["venue"],
                row["venueId"],
                row["homeConference"],
                row["awayConference"],
                row["margin"],
                row["winner"],
                row["alpha"]
            )
            for _, row in games_df.iterrows()
        ]
        insert_sql = """
            INSERT INTO games (id, season, week, season_type, start_date, home_team, home_score, away_team, away_score, neutral_site, conference_game, venue, venueId, home_conference, away_conference, margin, winner, alpha)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                season = EXCLUDED.season,
                week = EXCLUDED.week,
                season_type = EXCLUDED.season_type,
                start_date = EXCLUDED.start_date,
                home_team = EXCLUDED.home_team,
                home_score = EXCLUDED.home_score,
                away_team = EXCLUDED.away_team,
                away_score = EXCLUDED.away_score,
                neutral_site = EXCLUDED.neutral_site,
                conference_game = EXCLUDED.conference_game,
                venue = EXCLUDED.venue,
                venueId = EXCLUDED.venueId,
                home_conference = EXCLUDED.home_conference,
                away_conference = EXCLUDED.away_conference,
                margin = EXCLUDED.margin,
                winner = EXCLUDED.winner,
                alpha = EXCLUDED.alpha
        """
        conn.execute(insert_sql, rows)
    print(f"Loaded {len(rows)} games for {year} week {week} into the database.")