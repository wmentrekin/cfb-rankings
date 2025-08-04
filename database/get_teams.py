import requests
import pandas as pd # type: ignore
from dotenv import load_dotenv # type: ignore
import os
from sqlalchemy import create_engine # type: ignore

load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.collegefootballdata.com"

def get_fbs_teams_by_year(year):
    api_key = API_KEY
    url = f"{BASE_URL}/teams/fbs?year={year}"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    teams_df = pd.DataFrame(response.json())

    teams_df["alternateNames"] = teams_df["alternateNames"].apply(list)
    teams_df["logos"] = teams_df["logos"].apply(list)
    location = pd.json_normalize(teams_df["location"])
    location = location.rename(columns={'id': 'stadium_id'})
    teams_df = pd.concat([teams_df.drop('location', axis=1), location], axis=1)
    teams_df["elevation"] = teams_df["elevation"].astype(float)
    teams_df["constructionYear"] = teams_df["constructionYear"].fillna(0).astype(int)
    teams_df["capacity"] = teams_df["capacity"].fillna(0).astype(int)
    teams_df["season"] = year

    return teams_df

def load_teams_to_db(year):
    teams_df = get_fbs_teams_by_year(year)
    teams_df = teams_df.where(pd.notnull(teams_df), None)

    # Use SQLAlchemy engine for DB connection
    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    engine = create_engine(db_url)
    with engine.begin() as conn:
        rows = [
            (
                row["id"],
                row["season"],
                row["school"],
                row["mascot"],
                row["abbreviation"],
                row["alternateNames"],
                row["conference"],
                row["division"],
                row["classification"],
                row["color"],
                row["alternateColor"],
                row["logos"],
                row["twitter"],
                row["stadium_id"],
                row["name"],
                row["city"],
                row["state"],
                row["zip"],
                row["countryCode"],
                row["timezone"],
                row["latitude"],
                row["longitude"],
                row["elevation"],
                row["capacity"],
                row["constructionYear"],
                row["grass"],
                row["dome"]
            ) for _, row in teams_df.iterrows()
        ]
        insert_sql = """
        INSERT INTO teams (
            id, season, school, mascot, abbreviation, alternateNames,
            conference, division, classification, color, alternateColor, logos, twitter,
            stadium_id, name, city, state, zip, countryCode, timezone,
            latitude, longitude, elevation, capacity, constructionYear, grass, dome
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id, season) DO NOTHING;
        """
        conn.execute(insert_sql, rows)
    print(f"Teams for the year {year} loaded successfully into the database.")