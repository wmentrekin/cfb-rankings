import requests # tye: ignore
import pandas as pd # type: ignore
from dotenv import load_dotenv # type: ignore
import os
from sqlalchemy import create_engine # type: ignore

def get_fbs_teams_by_year(year, week=None):
    """
    Fetch FBS teams for a given year from the College Football Data API.
    Args:
        year (int): Year of the season
        week (int, optional): Week number. Defaults to None.
    Returns:
        pd.DataFrame: DataFrame containing team information
    """
    API_KEY = os.getenv("API_KEY")
    BASE_URL = "https://api.collegefootballdata.com"
    url = f"{BASE_URL}/teams/fbs?year={year}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    teams = response.json()
    data = []
    for team in teams:
        location = team.get("location", {})
        data.append({
            'id': team.get('id'),
            'season': year,
            'school': team.get('school'),
            'mascot': team.get('mascot'),
            'abbreviation': team.get('abbreviation'),
            'alternateNames': team.get('alternateNames', []),
            'conference': team.get('conference'),
            'division': team.get('division'),
            'classification': team.get('classification'),
            'color': team.get('color'),
            'alternateColor': team.get('alternateColor'),
            'logos': team.get('logos', []),
            'twitter': team.get('twitter'),
            'stadium_id': location.get('id'),
            'name': location.get('name'),
            'city': location.get('city'),
            'state': location.get('state'),
            'zip': location.get('zip'),
            'countryCode': location.get('countryCode'),
            'timezone': location.get('timezone'),
            'latitude': location.get('latitude'),
            'longitude': location.get('longitude'),
            'elevation': location.get('elevation'),
            'capacity': location.get('capacity'),
            'constructionYear': location.get('constructionYear'),
            'grass': location.get('grass'),
            'dome': location.get('dome')
        })
    df = pd.DataFrame(data)
    df['alternateNames'] = df['alternateNames'].apply(lambda x: x if isinstance(x, list) else [])
    df['logos'] = df['logos'].apply(lambda x: x if isinstance(x, list) else [])
    df['grass'] = df['grass'].apply(lambda x: bool(x) if pd.notnull(x) else None)
    df['dome'] = df['dome'].apply(lambda x: bool(x) if pd.notnull(x) else None)
    return df

def load_teams_to_db(year, week=None):
    """
    Load FBS teams for a given year into the database.
    Args:
        year (int): Year of the season
        week (int, optional): Week number. Defaults to None.
    """
    load_dotenv()
    teams_df = get_fbs_teams_by_year(year, week)
    teams_df = teams_df.where(pd.notnull(teams_df), None)
    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    engine = create_engine(db_url)
    teams_df = teams_df.rename(columns={
        'alternateNames': 'alternatenames',
        'alternateColor': 'alternatecolor',
        'countryCode': 'countrycode',
        'constructionYear': 'constructionyear'
    })
    teams_df.to_sql("teams", engine, if_exists="append", index=False)
    engine.dispose()
    print(f"Teams for year {year} week {week} loaded into DB.")