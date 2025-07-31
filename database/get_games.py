import requests
import pandas as pd # type: ignore
from dotenv import load_dotenv # type: ignore
import os

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

    return pd.DataFrame(response.json())