import requests # type: ignore
import pandas as pd # type: ignore
from dotenv import load_dotenv # type: ignore
import os
from sqlalchemy import create_engine, Table, MetaData # type: ignore
from sqlalchemy.dialects.postgresql import insert # type: ignore

# Valid classification values CFBD documents today: fbs, fcs, ii, ii/iii, iii.
# This module keeps everything EXCEPT fbs -- fbs rows belong exclusively to
# `teams` (see database/get_teams.py) and must never land here.
FBS_CLASSIFICATION = "fbs"


def get_non_fbs_teams_by_year(year):
    """
    Fetch non-FBS Division-I teams for a given year from the College Football Data API.

    Calls the plain `/teams` endpoint (NOT `/teams/fbs`), which returns every
    Division-I team CFBD knows about for the season (FBS + FCS + II/III),
    roughly 265 rows for a recent season. `/teams` accepts `year` but exposes
    no `classification` query parameter, so the FBS/non-FBS split is done
    client-side here on each record's `classification` field.

    Args:
        year (int): Year of the season
    Returns:
        pd.DataFrame: DataFrame containing non-FBS team information (school,
            mascot, abbreviation, conference, classification, color,
            alternateColor, logos). Empty DataFrame (with the right columns)
            if the API returns no rows or no non-FBS rows.
    """
    API_KEY = os.getenv("API_KEY")
    BASE_URL = "https://api.collegefootballdata.com"
    url = f"{BASE_URL}/teams"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"year": year}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    teams = response.json()

    columns = [
        'id', 'season', 'school', 'mascot', 'abbreviation', 'conference',
        'classification', 'color', 'alternateColor', 'logos',
    ]

    non_fbs_teams = [
        team for team in (teams or [])
        if (team.get('classification') or '').strip().lower() != FBS_CLASSIFICATION
    ]

    if not non_fbs_teams:
        return pd.DataFrame(columns=columns)

    data = []
    for team in non_fbs_teams:
        data.append({
            'id': team.get('id'),
            'season': year,
            'school': team.get('school'),
            'mascot': team.get('mascot'),
            'abbreviation': team.get('abbreviation'),
            'conference': team.get('conference'),
            'classification': team.get('classification'),
            'color': team.get('color'),
            'alternateColor': team.get('alternateColor'),
            'logos': team.get('logos', []),
        })
    df = pd.DataFrame(data, columns=columns)
    df['logos'] = df['logos'].apply(lambda x: x if isinstance(x, list) else [])
    return df


def load_non_fbs_teams_to_db(year):
    """
    Load non-FBS Division-I teams for a given year into the `non_fbs_teams` table.
    Upserts on (season, school) so re-running a year is idempotent.

    Args:
        year (int): Year of the season
    Returns:
        dict: {'stored': int, 'with_logos': int} -- how many non-FBS team rows
            were upserted, and how many of those carried a non-empty `logos`
            array. It was not possible to confirm from this sandbox (no CFBD
            API key available) whether CFBD actually populates logos for
            FCS/II/III teams, so this is reported loudly rather than assumed.
    """
    load_dotenv()
    teams_df = get_non_fbs_teams_by_year(year)

    if teams_df.empty:
        print(
            f"load_non_fbs_teams_to_db: CFBD returned zero non-FBS Division-I teams "
            f"for year={year}. Nothing to store."
        )
        return {'stored': 0, 'with_logos': 0}

    with_logos = int(teams_df['logos'].apply(lambda x: isinstance(x, list) and len(x) > 0).sum())
    total = len(teams_df)

    teams_df = teams_df.where(pd.notnull(teams_df), None)
    teams_df = teams_df.rename(columns={'alternateColor': 'alternatecolor'})

    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    engine = create_engine(db_url)
    metadata = MetaData()
    table = Table('non_fbs_teams', metadata, autoload_with=engine)

    with engine.begin() as conn:
        for _, row in teams_df.iterrows():
            stmt = insert(table).values(**row.to_dict())
            update_dict = {
                c: getattr(stmt.excluded, c)
                for c in teams_df.columns if c not in ('season', 'school')
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=['season', 'school'],
                set_=update_dict,
            )
            conn.execute(stmt)
    engine.dispose()

    print(f"Non-FBS teams for year {year} loaded into DB (upsert completed): {total} stored.")
    if with_logos == 0:
        print(
            f"WARNING load_non_fbs_teams_to_db: 0 of {total} non-FBS teams for year={year} "
            f"had a non-empty `logos` array from CFBD. Non-FBS opponents will have no logo "
            f"data available even though rows now exist in non_fbs_teams -- this could not "
            f"be verified ahead of time without a CFBD API key."
        )
    else:
        print(
            f"load_non_fbs_teams_to_db: {with_logos} of {total} non-FBS teams for year={year} "
            f"had a non-empty `logos` array."
        )
    return {'stored': total, 'with_logos': with_logos}
