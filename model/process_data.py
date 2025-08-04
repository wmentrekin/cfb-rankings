import pandas as pd # type: ignore
from dotenv import load_dotenv # type: ignore
from connectivity import compute_connectivity_index
import os
from sqlalchemy import create_engine # type: ignore

def process_game_data(games_df, fbs_teams_df):

    teams = fbs_teams_df['school'].tolist()
    games = []
    fcs_losses = []

    game_counter = {}

    game_pairs = []

    records = {}
    for team in teams:
        records[team] = [0, 0, 0]

    for _, row in games_df.iterrows():
        team1, team2 = row['home_team'], row['away_team']
        game_pairs.append((team1, team2))
        team1_score, team2_score = row['home_score'], row['away_score']
        if pd.isnull(team1_score) or pd.isnull(team2_score):
            continue  # Skip games without scores

        i, j = team1, team2
        key = tuple(sorted([i, j]))  # unordered pair
        game_counter[key] = game_counter.get(key, 0)
        k = game_counter[key]
        game_counter[key] += 1

        team1_is_fbs = team1 in teams
        team2_is_fbs = team2 in teams

        margin = abs(team1_score - team2_score)
        if team1_score > team2_score:
            winner = team1
        elif team2_score > team1_score:
            winner = team2
        else:
            winner = "tie"

        location = "neutral" if row.get("neutral_site", False) else "home"
        alpha = 1 if location == "neutral" else (0.8 if row['home_team'] == winner else 1.2)

        if team1_is_fbs and team2_is_fbs and team1 == winner:
            records[team1][0] += 1
            records[team2][1] += 1
        elif team1_is_fbs and team2_is_fbs and team2 == winner:
            records[team1][1] += 1
            records[team2][0] += 1
        elif team1_is_fbs and team2_is_fbs and winner == "tie":
            records[team1][2] += 1
            records[team2][2] += 1
        elif team1_is_fbs and  not team2_is_fbs and team1 == winner:
            records[team1][0] += 1
        elif team1_is_fbs and not team2_is_fbs and team2 == winner:
            records[team1][1] += 1
        elif team1_is_fbs and not team2_is_fbs and winner == "tie":
            records[team1][2] += 1
        elif not team1_is_fbs and team2_is_fbs and team1 == winner:
            records[team2][1] += 1
        elif not team1_is_fbs and team2_is_fbs and team2 == winner:
            records[team2][0] += 1
        elif not team1_is_fbs and team2_is_fbs and winner == "tie":
            records[team2][2] += 1

        # Track FCS losses
        if team1_is_fbs and not team2_is_fbs and team1_score < team2_score:
            fcs_losses.append((team1, margin, alpha, row.get("week"), row.get("season")))
        elif team2_is_fbs and not team1_is_fbs and team2_score < team1_score:
            fcs_losses.append((team2, margin, alpha, row.get("week"), row.get("season")))
        elif team1_is_fbs and team2_is_fbs:
            games.append((
                team1,
                team2,
                k,
                winner,
                margin,
                alpha,
                row.get("week"),
                row.get("season")
            ))

    connectivity = compute_connectivity_index(game_pairs, teams)

    return teams, games, fcs_losses, records, connectivity

def get_games_by_year_week(year, week=None):
    load_dotenv()
    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    engine = create_engine(db_url)
    if week:
        query = f"SELECT * FROM games WHERE season = {year} AND week = {week};"
    else:
        query = f"SELECT * FROM games WHERE season = {year};"
    df = pd.read_sql_query(query, engine)
    engine.dispose()
    return df

def get_fbs_teams_by_year(year):
    load_dotenv()
    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    engine = create_engine(db_url)
    query = f"SELECT * FROM teams WHERE season = {year};"
    df = pd.read_sql_query(query, engine)
    engine.dispose()
    return df

def get_data_by_year_up_to_week(year, week = None):
    games_df = get_games_by_year_week(year, week)
    if week:
        games_df = games_df[games_df["week"] <= week]
    fbs_teams_df = get_fbs_teams_by_year(year)
    teams, games, fcs_losses, records, connectivity = process_game_data(games_df, fbs_teams_df)
    return teams, games, fcs_losses, records, connectivity