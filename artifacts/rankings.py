import pandas as pd  # type: ignore
from typing import Optional
from datetime import datetime, timezone


def get_ratings_with_conference(engine, year, week) -> pd.DataFrame:
    """
    Fetch ratings for a given season/week joined with the teams table to add conference.
    Args:
        engine: SQLAlchemy engine to query with.
        year (int): Season year.
        week (int): Week number.
    Returns:
        pd.DataFrame: Columns team, rating, wins, losses, ties, season, week, conference.
    """
    query = (
        "SELECT r.team, r.rating, r.wins, r.losses, r.ties, r.season, r.week, t.conference "
        "FROM ratings r "
        "LEFT JOIN teams t ON r.team = t.school AND r.season = t.season "
        f"WHERE r.season = {year} AND r.week = {week};"
    )
    df = pd.read_sql_query(query, engine)
    return df


def previous_week_with_data(engine, year, week) -> Optional[int]:
    """
    Find the most recent week strictly before the given week in the same season that has ratings data.
    Args:
        engine: SQLAlchemy engine to query with.
        year (int): Season year.
        week (int): Current week number.
    Returns:
        Optional[int]: The most recent prior week with ratings data, or None if there isn't one.
    """
    query = f"SELECT MAX(week) AS week FROM ratings WHERE season = {year} AND week < {week};"
    df = pd.read_sql_query(query, engine)
    value = df["week"].iloc[0]
    if pd.isnull(value):
        return None
    return int(value)


def compute_rank_and_delta(current_df: pd.DataFrame, previous_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Add rank and delta columns to current_df.
    Args:
        current_df (pd.DataFrame): Current week's ratings, must include team/rating columns.
        previous_df (Optional[pd.DataFrame]): Previous week's ratings for delta calc, or None/empty.
    Returns:
        pd.DataFrame: current_df with new int columns rank (1 = best, deterministic tie-break by
                      team name ascending) and delta (previous_rank - current_rank; positive means
                      the team moved up; 0 if there's no matching previous-week row).
    """
    df = current_df.sort_values(by=["rating", "team"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = df.index + 1

    prev_rank_map = {}
    if previous_df is not None and not previous_df.empty:
        prev_sorted = previous_df.sort_values(by=["rating", "team"], ascending=[False, True]).reset_index(drop=True)
        prev_rank_map = dict(zip(prev_sorted["team"], prev_sorted.index + 1))

    df["delta"] = df.apply(
        lambda row: int(prev_rank_map[row["team"]]) - int(row["rank"]) if row["team"] in prev_rank_map else 0,
        axis=1,
    )
    df["rank"] = df["rank"].astype(int)
    return df


def build_payload(year, week, ranked_df: pd.DataFrame) -> dict:
    """
    Build the JSON-serializable rankings artifact payload.
    Args:
        year (int): Season year.
        week (int): Week number.
        ranked_df (pd.DataFrame): Output of compute_rank_and_delta, must include rank/team/
                                   conference/wins/losses/ties/rating/delta columns.
    Returns:
        dict: {season, week, generated_at_utc, rankings: [{rank, team, conference, record, rating, delta}]}
    """
    ordered = ranked_df.sort_values("rank", ascending=True)
    rankings = []
    for _, row in ordered.iterrows():
        wins = int(row["wins"])
        losses = int(row["losses"])
        ties = int(row["ties"]) if pd.notnull(row.get("ties")) else 0
        record = f"{wins}-{losses}"
        if ties > 0:
            record += f"-{ties}"
        rankings.append({
            "rank": int(row["rank"]),
            "team": row["team"],
            "conference": row["conference"],
            "record": record,
            "rating": round(float(row["rating"]), 2),
            "delta": int(row["delta"]),
        })
    return {
        "season": year,
        "week": week,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rankings": rankings,
    }
