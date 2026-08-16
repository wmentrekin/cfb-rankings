import logging

import pandas as pd  # type: ignore
from typing import Optional
from datetime import datetime, timezone

# Reuse the same logger name main.py configures via utils.setup_logging, so warnings from this
# module surface through the pipeline's existing stdout/file handlers when run via main.py, and
# still fall back to Python's default stderr handler when this module is used standalone.
logger = logging.getLogger("cfb_lp")

# Maps teams.conference raw values (as stored in Postgres, sourced from CFBD) to the standardized
# display strings the rankings artifact should expose. Covers the 11 raw values confirmed present
# in teams.conference for season=2026 via direct DB query. An unmapped raw value (e.g. from future
# conference realignment) is passed through unchanged rather than raising -- see build_payload().
CONFERENCE_DISPLAY_NAMES = {
    "Big Ten": "BIG 10",
    "Big 12": "BIG 12",
    "ACC": "ACC",
    "SEC": "SEC",
    "American Athletic": "American",
    "Pac-12": "PAC 12",
    "Mountain West": "MWC",
    "Mid-American": "MAC",
    "Sun Belt": "SBC",
    "Conference USA": "CUSA",
    "FBS Independents": "FBS Independent",
}


def get_ratings_with_conference(engine, year, week) -> pd.DataFrame:
    """
    Fetch ratings for a given season/week joined with the teams table to add conference.
    Args:
        engine: SQLAlchemy engine to query with.
        year (int): Season year.
        week (int): Week number.
    Returns:
        pd.DataFrame: Columns team, rating, wins, losses, ties, season, week, conference, logos.
    """
    query = (
        "SELECT r.team, r.rating, r.wins, r.losses, r.ties, r.season, r.week, t.conference, t.logos "
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


def _display_conference(raw_conference) -> str:
    """
    Map a raw teams.conference value to its standardized display string via
    CONFERENCE_DISPLAY_NAMES. Unmapped values are logged and passed through unchanged.
    Args:
        raw_conference: Raw conference value from the teams table (str, or None/NaN).
    Returns:
        str: The standardized display string, or the raw value unchanged if unmapped.
    """
    if raw_conference not in CONFERENCE_DISPLAY_NAMES:
        logger.warning("Unmapped conference value %r; passing through unchanged.", raw_conference)
        return raw_conference
    return CONFERENCE_DISPLAY_NAMES[raw_conference]


def _resolve_logo(logos) -> Optional[str]:
    """
    Resolve the dark-background-optimized 32px logo URL from a team's logos array by substring
    match, not positional indexing, so this stays correct if CFBD ever reorders the array.
    Args:
        logos: The teams.logos value -- a list of URL strings, or None/NaN/empty.
    Returns:
        Optional[str]: The matching URL, or None if logos is missing/empty or no URL matches.
    """
    if logos is None or (isinstance(logos, float) and pd.isnull(logos)):
        return None
    for url in logos:
        if url and "/logos-dark/32/" in url:
            return url
    return None


def build_payload(year, week, ranked_df: pd.DataFrame) -> dict:
    """
    Build the JSON-serializable rankings artifact payload.
    Args:
        year (int): Season year.
        week (int): Week number.
        ranked_df (pd.DataFrame): Output of compute_rank_and_delta, must include rank/team/
                                   conference/logos/wins/losses/ties/rating/delta columns.
    Returns:
        dict: {season, week, generated_at_utc,
               rankings: [{rank, team, conference, logo, record, rating, delta}]}
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
            "conference": _display_conference(row["conference"]),
            "logo": _resolve_logo(row.get("logos")),
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
