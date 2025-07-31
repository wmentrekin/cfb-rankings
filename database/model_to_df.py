import pandas as pd
from typing import Dict, List, Tuple, Any

def ratings_to_df(ratings: Dict[str, float], records: Dict[str, Tuple[int, int]]) -> pd.DataFrame:
    """
    Convert ratings and records to a DataFrame for database upload.
    Args:
        ratings: Dict of team name to rating.
        records: Dict of team name to (wins, losses).
    Returns:
        DataFrame with columns: team, rating, wins, losses, record
    """
    data = []
    for team, rating in ratings.items():
        wins, losses = records.get(team, (None, None))
        record_str = f"{wins}-{losses}" if wins is not None and losses is not None else None
        data.append({
            'team': team,
            'rating': rating,
            'wins': wins,
            'losses': losses,
            'record': record_str
        })
    return pd.DataFrame(data)

def fcs_losses_to_df(fcs_losses: List[Tuple[Any, ...]]) -> pd.DataFrame:
    """
    Convert FCS losses list to a DataFrame for database upload.
    Args:
        fcs_losses: List of tuples (team, margin, location_multiplier, ...)
    Returns:
        DataFrame with columns: team, margin, location_multiplier
    """
    columns = ['team', 'margin', 'location_multiplier']
    data = [dict(zip(columns, f[:3])) for f in fcs_losses]
    return pd.DataFrame(data)

# Example usage (to be replaced with actual model output):
# ratings = {'Alabama': 95.2, 'Georgia': 92.1}
# records = {'Alabama': (12, 1), 'Georgia': (11, 2)}
# games = [('Alabama', 'Georgia', 1, 'Alabama', 7, 1.0, ...)]
# fcs_losses = [('Alabama', 3, 1.0, ...)]
# df_ratings = ratings_to_df(ratings, records)
# df_games = games_to_df(games)
# df_fcs = fcs_losses_to_df(fcs_losses)
