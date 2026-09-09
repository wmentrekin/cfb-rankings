"""Pure scalar derivations for a single game's winner, margin, and alpha.

Extracted out of get_games.py's ingest transform so the override path
(database/game_overrides.py) can produce byte-identical winner/margin/alpha
values without duplicating -- and risking drift from -- the ingest logic.
"""


def compute_margin(home_score, away_score) -> int:
    """Absolute point differential between the two teams."""
    return abs(home_score - away_score)


def compute_winner(home_team, away_team, home_score, away_score):
    """The winning team's name, or None if the game hasn't been played yet.

    CFBD stores an unplayed/future game as home_score=away_score=0, not NULL
    (confirmed repeatedly elsewhere in this pipeline, e.g. schedule_grid's
    status derivation). Without this guard, `0 > 0` is False, so every
    unplayed game silently declared the AWAY team the "winner" -- a real bug
    that corrupted both displayed records and the rating model's game
    inputs whenever this ran mid-week with future games still in the query
    window (confirmed live: 2026 week 1, 2026-09-06 -- Notre Dame, Ole Miss,
    Washington, and Florida State each false-recorded as losers of games
    that hadn't been played yet). A genuine 0-0 FINAL is not realistic in
    modern FBS/FCS football (mandatory overtime since 1996), so treating a
    true 0-0 as "not yet played" (winner=None) rather than a real result is
    the safe reading -- model/process_data.py is responsible for skipping
    these rows entirely rather than miscounting them as a decided game.

    Returns:
        str | None
    """
    if home_score == 0 and away_score == 0:
        return None
    return home_team if home_score > away_score else away_team


def compute_alpha(home_team, winner, neutral_site) -> float:
    """Home-field weighting factor used by the rating model."""
    return 1 if neutral_site else (0.8 if home_team == winner else 1.2)
