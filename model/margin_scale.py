"""The margin scale shared by the weekly model and the prior model.

Both models turn a game's margin of victory into the rating gap it "demands":

    demanded gap = min(k_margin * sqrt(margin), cap)
    k_margin     = target_gap / sqrt(median margin of the games being rated)

so that a game decided by the median margin demands exactly `target_gap` rating points. That is
the formula the methodology page documents.

WHY THIS IS ITS OWN MODULE: each model used to compute the median inline, and both copies read
the wrong field. A game tuple is (i, j, k, winner, margin, alpha, week, season) -- see
model/process_data.py -- and the inline code took position 5 (alpha, the 0.8 / 1.0 / 1.2
home-away multiplier) where it meant position 4 (margin). The "median margin" was therefore
always about 1, k_margin about 7, and the cap was reached by any game decided by more than
roughly 8 points: beyond that, margin of victory added nothing. One helper, named indices and a
test on the tuple shape keep the two models from drifting apart or regressing.
"""
import numpy as np  # type: ignore

# Positions in a game tuple built by model.process_data.process_game_data.
GAME_MARGIN_INDEX = 4
GAME_ALPHA_INDEX = 5

# Used when there are no games to take a median over.
FALLBACK_MEDIAN_MARGIN = 1.0
MIN_SQRT_MEDIAN = 1e-6


def median_margin(games) -> float:
    """Median margin of victory over `games`; FALLBACK_MEDIAN_MARGIN when there are none.

    Missing (None / NaN) and negative margins are left out, as the inline code intended.
    """
    margins = []
    for game in games:
        margin = game[GAME_MARGIN_INDEX]
        if margin is None:
            continue
        margin = float(margin)
        if margin >= 0:  # False for NaN as well
            margins.append(margin)
    if not margins:
        return FALLBACK_MEDIAN_MARGIN
    return float(np.median(margins))


def compute_k_margin(games, target_gap: float) -> float:
    """Rating points per sqrt(point of margin), scaled so the median game demands `target_gap`."""
    return target_gap / max(float(np.sqrt(median_margin(games))), MIN_SQRT_MEDIAN)
