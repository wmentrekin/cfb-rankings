"""The model's tunable constants, in one place.

The defaults ARE the production model: `ModelParams()` reproduces exactly what the weekly run
publishes, and nothing in the pipeline passes anything else. The fields exist so that a
candidate retune can be solved side by side with production (scripts/compare_margin_scale.py)
without editing the model, and so the weekly model and the prior model cannot drift apart.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelParams:
    # Lower bound on a team's rating inside the solve. None = no bound, which is production.
    # A bound of 0.01 was the model until the 2026 retune: about a quarter of FBS sat on it
    # every week, all tied, in no meaningful order (a loss asks the loser to sit a set gap
    # below the winner, and a team that lost to low-rated teams "wants" to be below zero).
    r_min: Optional[float] = None
    # What the lowest team's PUBLISHED rating is shifted to. The solve is anchored at zero by
    # the loss-rate penalty, so its raw ratings run negative at the bottom; every rating is
    # moved up by the same amount so the lowest reads `display_min`. A uniform shift changes no
    # gap and no rank. None = publish raw. The prior model's output is never shifted: it is fed
    # back into the next season's solve, which works on the raw scale.
    display_min: Optional[float] = 0.01
    r_max: float = 100.0         # maximum team rating
    r_fcs_min: float = 0.01      # minimum rating of the stand-in FCS team
    r_fcs_max: float = 15.0      # maximum rating of the stand-in FCS team
    gamma_loss: float = 0.5      # weight of the loss-rate penalty
    gamma_fcs: float = 5.0       # weight of the FCS-loss penalty
    gamma_margin: float = 0.05   # weight of the margin penalty
    target_gap: float = 7.0      # rating points a median-margin win should demand
    max_gap: float = 20.0        # most rating points any one margin can demand
    # False: a result only penalises a winner that is not far ENOUGH above the loser (a hinge).
    # True: it also penalises one that is too far above, pinning the pair to the demanded gap.
    two_sided_margin: bool = False


DEFAULT_PARAMS = ModelParams()

# The model as it was published before the 2026 retune (apart from the margin scale, which
# scripts/compare_margin_scale.py swaps separately): a hard floor and no shift.
LEGACY_FLOOR_PARAMS = ModelParams(r_min=0.01, display_min=None)


def shift_to_display(ratings: dict, params: ModelParams) -> dict:
    """Move every rating by one constant so the lowest TEAM reads params.display_min.

    The stand-in FCS team ("fcs") moves with the rest but is not what sets the shift.
    """
    if params.display_min is None:
        return ratings
    team_values = [value for name, value in ratings.items() if name != "fcs"]
    if not team_values:
        return ratings
    offset = params.display_min - min(team_values)
    return {name: value + offset for name, value in ratings.items()}
