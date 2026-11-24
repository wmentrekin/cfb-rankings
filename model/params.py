"""The model's tunable constants, in one place.

The defaults ARE the production model: `ModelParams()` reproduces exactly what the weekly run
publishes, and nothing in the pipeline passes anything else. The fields exist so that a
candidate retune can be solved side by side with production (scripts/compare_margin_scale.py)
without editing the model, and so the weekly model and the prior model cannot drift apart.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    r_min: float = 0.01          # minimum team rating (the "floor")
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
