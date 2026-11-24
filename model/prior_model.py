import cvxpy as cp # type: ignore
import networkx as nx # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore

from model.margin_scale import compute_k_margin
from model.params import DEFAULT_PARAMS
from model.process_data import get_data_by_year_up_to_week

def get_prior_ratings(year, params = DEFAULT_PARAMS):
    """
    Get prior team ratings for a given year using all games from that year.
    Args:
        year (int): Year of the season
    Returns:
        prior_ratings (dict): Dictionary of team ratings
    Raises:
        ValueError: If the optimization problem is infeasible or unbounded
    Notes:
        - Uses CVXPY for convex optimization
        - Handles FCS losses with a dummy team
        - Regularizes ratings to be non-negative and within bounds for FCS team
        - Prints diagnostic information about the solution
    """

    # Get Data
    teams, games, fcs_losses, records, connectivity = get_data_by_year_up_to_week(year)

    # Datasets
    teams = teams  # List of FBS team names
    games = games  # List of tuples: (i, j, k, winner, margin, location_multiplier, week, season)
    fcs_losses = fcs_losses  # List of tuples: (team, margin, location_multiplier, week, season)

    # Parameters
    # Values and their meanings live in model/params.py; the defaults are the production model.
    r_min, r_max = params.r_min, params.r_max
    r_fcs_min, r_fcs_max = params.r_fcs_min, params.r_fcs_max
    gamma_loss, gamma_fcs, gamma_margin = params.gamma_loss, params.gamma_fcs, params.gamma_margin

    # Margin Regularization Parameters
    TARGET_GAP_FOR_MEDIAN = params.target_gap      # rating points the median margin should represent
    MAX_RATING_GAP_FROM_MARGIN = params.max_gap    # cap for maximum rating gap any margin can demand
    k_margin = compute_k_margin(games, TARGET_GAP_FOR_MEDIAN)

    # Decision Variables
    r = {team: cp.Variable(name = f"r_{team}") for team in teams} # team rating
    r_fcs = cp.Variable(name = "r_fcs") # rating for dummy FCS team

    # FCS Loss Penalty Terms
    fcs_loss_terms = []
    for (team, margin, alpha, _, _) in fcs_losses:
        fcs_loss_terms.append(cp.pos(r[team] + (margin * alpha) - r_fcs)**2 * gamma_fcs)
    # cp.sum([]) is a plain 0 with no .value, which crashed the diagnostics below in a season
    # with no FCS losses. Same guard as model/model.py.
    fcs_loss_penalty = cp.sum(fcs_loss_terms) if fcs_loss_terms else cp.Constant(0.0)

    # Soft Margin Penalty Terms
    soft_margin_terms = []
    for (i, j, k, winner, margin, alpha, _, _) in games:
        margin_val = max(0.0, float(margin))
        margin_scaled = k_margin * np.sqrt(margin_val)
        margin_used = min(margin_scaled, MAX_RATING_GAP_FROM_MARGIN)
        if winner == i:
            winner_team = i
            loser_team = j
        else:
            winner_team = j
            loser_team = i
        shortfall = r[loser_team] + (margin_used * alpha) - r[winner_team]
        if not params.two_sided_margin:
            shortfall = cp.pos(shortfall)
        soft_margin_terms.append(shortfall**2 * gamma_margin)
    soft_margin_penalty = cp.sum(soft_margin_terms)

    # Loss Rate Penalty Terms
    loss_rate_terms = []
    for team, record in records.items():
        wins = record[0]
        losses = record[1]
        games_played = wins + losses
        if games_played > 0:
            loss_rate = losses / games_played
            loss_rate_terms.append((r[team] * loss_rate)**2 * gamma_loss)
    loss_rate_penalty = cp.sum(loss_rate_terms)

    # Objective Function
    objective = cp.Minimize(soft_margin_penalty + loss_rate_penalty + fcs_loss_penalty)

    # Constraints
    constraints = []
    for (team) in teams:
        constraints.append(r[team] >= r_min)
        constraints.append(r[team] <= r_max)
    constraints.append(r_fcs >= r_fcs_min)
    constraints.append(r_fcs <= r_fcs_max)
    # Redundant while r_min is positive (the production default); skipped for a candidate
    # that lets ratings go below zero, where it would silently put the floor back.
    if r_min >= 0:
        for (team) in teams:
            constraints.append(r[team] >= 0)

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose = False)
    if problem.status not in ["infeasible", "unbounded"]:
        print("Optimal value: %s" % problem.value)
        prior_ratings = {}
        for variable in problem.variables():
            name = variable.name()
            if name.startswith("r_"):
                team_name = name[2:]
                try:
                    prior_ratings[team_name] = float(variable.value)
                except Exception:
                    prior_ratings[team_name] = float(np.asarray(variable.value).item())
        prior_ratings = dict(sorted(prior_ratings.items(), key = lambda item: item[1], reverse = True))
        
        violations = []
        for (i, j, k, winner, margin, alpha, _, _) in games:
            if winner == i:
                r_w = r[i].value
                r_l = r[j].value
            else:
                r_w = r[j].value
                r_l = r[i].value
            if r_w is None or r_l is None:
                continue
            if (r_w - r_l) < 0:
                violations.append(((i, j, k), r_w, r_l))
        print("Num strict violations (winner below loser):", len(violations))
        print("Margin penalty:", soft_margin_penalty.value)
        print("Loss rate penalty:", loss_rate_penalty.value)
        print("FCS margin penalty:", fcs_loss_penalty.value)

    return prior_ratings