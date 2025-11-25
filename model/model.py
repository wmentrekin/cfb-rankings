import cvxpy as cp # type: ignore
import networkx as nx # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore

from model.process_data import get_data_by_year_up_to_week
from model.prior_model import get_prior_ratings

def get_ratings(year, week = None):
    """
    Get team ratings for a given year and optional week.
    Args:
        year (int): Year of the season
        week (int, optional): Week number. Defaults to None.
    Returns:
        ratings (dict): Dictionary of team ratings
        records (dict): Dictionary of team records
    Raises:
        ValueError: If the optimization problem is infeasible or unbounded
    Notes:
        - Uses CVXPY for convex optimization
        - Handles FCS losses with a dummy team
        - Regularizes ratings based on prior year ratings
        - Prints diagnostic information about the solution
    """

    # Get Data
    teams, games, fcs_losses, records, connectivity = get_data_by_year_up_to_week(year, week)
    prior_ratings = get_prior_ratings(year - 1)

    # Datasets
    teams = teams  # List of FBS team names
    games = games  # List of tuples: (i, j, k, winner, margin, location_multiplier)
    fcs_losses = fcs_losses  # List of tuples: (team, margin, location_multiplier)
    prior_ratings = prior_ratings  # {team: prior_rating}

    # Assign Prior Rating to New FBS Teams
    for team in teams:
        if team not in prior_ratings.keys():
            prior_ratings[team] = 1

    # Parameters
    r_min = 0.01 # minimum team rating
    r_max = 100 # maximum team rating
    r_fcs_min = 0.01 # mininum FCS rating
    r_fcs_max = 15 # maximum FCS rating
    gamma_loss = 0.5 # regularitzation constant for loss rate
    gamma_fcs = 5 # regularization constant for FCS loss
    gamma_margin = 0.05 # small margin penalty coefficient

    # Prior Regularization Parameter
    _lambda = cp.Parameter(nonneg=True) # calculate term based on lambda_max and connectivity matrix
    if week < 7:
        print(f"Connectivity: {connectivity}")
        _lambda.value = (7 - week) / 700
        print(f"Using week-based lambda: {_lambda.value}")
    else:
        _lambda.value = 0

    # Margin Regularization Parameters
    TARGET_GAP_FOR_MEDIAN = 7.0        # rating points the median margin should represent
    MAX_RATING_GAP_FROM_MARGIN = 20.0  # cap for maximum rating gap any margin can demand
    margins = [abs(m) for (_, _, _, _, _, m, _, _) in games if m is not None and m >= 0]
    if len(margins) == 0:
        median_margin = 1.0
    else:
        median_margin = float(np.median(margins))
    k_margin = TARGET_GAP_FOR_MEDIAN / max(np.sqrt(median_margin), 1e-6)

    # Decision Variables
    r = {team: cp.Variable(name = f"r_{team}") for team in teams} # team rating
    r_fcs = cp.Variable(name = "r_fcs") # rating for dummy FCS team

    # Prior Rating Terms
    prior_term = _lambda * cp.sum([(r[i] - prior_ratings[i])**2 for i in teams])

    # FCS Loss Penalty Terms
    fcs_loss_terms = []
    for (team, margin, alpha, _, _) in fcs_losses:
        fcs_loss_terms.append(cp.pos(r[team] + (margin * alpha) - r_fcs)**2 * gamma_fcs)
    fcs_loss_penalty = cp.sum(fcs_loss_terms)

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
        soft_margin_terms.append(cp.pos(r[loser_team] + (margin_used * alpha) - r[winner_team])**2 * gamma_margin)
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
    objective = cp.Minimize(soft_margin_penalty + loss_rate_penalty + fcs_loss_penalty + prior_term)

    # Constraints
    constraints = []
    for (team) in teams:
        constraints.append(r[team] >= r_min)
        constraints.append(r[team] <= r_max)
    constraints.append(r_fcs >= r_fcs_min)
    constraints.append(r_fcs <= r_fcs_max)

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose = True)

    # Diagnostics
    if problem.status not in ["infeasible", "unbounded"]:
        print("Optimal value: %s" % problem.value)
        ratings = {}
        for variable in problem.variables():
            name = variable.name()
            if name.startswith("r_"):
                team_name = name[2:]
                try:
                    ratings[team_name] = float(variable.value)
                except Exception:
                    ratings[team_name] = float(np.asarray(variable.value).item())
        ratings = dict(sorted(ratings.items(), key = lambda item: item[1], reverse = True))
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
        print("Prior term:", prior_term.value)
        print("Margin penalty:", soft_margin_penalty.value)
        print("FCS Margin penalty:", fcs_loss_penalty.value)
        print("Loss rate penalty:", loss_rate_penalty.value)

        return ratings, records

    else:
        raise ValueError(f"Problem status: {problem.status}")