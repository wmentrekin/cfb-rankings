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
            prior_ratings[team] = 35

    # Parameters
    _lambda = cp.Parameter(nonneg=True) # calculate term based on lambda_max and connectivity matrix
    if week:
        print(f"Connectivity: {connectivity}")
        _lambda.value = (7 - week) / 700
        print(f"Using week-based lambda: {_lambda.value}")
    else:
        _lambda.value = 0

    M = 200 # Big M 200 > 138 = Number of FBS teams
    mu = 20 # regularization penalty for on FCS rating
    beta = 2.0 # penalty multipler for FCS loss slack
    R_min = 5 # mininum FCS rating
    R_max = 15 # maximum FCS rating
    gamma = 1 # small regularization constant
    nu = 500 # large regularization constant

    # Decision Variables
    r = {team: cp.Variable(name = f"r_{team}") for team in teams} # team rating
    z = {(i, j, k): cp.Variable(nonneg=True, name = f"z_{i},{j}^{k}") for (i, j, k, _, _, _, _, _) in games} # slack variable
    r_fcs = cp.Variable(name = "r_fcs") # rating for dummy FCS team
    z_fcs = {team: cp.Variable(nonneg=True, name = "z_fcs_team") for (team, _, _, _, _) in fcs_losses} # slack variable for loss to dummy FCS team

    # Prior Rating Terms
    prior_term = _lambda * cp.sum([(r[i] - prior_ratings[i])**2 for i in teams])

    # Slack Terms
    slack_terms = []
    slack_term_infos = []
    for (i, j, k, winner, margin, alpha, _, _) in games:
        if winner == i:
            winner_team = i
            loser_team = j
            factor = nu * margin * alpha
            z_var = z[(i, j, k)]
        else:
            winner_team = j
            loser_team = i
            alpha_safe = alpha if (alpha is not None and alpha != 0) else 1.0
            factor = nu * margin * (1.0 / alpha_safe)
            z_var = z[(i, j, k)]

        slack_terms.append(factor * z_var)
        slack_term_infos.append(((i, j, k), winner_team, loser_team, margin, alpha, factor, z_var))

    # FCS Slack Terms
    fcs_slack = cp.sum([beta * z_fcs[team] for (team, _, _, _, _) in fcs_losses])

    # FCS Rating Regularization
    fcs_reg = mu * (r_fcs - R_min)**2

    # Soft Margin Penalty
    soft_margin_terms = []
    for (i, j, k, winner, margin, alpha, _, _) in games:
        if winner == i:
            winner_team = i
            loser_team = j
        else:
            winner_team = j
            loser_team = i
        soft_margin_terms.append(cp.pos(r[loser_team] + margin - r[winner_team])**2 * gamma)
    soft_margin_penalty = cp.sum(soft_margin_terms)

    # Objective Function
    objective = cp.Minimize(cp.sum(slack_terms) + soft_margin_penalty + fcs_slack + fcs_reg + prior_term)

    # Constraints
    constraints = []
    for (i, j, k, winner, _, _, _, _) in games:
        if winner == i:
            winner_team = i
            loser_team = j
        else:
            winner_team = j
            loser_team = i
        constraints.append(r[loser_team] + z[(i, j, k)] <= r[winner_team] + M) # slack constraints for losses to lower ranked teams
    for (team, _, _, _, _) in fcs_losses:
        constraints.append(r[team] + z_fcs[team] <= r_fcs + M) # slack constraints for losses to FCS teams
    for (team) in teams:
        constraints.append(r[team] >= 0)
    constraints.append(r_fcs >= R_min)
    constraints.append(r_fcs <= R_max)

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose = True)
    if problem.status not in ["infeasible", "unbounded"]:
        print("Optimal value: %s" % problem.value)
        ratings = {}
        slack = []
        for variable in problem.variables():
            name = variable.name()
            if name.startswith("r_"):
                team_name = name[2:]
                try:
                    ratings[team_name] = float(variable.value)
                except Exception:
                    ratings[team_name] = float(np.asarray(variable.value).item())
            elif name.startswith("z_"):
                try:
                    val = float(variable.value)
                except Exception:
                    val = float(np.asarray(variable.value).item())
                slack.append((name, val))
        ratings = dict(sorted(ratings.items(), key = lambda item: item[1], reverse = True))
        for s in slack:
            print(f"Slack {s[0]}: {s[1]}")
        violations = []
        tol = 1e-6
        for (i, j, k, winner, margin, alpha, _, _) in games:
            if winner == i:
                r_w = r[i].value
                r_l = r[j].value
            else:
                r_w = r[j].value
                r_l = r[i].value
            if r_w is None or r_l is None:
                continue
            if (r_w - r_l) < -tol:
                violations.append(((i, j, k), r_w, r_l))
        print("Num strict violations (winner below loser):", len(violations))
        slack_nonzero = [(idx, zvar.value) for idx, zvar in z.items() if zvar.value is not None and abs(zvar.value) > 1e-6]
        print("Number of slacks > 1e-6:", len(slack_nonzero))
        try:
            prior_val = prior_term.value
        except Exception:
            prior_val = None
        slack_obj_sum = 0.0
        for ((i, j, k), winner_team, loser_team, margin, alpha, factor, zvar) in slack_term_infos:
            zv = zvar.value
            if zv is None:
                continue
            slack_obj_sum += factor * float(zv)
        try:
            soft_margin_val = float(soft_margin_penalty.value)
        except Exception:
            soft_margin_val = None

        print("Prior term:", prior_val)
        print("Slack objective (weighted sum):", slack_obj_sum)
        print("Soft margin penalty:", soft_margin_val)

        return ratings, records

    else:
        raise ValueError(f"Problem status: {problem.status}")