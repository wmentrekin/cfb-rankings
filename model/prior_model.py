import cvxpy as cp # type: ignore
import networkx as nx # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore

from model.process_data import get_data_by_year_up_to_week

def get_prior_ratings(year):
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
    M = 200 # Big M 200 > 138 = Number of FBS teams
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
    objective = cp.Minimize(cp.sum(slack_terms) + soft_margin_penalty + fcs_slack)

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
    problem.solve(verbose = False)
    if problem.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
        print("Optimal value: %s" % problem.value)
        prior_ratings = {}
        slack = []
        for variable in problem.variables():
            if variable.name()[0:2] == "r_":
                prior_ratings[variable.name()[2:]] = variable.value.astype(float)
            elif variable.name()[0:2] == "z_":
                slack.append((variable.name()[2:], variable.value.astype(float)))    
        prior_ratings = dict(sorted(prior_ratings.items(), key = lambda item: item[1], reverse = True))

    return prior_ratings