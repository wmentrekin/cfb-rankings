
import cvxpy as cp # type: ignore
import networkx as nx # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore

from model.process_data import get_data_by_year_up_to_week
from model.prior_model import get_prior_ratings
from model.connectivity import lambda_decay

def get_ratings(year, week = None):

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
    lambda_max = 10
    full_trust_week = 7
    _lambda = cp.Parameter(nonneg=True) # calculate term based on lambda_max and connectivity matrix
    if week:
        _lambda.value = lambda_decay(week, connectivity, lambda_max, full_trust_week)
    else:
        _lambda.value = 0
    M = 200 # Big M 200 > 138 = Number of FBS teams
    mu = 20 # regularization penalty for on FCS rating
    beta = 2.0 # penalty multipler for FCS loss slack
    R_min = 5 # mininum FCS rating
    R_max = 15 # maximum FCS rating
    gamma = 1 # small regularization constant

    # Decision Variables
    r = {team: cp.Variable(name = f"r_{team}") for team in teams} # team rating
    z = {(i, j, k): cp.Variable(nonneg=True, name = f"z_{i},{j}^{k}") for (i, j, k, _, _, _, _, _) in games} # slack variable
    r_fcs = cp.Variable(name = "r_fcs") # rating for dummy FCS team
    z_fcs = {team: cp.Variable(nonneg=True, name = "z_fcs_team") for (team, _, _, _, _) in fcs_losses} # slack variable for loss to dummy FCS team

    # Prior Rating Terms
    prior_term = _lambda * cp.sum([(r[i] - prior_ratings[i])**2 for i in teams])

    # Slack Terms
    slack_terms = []
    for (i, j, k, winner, margin, alpha, _, _) in games:
        if winner == j:
            slack_terms.append(margin * alpha * z[(i, j, k)])

    # FCS Slack Terms
    fcs_slack = cp.sum([beta * z_fcs[team] for (team, _, _, _, _) in fcs_losses])

    # FCS Rating Regularization
    fcs_reg = mu * (r_fcs - R_min)**2

    # Soft Margin Penalty
    soft_margin_penalty = cp.sum([
        cp.pos(r[j] + margin - r[i])**2
        for (i, j, k, winner, margin, alpha, _, _) in games
        if winner == i  # i beat j
    ])

    # Objective Function
    objective = cp.Minimize(cp.sum(slack_terms) + gamma * soft_margin_penalty + fcs_slack + fcs_reg + prior_term)

    # Constraints
    constraints = []
    for (i, j, k, winner, _, _, _, _) in games:
        constraints.append(r[i] + z[(i, j, k)] <= r[j] + M) # slack constraints for losses to lower ranked teams
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
        # Otherwise, problem.value is inf or -inf, respectively.
        print("Optimal value: %s" % problem.value)
        ratings = {}
        slack = []
        for variable in problem.variables():
            print("Variable %s: value %s" % (variable.name(), variable.value))
            if variable.name()[0:2] == "r_":
                ratings[variable.name()[2:]] = variable.value.astype(float)
            elif variable.name()[0:2] == "z_":
                slack.append((variable.name()[2:], variable.value.astype(float)))
        ratings = dict(sorted(ratings.items(), key = lambda item: item[1], reverse = True))
        count = 1
        for team in ratings.keys():
            if team == "fcs":
                print(count, team, ratings[team])
            else:
                print(count, team, ratings[team], f"{records[team][0]}-{records[team][1]}")
            count += 1

        return ratings, slack, records