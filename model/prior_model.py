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
    r_min = 0 # minimum team rating
    r_max = 100 # maximum team rating
    r_fcs_min = 5 # mininum FCS rating
    r_fcs_max = 15 # maximum FCS rating
    gamma_margin = 0.05 # small margin penalty coefficient
    gamma_loss = 0.5 # regularization constant for loss rate
    gamma_fcs = 5 # regularization constant for FCS loss

    # Decision Variables
    r = {team: cp.Variable(name = f"r_{team}") for team in teams} # team rating
    r_fcs = cp.Variable(name = "r_fcs") # rating for dummy FCS team

    # FCS Loss Penalty Terms
    fcs_loss_terms = []
    for (team, margin, alpha, _, _) in fcs_losses:
        fcs_loss_terms.append(cp.pos(r[team] + (margin * alpha) - r_fcs)**2 * gamma_fcs)
    fcs_loss_penalty = cp.sum(fcs_loss_terms)

    # Soft Margin Penalty Terms
    soft_margin_terms = []
    for (i, j, k, winner, margin, alpha, _, _) in games:
        if winner == i:
            winner_team = i
            loser_team = j
        else:
            winner_team = j
            loser_team = i
        soft_margin_terms.append(cp.pos(r[loser_team] + (margin * alpha) - r[winner_team])**2 * gamma_margin)
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
    for (team) in teams:
        constraints.append(r[team] >= 0)
    constraints.append(r_fcs >= r_fcs_min)
    constraints.append(r_fcs <= r_fcs_max)

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