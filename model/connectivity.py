
import networkx as nx # type: ignore

def compute_connectivity_index(games, teams):
    '''
    Compute the connectivity index C / |T| where:
    - games is a list of tuples (team1, team2)
    - teams is a list of all FBS team names (no duplicates)
    '''
    teams = list(set(teams))
    team_set = set(teams)

    G = nx.Graph()
    G.add_nodes_from(teams)

    for i, j in games:
        if i in team_set and j in team_set:  # Only connect FBS <-> FBS
            G.add_edge(i, j)

    largest_cc = max(nx.connected_components(G), key=len)

    all_nodes = set(G.nodes)
    extra_nodes = all_nodes - team_set
    if extra_nodes:
        print("Non-FBS teams in graph:", extra_nodes)

    return len(largest_cc) / len(teams)
