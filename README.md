# **College Football Ranking Optimization**

## **What This Project Does**

This project builds an automated, data-driven **college football team ranking system** using convex quadratic programming (QP). Each week, the model ingests updated game results, optimizes team rating values that best explain on-field outcomes, and pushes the computed rankings to a cloud database for storage and publication.

The model is designed to:
- Rank all **FBS teams** based on results from the season up to the current week.  
- Account for **margin of victory, game location, and opponent strength.**  
- Handle **FCS losses** with a dummy “FCS” rating variable.  
- Smoothly transition from **prior-season ratings** to **current-season rankings** as the season progresses. 

---

## **Architecture Summary**

| Component | Purpose | Technology |
|------------|----------|-------------|
| **Data Source** | `requests` → `Supabase` | [College Football Data API](https://collegefootballdata.com/) |
| **Database** | Stores team, game, and model results tables | **[Supabase](https://supabase.com/) PostgreSQL** |
| **Data Processing** | Loads teams/games and prepares features | `process_data.py`, `get_games.py`, `get_teams.py` |
| **Model** | Solves convex QP for team ratings | `cvxpy`, `numpy`, `pandas` |
| **Automation** | Weekly GitHub Action scheduled via cron (Sundays 3AM ET) | `.github/workflows/weekly_update.yml` |
| **Frontend (planned)** | Public rankings display | **[Streamlit](https://cfb-rankings-wmentrekin.streamlit.app/)** |

**Data Flow Summary:**
1. GitHub Action triggers `main.py` every Sunday morning.  
2. New games are pulled from the API and inserted into Supabase.  
3. The model runs using all games up to the current week.  
4. Ratings are written back to the database.  
5. Logs are saved for diagnostics.

---

## **Model**

The model assigns each FBS team i a continuous rating rᵢ, and a dummy FCS rating r_fcs.  
It minimizes total “ranking inconsistency” subject to logical constraints about game results and prior-season expectations.

### Objective Function
We minimize a weighted combination of slack penalties, prior regularization, and soft margin terms:

**Minimize:**

Σ_{all games} ( ν · margin · α · z_{winner,loser} )  [Slack penalty for ranking violations]
+ Σ_{all games} ( γ · [ max(0, (r_loser + margin − r_winner))² ] )  [Soft margin penalty]
+ Σ_{all FCS losses} ( β · z_fcs_team )  [FCS loss penalty]
+ μ · (r_fcs − R_min)²  [FCS rating regularization]
+ λ · Σ_{all teams} (r_team − prior_team)²  [Prior season regularization]

Where:
- For each game, winner and loser are determined by the actual game outcome
- α adjusts for home/away games (α = 1.0 for neutral site games, 0.8 if home team wins, 1.2 if away team wins)
- margin is the point differential in the game
- z_{winner,loser} is the slack variable when the winner's rating is below the loser's
- λ decays to zero after week 7 to rely fully on current season data

---

### Constraints
For every game `(i, j, k)` where team *i* played team *j* for the *k*th time:

1. **Loss slack constraint:**  
   r_loser + z_{winner,loser} ≤ r_winner + M  
   (If r_loser > r_winner, z_{winner,loser} must absorb the violation)

2. **FCS loss constraint:**  
   r_team + z_fcs_team ≤ r_fcs + M  
   (For teams that lost to FCS opponents)

3. **Rating bounds:**  
   rᵢ ≥ 0   ∀ i ∈ teams  
   R_min ≤ r_fcs ≤ R_max

---

### Decision Variables
- **rᵢ:** continuous rating for each FBS team  
- **r_fcs:** single rating for the dummy FCS team  
- **zᵢⱼᵏ:** nonnegative slack variable representing a violation (team *i* ranked above *j* despite losing)  
- **z_fcsᵢ:** slack for losses to FCS teams

---

### Parameters
| Symbol | Description | Default Value |
|:-------:|:-------------|:---------------:|
| λ | Weight on prior regularization term, decays with week | (7 − week) / 700 before week 7, 0 after |
| μ | Penalty for deviation of FCS rating from baseline | 20 |
| β | Weight for FCS loss slack | 2.0 |
| γ | Penalty on small winning margins (soft margin regularization) | 1.0 |
| ν | Scaling factor for loss slack (margin and home-field adjusted) | 500 |
| M | Big-M constant (should exceed number of teams) | 200 |
| R_min | Lower bound for FCS team rating | 5 |
| R_max | Upper bound for FCS team rating | 15 |

---

### Model Notes
- **Prior ratings** are carried over from the previous season's final model output. New FBS teams receive a default prior rating of 35.
- The **λ** parameter enforces prior-season influence through Week 6, then drops to 0 to rely entirely on current season results.
- **FCS losses** are handled via a dummy FCS team whose rating is constrained between R_min and R_max.
- **Slack variables** (z) capture ranking violations, weighted by margin and location (home/away/neutral).
- **Soft margin penalties** encourage appropriate rating separation based on game margins.
- **Home/away adjustments** use α multipliers to account for game location advantage.
- The optimization is solved using [**CVXPY**](https://www.cvxpy.org/) with default convex solvers, returning team ratings sorted in descending order.