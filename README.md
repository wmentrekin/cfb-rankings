# **College Football Ranking Optimization**

## **What This Project Does**

This project builds an automated, data-driven **college football team ranking system** using convex quadratic programming (QP).  
Each week, the model ingests updated game results, optimizes team “rating” values that best explain on-field outcomes, and pushes the computed rankings to a cloud database for storage and publication.

The model is designed to:
- Rank all **FBS teams** based on results from the season up to the current week.  
- Account for **margin of victory, game location, and opponent strength.**  
- Handle **FCS losses** with a dummy “FCS” rating variable.  
- Smoothly transition from **prior-season ratings** to **data-driven current-season rankings** as the season progresses.  
- Run automatically via a **GitHub Action** each Sunday morning, pulling new data, solving the model, and updating results in Supabase.

---

## **Quick Architecture Summary**

| Component | Purpose | Technology |
|------------|----------|-------------|
| **Data Source** | [College Football Data API](https://collegefootballdata.com/) | `requests` → `Supabase` |
| **Database** | Stores team, game, and model results tables | **[Supabase](https://supabase.com/) PostgreSQL** |
| **Data Processing** | Loads teams/games and prepares features | `process_data.py`, `get_games.py`, `get_teams.py` |
| **Model** | Solves convex QP for team ratings | `cvxpy`, `numpy`, `pandas` |
| **Automation** | Weekly GitHub Action scheduled via cron (Sundays 3AM ET) | `.github/workflows/weekly_update.yml` |
| **Frontend (planned)** | Public rankings display | `Streamlit` (to be added) |

**Data Flow Summary:**
1. GitHub Action triggers `main.py` weekly.  
2. New games are pulled from the API and inserted into Supabase.  
3. The model runs using all games up to the current week.  
4. Ratings are written back to the database.  
5. Logs are saved for diagnostics.

---

## **Model**

The model assigns each FBS team \( i \) a continuous rating \( r_i \), and a dummy FCS rating \( r_{\text{FCS}} \).  
It minimizes total “ranking inconsistency” subject to logical constraints about game results and prior-season expectations.

### **Decision Variables**
\[
\begin{aligned}
r_i &\ge 0 &&\text{rating of team } i \\
r_{\text{FCS}} &\in [R_{\min}, R_{\max}] &&\text{rating of dummy FCS team} \\
z_{ijk} &\ge 0 &&\text{slack variable for game } k \text{ between teams } i,j \\
z^{\text{FCS}}_i &\ge 0 &&\text{slack variable for FCS losses}
\end{aligned}
\]

### **Parameters**
| Symbol | Meaning | Typical Value |
|---------|----------|----------------|
| \( M \) | Big-M for slack constraints | 200 |
| \( \mu \) | FCS regularization penalty | 20 |
| \( \beta \) | Penalty multiplier for FCS loss slack | 2.0 |
| \( R_{\min}, R_{\max} \) | Bounds on FCS rating | 5, 15 |
| \( \gamma \) | Soft margin penalty weight | 0.01 |
| \( \nu \) | Loss slack penalty | 500 |
| \( \lambda \) | Prior regularization weight | \((7 - \text{week}) / 700\) |

### **Objective Function**

The model minimizes a composite loss:
\[
\min_{r, r_{\text{FCS}}, z} 
\underbrace{\sum_{(i,j,k) \in G} \nu \, \text{margin}_{ijk} \, \alpha_{ijk} \, z_{ijk}}_{\text{Loss slack penalty}}
+ \underbrace{\gamma \sum_{(i,j,k) \in G_w} \max(0, r_j + \text{margin}_{ijk} - r_i)^2}_{\text{Soft margin penalty}}
+ \underbrace{\beta \sum_{i \in FCS} z^{\text{FCS}}_i}_{\text{FCS loss slack}}
+ \underbrace{\mu (r_{\text{FCS}} - R_{\min})^2}_{\text{FCS regularization}}
+ \underbrace{\lambda \sum_i (r_i - r_i^{\text{prior}})^2}_{\text{Prior smoothness term}}
\]

Where:
- \( G \) is the set of all games,
- \( G_w \subset G \) is the subset of games where the winner is known,
- \( \alpha_{ijk} \) is the location multiplier,
- \( \text{margin}_{ijk} \) is the final score differential.

### **Constraints**
\[
\begin{aligned}
r_i + z_{ijk} &\le r_j + M &&\forall (i,j,k) \in G \\
r_i + z^{\text{FCS}}_i &\le r_{\text{FCS}} + M &&\forall i \text{ with FCS losses} \\
r_i &\ge 0 &&\forall i \\
R_{\min} \le r_{\text{FCS}} &\le R_{\max}
\end{aligned}
\]

### **Interpretation**
- **Slack variables** \( z \) absorb violations where a team is rated below an opponent it lost to.  
- **Soft margin penalties** gently reward greater separation between winners and losers.  
- **FCS terms** keep lower-division results from overwhelming the model.  
- **Prior regularization** anchors early-season ratings to the previous season, fading linearly through Week 7.  

The optimization is solved using [**CVXPY**](https://www.cvxpy.org/) with default convex solvers, returning team ratings sorted in descending order.
