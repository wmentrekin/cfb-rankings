# **College Football Ranking Optimization**

📊 **[View Rankings App](https://cfb-rankings-wmentrekin.streamlit.app/)**

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
| **Frontend** | Public rankings display | **[Streamlit](https://cfb-rankings-wmentrekin.streamlit.app/)** |

**Data Flow Summary:**
1. GitHub Action triggers `main.py` every Sunday morning.  
2. New games are pulled from the API and inserted into Supabase.  
3. The model runs using all games up to the current week.  
4. Ratings are written back to the database.  
5. Logs are saved for diagnostics.

---