# streamlit_app.py
import streamlit as st # type: ignore
from supabase import create_client, Client # type: ignore
import pandas as pd
import numpy as np
import os
import ast
from typing import List, Optional

# ---------------------------
## ---------------------------
# Footer / Documentation
# ---------------------------
st.markdown("---")

st.markdown("""
## Model Details

The model assigns each FBS team $i$ a continuous rating $r_i$, and a dummy FCS rating $r_{fcs}$.  
It minimizes total "ranking inconsistency" subject to logical constraints about game results and prior-season expectations.

### Objective Function
We minimize a weighted combination of slack penalties, prior regularization, and soft margin terms:

Minimize:

$\sum_{\text{games}} \nu \cdot \text{margin} \cdot \alpha \cdot z_{\text{winner,loser}}$ [Slack penalty]

$+ \sum_{\text{games}} \gamma \cdot [\max(0, r_{\text{loser}} + \text{margin} - r_{\text{winner}})]^2$ [Soft margin]

$+ \sum_{\text{FCS losses}} \beta \cdot z_{\text{fcs}}$ [FCS penalty]

$+ \mu \cdot (r_{\text{fcs}} - R_{\text{min}})^2$ [FCS regularization]

$+ \lambda \sum_{\text{teams}} (r_{\text{team}} - \text{prior}_{\text{team}})^2$ [Prior regularization]

Where:
- For each game, winner and loser are determined by the actual game outcome
- $\\alpha$ adjusts for home/away games ($\\alpha = 1.0$ for neutral site games, $0.8$ if home team wins, $1.2$ if away team wins)
- margin is the point differential in the game
- $z_{\\text{winner,loser}}$ is the slack variable when the winner's rating is below the loser's
- $\\lambda$ decays to zero after week 7 to rely fully on current season data

### Constraints
For every game $(i, j, k)$ where team $i$ played team $j$ for the $k$th time:

1. **Loss slack constraint:**  
   $r_{\text{loser}} + z_{\text{winner,loser}} \leq r_{\text{winner}} + M$  
   (If $r_{\text{loser}} > r_{\text{winner}}$, $z_{\text{winner,loser}}$ must absorb the violation)

2. **FCS loss constraint:**  
   $r_{\text{team}} + z_{\text{fcs,team}} \leq r_{\text{fcs}} + M$  
   (For teams that lost to FCS opponents)

3. **Rating bounds:**  
   $r_i \geq 0 \quad \forall i \in \text{teams}$  
   $R_{\text{min}} \leq r_{\text{fcs}} \leq R_{\text{max}}$

### Parameters
| Parameter | Description | Default Value |
|:-------:|:-------------|:---------------:|
| $\\lambda$ | Weight on prior regularization term | $(7 - week) / 700$ before week 7, 0 after |
| $\\mu$ | Penalty for deviation of FCS rating | 20 |
| $\\beta$ | Weight for FCS loss slack | 2.0 |
| $\\gamma$ | Penalty on small winning margins | 1.0 |
| $\\nu$ | Scaling factor for loss slack | 500 |
| $M$ | Big-M constant | 200 |
| $R_{min}$ | Lower bound for FCS rating | 5 |
| $R_{max}$ | Upper bound for FCS rating | 15 |

### Model Notes
- **Prior ratings** are carried over from the previous season's final model output. New FBS teams receive a default prior rating of 35.
- The $\\lambda$ parameter enforces prior-season influence through Week 6, then drops to 0 to rely entirely on current season results.
- **FCS losses** are handled via a dummy FCS team whose rating is constrained between $R_{min}$ and $R_{max}$.
- **Slack variables** ($z$) capture ranking violations, weighted by margin and location (home/away/neutral).
- **Soft margin penalties** encourage appropriate rating separation based on game margins.
- **Home/away adjustments** use $\\alpha$ multipliers to account for game location advantage.
- The optimization is solved using [CVXPY](https://www.cvxpy.org/) with default convex solvers, returning team ratings sorted in descending order.
""") Init
# ---------------------------
st.set_page_config(
    page_title="CFB Rankings",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 College Football Ranking")
st.caption("a convex quadratic program to optimally rank college football teams based on game outcomes")
st.markdown("College Football Data API (https://collegefootballdata.com/)")
st.markdown("Github Repository (https://github.com/wmentrekin/cfb-rankings/tree/main)")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# Utility helpers
# ---------------------------
def safe_parse_logos(value) -> List[str]:
    """
    Parse the logos column which may be stored as:
      - Postgres text[] returning a Python list (already)
      - JSON-like string "['url1','url2']"
      - 'null' or None
    Return list of urls (possibly empty).
    """
    if value is None:
        return []
    # If already a list
    if isinstance(value, list):
        return value
    # If it's a string that looks like a Python list / JSON
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # fallback: if comma-separated
    if isinstance(value, str) and (value.startswith("http") or "," in value):
        parts = [p.strip() for p in value.split(",") if p.strip().startswith("http")]
        return parts
    return []

def logo_img_html(logo_urls: List[str], width: int = 36) -> str:
    """
    Return HTML snippet for the first logo URL found.
    """
    if not logo_urls:
        return ""
    url = logo_urls[0]
    return f'<img src="{url}" alt="logo" style="width:{width}px;height:auto;border-radius:4px;margin-right:8px;vertical-align:middle;">'

def format_delta_cell(delta: int) -> str:
    """
    Return an HTML string representing delta with colors:
      positive (moved up) -> green with up arrow
      negative (moved down) -> red with down arrow
      zero -> yellow dash
    """
    if delta > 0:
        return f'<span style="color:green;font-weight:600;">▲ {delta}</span>'
    elif delta < 0:
        return f'<span style="color:red;font-weight:600;">▼ {abs(delta)}</span>'
    else:
        return '<span style="color:goldenrod;font-weight:600;">—</span>'

# ---------------------------
# DB fetchers (cached)
# ---------------------------
@st.cache_data(ttl=3600)
def get_available_seasons() -> List[int]:
    """Query distinct seasons from rankings table"""
    res = supabase.table("ratings").select("season", count="exact").execute()
    seasons = sorted({row["season"] for row in res.data}, reverse=True)
    return seasons

@st.cache_data(ttl=3600)
def get_weeks_for_season(season: int) -> List[int]:
    """Query distinct weeks for a given season from rankings table"""
    res = supabase.table("ratings").select("week").eq("season", season).execute()
    weeks = sorted({row["week"] for row in res.data})
    return weeks

@st.cache_data(ttl=3600)
def load_rankings_from_db(season: int, week: int) -> pd.DataFrame:
    """
    Load rankings rows for season/week.
    Expected columns: team, season, week, wins, losses, rating, logos (from teams join if needed)
    We'll try to fetch logos from the 'teams' table by joining in Python if not present.
    """
    # Primary query: rankings table (assumes team name stored in 'team')
    res = supabase.table("ratings").select("*").eq("season", season).eq("week", week).execute()
    df = pd.DataFrame(res.data)
    if df.empty:
        return df

    # Ensure wins/losses numeric
    for col in ["wins", "losses", "rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # if missing, add defaults
            df[col] = 0

    # Try to enrich with logos from teams table if logos column missing
    if "logos" not in df.columns:
        # fetch teams table for the season
        team_rows = supabase.table("teams").select("school, logos").eq("season", season).execute()
        team_map = {row["school"]: row.get("logos", []) for row in team_rows.data}
        df["logos"] = df["team"].map(lambda t: team_map.get(t, []))
    else:
        # parse logos values if necessary
        df["logos"] = df["logos"].apply(lambda v: safe_parse_logos(v))

    return df

@st.cache_data(ttl=3600)
def load_previous_rankings_for_week(season: int, week: int) -> pd.DataFrame:
    """Load the rankings for the previous week (week - 1). If not available, return empty DF."""
    if week <= 0:
        return pd.DataFrame()
    prev_week = week - 1
    res = supabase.table("ratings").select("*").eq("season", season).eq("week", prev_week).execute()
    prev_df = pd.DataFrame(res.data)
    return prev_df

# ---------------------------
# Layout - filters on top
# ---------------------------
# Retrieve seasons and weeks
seasons = get_available_seasons()
if not seasons:
    st.warning("No seasons found in the rankings table.")
    st.stop()

# Place filters inline above table
col_a, col_b, col_c = st.columns([1, 1, 6])
with col_a:
    selected_season = st.selectbox("Season", seasons, index=0)
with col_b:
    weeks = get_weeks_for_season(selected_season)
    if not weeks:
        st.warning("No weeks found for the selected season.")
        st.stop()
    # default to last (latest) week
    selected_week = st.selectbox("Week", weeks, index=len(weeks)-1)
with col_c:
    st.write("")  # spacer column for layout balance

st.markdown("---")

# ---------------------------
# Load data & compute ranking table
# ---------------------------
try:
    df = load_rankings_from_db(selected_season, selected_week)
except Exception as e:
    st.error(f"Failed to load rankings: {e}")
    st.stop()

if df.empty:
    st.info(f"No rankings found for Season {selected_season} Week {selected_week}.")
    st.stop()

# sort by rating desc
df = df.sort_values("rating", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

# compute record string
if "wins" in df.columns and "losses" in df.columns:
    df["record"] = df["wins"].fillna(0).astype(int).astype(str) + "–" + df["losses"].fillna(0).astype(int).astype(str)
else:
    df["record"] = ""

# compute delta vs previous week
try:
    prev_df = load_previous_rankings_for_week(selected_season, selected_week)
    if not prev_df.empty:
        prev_sorted = prev_df.sort_values("rating", ascending=False).reset_index(drop=True)
        prev_sorted["rank"] = prev_sorted.index + 1
        # mapping team -> previous rank
        prev_rank_map = {row["team"]: int(row["rank"]) for _, row in prev_sorted.iterrows()}
        df["prev_rank"] = df["team"].map(lambda t: prev_rank_map.get(t, np.nan))
        # delta = prev_rank - current_rank (positive = moved up (improved))
        df["delta"] = df["prev_rank"].fillna(df["rank"]).astype(float) - df["rank"]
        df["delta"] = df["delta"].astype(int)
    else:
        df["delta"] = 0
except Exception:
    # if previous week not available or error, set zero deltas
    df["delta"] = 0

# prepare display columns and HTML fragments
def make_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    disp = pd.DataFrame()
    disp["Rank"] = df["rank"]
    # Logo + Team HTML
    def team_html(row):
        logos = row["logos"] if "logos" in row and row["logos"] is not None else []
        logo_html = logo_img_html(safe_parse_logos(logos))
        name = row["team"]
        return f'<div style="display:flex;align-items:center;">{logo_html}<span style="vertical-align:middle">{name}</span></div>'

    disp["Team"] = df.apply(team_html, axis=1)
    disp["Record"] = df["record"]
    disp["Rating"] = df["rating"].map(lambda x: f"{x:.2f}")
    disp["Δ"] = df["delta"].map(format_delta_cell)
    return disp

disp_df = make_display_dataframe(df)

# Add a small style block for the HTML table
table_style = """
<style>
table {border-collapse: collapse; width: 100%;}
th, td {text-align: left; padding: 8px;}
</style>
"""

# th, tr {background-color: #f8f9fa;}

# Render table as HTML
html_table = table_style + disp_df.to_html(escape=False, index=False)
st.markdown(html_table, unsafe_allow_html=True)

st.markdown(f"**Last updated:** Season {selected_season}, Week {selected_week}")

# ---------------------------
# Footer / small notes
# ---------------------------
st.markdown("---")
st.markdown(
    "This project builds an automated, data-driven college football team ranking system using convex quadratic programming (QP)."
    "Each week, the model ingests updated game results, optimizes team “rating” values that best explain on-field outcomes, and pushes the computed rankings to a cloud database for storage and publication."
    
    "The model is designed to:"
        "Rank all FBS teams based on results from the season up to the current week."
        "Account for margin of victory, game location, and opponent strength."
        "Handle FCS losses with a dummy “FCS” rating variable."
        "Smoothly transition from prior-season ratings to data-driven current-season rankings as the season progresses."
        "Run automatically via a GitHub Action each Sunday morning, pulling new data, solving the model, and updating results in Supabase."

)
