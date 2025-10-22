# streamlit_app.py
import streamlit as st # type: ignore
from supabase import create_client, Client # type: ignore
import pandas as pd
import numpy as np
import os
import ast
from typing import List, Optional

# ---------------------------
# Streamlit app configuration
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
    Generate an HTML img tag for displaying a team logo.
    Args:
        logo_urls (List[str]): List of URLs to potential logo images
        width (int, optional): Width in pixels for the logo image. Defaults to 36.
    Returns:
        str: HTML string containing an img tag with the first valid logo URL,
             or empty string if no logos available
    """
    if not logo_urls:
        return ""
    url = logo_urls[0]
    return f'<img src="{url}" alt="logo" style="width:{width}px;height:auto;border-radius:4px;margin-right:8px;vertical-align:middle;">'

def format_delta_cell(delta: int) -> str:
    """
    Format a ranking change value as colored HTML with directional indicators.
    Args:
        delta (int): The change in ranking (positive = moved up, negative = moved down)
    
    Returns:
        str: HTML string with:
            - Green up arrow (▲) for positive changes
            - Red down arrow (▼) for negative changes
            - Yellow dash (—) for no change
            Value is formatted with appropriate color and arrow/dash
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
    """
    Retrieve all unique season years available in the rankings database.
    Returns:
        List[int]: List of season years in descending order (most recent first)
    Note:
        Results are cached for 1 hour (3600 seconds) using Streamlit's caching
    """
    res = supabase.table("ratings").select("season", count="exact").execute()
    seasons = sorted({row["season"] for row in res.data}, reverse=True)
    return seasons

@st.cache_data(ttl=3600)
def get_weeks_for_season(season: int) -> List[int]:
    """
    Retrieve all available weeks for a specific season from the rankings database.
    Args:
        season (int): The year of the season to query
    Returns:
        List[int]: List of week numbers in ascending order
    Note:
        Results are cached for 1 hour (3600 seconds) using Streamlit's caching
    """
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
    res = supabase.table("ratings").select("*").eq("season", season).eq("week", week).execute()
    df = pd.DataFrame(res.data)
    if df.empty:
        return df
    for col in ["wins", "losses", "rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0
    if "logos" not in df.columns:
        team_rows = supabase.table("teams").select("school, logos").eq("season", season).execute()
        team_map = {row["school"]: row.get("logos", []) for row in team_rows.data}
        df["logos"] = df["team"].map(lambda t: team_map.get(t, []))
    else:
        df["logos"] = df["logos"].apply(lambda v: safe_parse_logos(v))

    return df

@st.cache_data(ttl=3600)
def load_previous_rankings_for_week(season: int, week: int) -> pd.DataFrame:
    """
    Retrieve rankings from the previous week for delta calculations.
    Args:
        season (int): Season year
        week (int): Current week number
    Returns:
        pd.DataFrame: Rankings data from the previous week (week - 1) containing
                     columns: team, rating, etc. Returns empty DataFrame if week <= 0
                     or if previous week's data is not available
    Note:
        Results are cached for 1 hour (3600 seconds) using Streamlit's caching
    """
    if week <= 0:
        return pd.DataFrame()
    prev_week = week - 1
    res = supabase.table("ratings").select("*").eq("season", season).eq("week", prev_week).execute()
    prev_df = pd.DataFrame(res.data)
    return prev_df

# ---------------------------
# Layout - filters on top
# ---------------------------
seasons = get_available_seasons()
if not seasons:
    st.warning("No seasons found in the rankings table.")
    st.stop()
col_a, col_b, col_c = st.columns([1, 1, 6])
with col_a:
    selected_season = st.selectbox("Season", seasons, index=0)
with col_b:
    weeks = get_weeks_for_season(selected_season)
    if not weeks:
        st.warning("No weeks found for the selected season.")
        st.stop()
    selected_week = st.selectbox("Week", weeks, index=len(weeks)-1)
with col_c:
    st.write("")
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
df = df.sort_values("rating", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1
if "wins" in df.columns and "losses" in df.columns:
    df["record"] = df["wins"].fillna(0).astype(int).astype(str) + "–" + df["losses"].fillna(0).astype(int).astype(str)
else:
    df["record"] = ""
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
    df["delta"] = 0

def make_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw rankings DataFrame into a display-ready format with HTML formatting.
    Args:
        df (pd.DataFrame): Raw rankings data containing columns:
            - rank: Current ranking position
            - team: Team name
            - logos: List of logo URLs
            - record: Win-loss record
            - rating: Numerical rating value
            - delta: Change in ranking from previous week
    Returns:
        pd.DataFrame: Formatted DataFrame with columns:
            - Rank: Numerical ranking
            - Team: HTML-formatted team name with logo
            - Record: Win-loss record
            - Rating: Formatted rating value
            - Δ: HTML-formatted ranking change indicator
    """
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

# Render table as HTML
html_table = table_style + disp_df.to_html(escape=False, index=False)
st.markdown(html_table, unsafe_allow_html=True)

st.markdown(f"**Last updated:** Season {selected_season}, Week {selected_week}")

# ---------------------------
# Model Documentation
# ---------------------------
st.markdown("---")
st.header("Model Documentation")

st.markdown("""
The model assigns each FBS team $i$ a continuous rating $r_i$, and a dummy FCS rating $r_{\\text{fcs}}$.  
It minimizes total "ranking inconsistency" subject to logical constraints about game results and prior-season expectations.
""")

with st.expander("📐 Objective Function", expanded=False):
    st.write("""
    We minimize a weighted combination of slack penalties, prior regularization, and soft margin terms:
    """)
    
    st.latex(r"\sum_{\text{games}} \nu \cdot \text{margin} \cdot \alpha \cdot z_{\text{winner,loser}} \quad \text{[Slack penalty]}")
    
    st.latex(r"\quad + \sum_{\text{games}} \gamma \cdot [\max(0, r_{\text{loser}} + \text{margin} - r_{\text{winner}})]^2 \quad \text{[Soft margin penalty]}")
    
    st.latex(r"\quad + \sum_{\text{FCS losses}} \beta \cdot z_{\text{fcs}} \quad \text{[FCS loss penalty]}")
    
    st.latex(r"\quad + \lambda \sum_{\text{teams}} (r_{\text{team}} - \text{prior}_{\text{team}})^2 \quad \text{[Prior regularization]}")
    
    st.markdown("""
    Where:
    - For each game, winner and loser are determined by the actual game outcome
    - $\\alpha$ adjusts for home/away games:
        - $\\alpha = 1.0$ for neutral site games
        - $\\alpha = 0.8$ if home team wins
        - $\\alpha = 1.2$ if away team wins
    - $\\text{margin}$ is the point differential in the game
    - $z_{\\text{winner,loser}}$ is the slack variable representing ranking violation
    - $\\lambda$ decays to zero after week 7 to rely fully on current season data
    """)

with st.expander("⚖️ Constraints", expanded=False):
    st.write("For every game $(i, j, k)$ where team $i$ played team $j$ for the $k$th time:")

    st.markdown("""
    1. **Loss slack constraint:**
    """)
    st.latex(r"r_{\text{loser}} + z_{\text{winner,loser}} \leq r_{\text{winner}} + M")
    st.write("(If $r_{\\text{loser}} > r_{\\text{winner}}$, $z_{\\text{winner,loser}}$ must absorb the violation)")

    st.markdown("""
    2. **FCS loss constraint:**
    """)
    st.latex(r"r_{\text{team}} + z_{\text{fcs,team}} \leq r_{\text{fcs}} + M")
    st.write("(For teams that lost to FCS opponents)")

    st.markdown("""
    3. **Rating bounds:**
    """)
    st.latex(r"r_i \geq 0 \quad \forall i \in \text{teams}")
    st.latex(r"R_{\text{min}} \leq r_{\text{fcs}} \leq R_{\text{max}}")

with st.expander("🔢 Decision Variables", expanded=False):
    st.markdown("""
    - $r_i$: continuous rating for each FBS team  
    - $r_{\\text{fcs}}$: single rating for the dummy FCS team  
    - $z_{ijk}$: nonnegative slack variable representing a violation (team $i$ ranked above $j$ despite losing)  
    - $z_{\\text{fcs},i}$: slack for losses to FCS teams
    """)

with st.expander("🎛️ Parameters", expanded=False):
    st.markdown("""
    | Parameter | Description | Default Value |
    |:---------:|:------------|:-------------:|
    | $\\lambda$ | Prior regularization weight | $(7 - \\text{week}) / 700$ before week 7, 0 after |
    | $\\beta$ | FCS loss slack weight | 2.0 |
    | $\\gamma$ | Small margin penalty | 1.0 |
    | $\\nu$ | Loss slack scaling | 500 |
    | $M$ | Big-M constant | 200 |
    | $R_{\\text{min}}$ | FCS rating lower bound | 5 |
    | $R_{\\text{max}}$ | FCS rating upper bound | 15 |
    """)

with st.expander("📝 Implementation Notes", expanded=False):
    st.markdown("""
    - **Prior ratings** are carried over from the previous season's final model output. New FBS teams receive a default prior rating of 35.
    - The $\\lambda$ parameter enforces prior-season influence through Week 6, then drops to 0 to rely entirely on current season results.
    - **FCS losses** are handled via a dummy FCS team whose rating is constrained between $R_{\\text{min}}$ and $R_{\\text{max}}$.
    - **Slack variables** ($z$) capture ranking violations, weighted by margin and location (home/away/neutral).
    - **Soft margin penalties** encourage appropriate rating separation based on game margins.
    - **Home/away adjustments** use $\\alpha$ multipliers to account for game location advantage.
    - The optimization is solved using [CVXPY](https://www.cvxpy.org/) with default convex solvers, returning team ratings sorted in descending order.
    """)
st.markdown("---")
