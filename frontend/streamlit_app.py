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

st.title("🏈 College Football Rankings")
st.caption("A convex quadratic program to optimally rank college football teams based on game outcomes")
st.caption("Developed by Wyatt Entrekin")

st.markdown("""
### Project Links

**Data Source:** [📊 College Football Data API](https://collegefootballdata.com/)  
**Source Code:** [💻 GitHub Repository](https://github.com/wmentrekin/cfb-rankings/tree/main)  
**Data Storage:** [🔐 Supabase](https://supabase.com/)  
**Frontend:** [🚀 Streamlit](https://streamlit.io/)  
**Documentation:** [📘 Methodology](#methodology)  
**Contact:**  
- ✉️ wmentrekin@gmail.com
- [🔗 LinkedIn](https://linkedin.com/in/wmentrekin)
""")

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
st.header("Methodology")

st.markdown("""
The inspiration for this ranking model came from a personal desire to objectively rank college football teams solely based on game outcomes.
            
Year after year, I find that college football discourse is filled with subjective opinions about which teams are "better" or "worse," often influenced by biases, media narratives, and historical prestige.
I'm tired of hearing debates about hypothetical matchups when we are lucky enough as fans to witness so many real games each season.
            
Thus, I was inspired to spend time developing a model that ranked teams solely based on game outcomes, without the influence of subjective factors such as recruiting rankings, preseason expectations, or traditional power ratings.
I chose to implement this using a constraint optimization approach, specifically a convex quadratic programming model, because I hadn't seen many existing models that took this approach in a transparent and mathematically rigorous way.
At the core, this model is very simple, it tries to assign ratings to teams such that:
- Teams that win games have higher ratings than the teams they beat relative to a factor of the margin of victory, adjusted for home/away/neutral site.
- Teams that lose to FCS opponents are penalized severely
- In order to avoid arbitrary early-season rankings, teams' ratings are tied more closely to their prior-season performance earlier in the season, until enough games are played to connect enough teams through common opponents. Imagine a network graph where teams are nodes and games are edges; as more edges are added, the relative positions of nodes become clearer.
- It minimizes total "ranking inconsistency" subject to logical constraints about game results.

Additionally, this model has been a great exercise for me to apply my optimization and mathematical modeling knowledge from my undergraduate background in industrial engineering and operations research, my cloud computing and automation experience from working professionally as a data engineer, and my passion for college football.
Unlike some computer rankings, my code and methodology are fully open and transparent for anyone to review, critique, and improve upon, so please feel free to reach out to me if you have any suggestions, thoughts, or questions.
I plan on continuing to refine and improve this model over time and adding more features to this app as well.
I hope you find this model as interesting and useful as I have building it!
""")

with st.expander("🔢 Decision Variables & Parameters", expanded=False):
    st.markdown("""
    **Sets**
    - $\mathcal{T}$: set of all FBS teams
    - $\mathcal{G}$: set of all games $(i,j,k)$ where team $i \in \mathcal{T}$ played team $j \in \mathcal{T}$ in their $k$th matchup
    - $\mathcal{F}$: set of FBS teams that lost to FCS opponents

    **Variables**
    - $r_i \in \mathbb{R}_+$ : rating for team $i$, $\\forall i \in \mathcal{T}$
    - $r_{\\text{fcs}} \in \mathbb{R}_+$ : rating for the dummy FCS team
    - $z_{i,j,k} \in \mathbb{R}_+$ : ranking violation slack for game $(i,j,k)$, $\\forall (i,j,k) \in \mathcal{G}$
    - $z_{\\text{fcs},i} \in \mathbb{R}_+$ : FCS loss slack for team $i$, $\\forall i \in \mathcal{F}$
    """)

    st.markdown("""
    **Parameters**
    - $\\lambda(w)$ : Prior regularization weight where:
      $\\lambda(w) = \\begin{cases} 
      \\frac{7-w}{700} & \\text{if } w < 7 \\\\
      0 & \\text{if } w \\geq 7
      \\end{cases}$
      where $w$ is the week number
    - $\\beta = 2.0$ : FCS loss slack weight
    - $\\gamma = 1.0$ : Small margin penalty coefficient
    - $\\nu = 500$ : Loss slack scaling factor
    - $M = 200$ : Big-M constant for constraint formulation
    - $R_{\\text{min}} = 5$ : Lower bound for FCS team rating
    - $R_{\\text{max}} = 15$ : Upper bound for FCS team rating
    - $\\alpha = \\begin{cases}
      1.0 & \\text{neutral site} \\\\
      0.8 & \\text{home team win} \\\\
      1.2 & \\text{away team win}
      \\end{cases}$
    - $\\text{margin}_{i,j,k}$ : point differential in game $(i,j,k)$
    - $\\text{prior}_i$ : prior rating for team $i$ from previous season's final rankings (default 35 for new FBS teams)
    """)

with st.expander("📐 Objective Function", expanded=False):
    st.markdown("""
    $$
    \\begin{aligned}
    \\text{minimize} \\quad & \\sum_{(i,j,k) \\in \\mathcal{G}} \\nu \\cdot \\text{margin}_{i,j,k} \\cdot \\alpha_{i,j,k} \\cdot z_{i,j,k} & [\\text{Slack penalty}] \\\\
    & + \\sum_{(i,j,k) \\in \\mathcal{G}} \\gamma \\cdot [\\max(0, r_{\\text{loser}} + \\text{margin}_{i,j,k} - r_{\\text{winner}})]^2 & [\\text{Soft margin penalty}] \\\\
    & + \\sum_{i \\in \\mathcal{F}} \\beta \\cdot z_{\\text{fcs},i} & [\\text{FCS loss penalty}] \\\\
    & + \\lambda \\sum_{i \\in \\mathcal{T}} (r_i - \\text{prior}_i)^2 & [\\text{Prior regularization}]
    \\end{aligned}
    $$
    """)
    
    st.markdown("""
    """)

with st.expander("⚖️ Constraints", expanded=False):
    st.markdown("""
    Subject to the following constraints:
    
    **Loss slack constraints:**
    $$
    r_i + z_{i,j,k} \\leq r_j + M \\quad \\forall (i,j,k) \\in \\mathcal{G}
    $$

    **FCS loss constraints:**
    $$
    r_i + z_{\\text{fcs},i} \\leq r_{\\text{fcs}} + M \\quad \\forall i \\in \\mathcal{F}
    $$
                
    **Non-negativity constraints:**
    $$
    r_i \\geq 0 \\quad \\forall i \\in \\mathcal{T}
    $$          

    **FCS rating bounds:**
    $$
    R_{\\text{min}} \\leq r_{\\text{fcs}} \\leq R_{\\text{max}}
    $$
    """)

with st.expander("📝 Implementation Notes", expanded=False):
    st.markdown("""
    - **Prior ratings** are carried over from the previous season's final model output. New FBS teams receive a default prior rating of 35.
    - The $\\lambda$ parameter enforces prior-season influence, linearly dropping to 0 from Week 1 to Week 7, eventually to rely entirely on current season results.
    - **FCS losses** are handled via a dummy FCS team whose rating is constrained between $R_{\\text{min}}$ and $R_{\\text{max}}$. This allows the model to penalize FBS teams losing to FCS opponents without needing to explicitly rate every FCS team.
    - **Slack variables** ($z$) capture ranking violations, weighted by margin and location (home/away/neutral).
    - **Soft margin penalties** encourage appropriate rating separation based on game margins.
    - **Home/away adjustments** use $\\alpha$ multipliers to account for game location advantage.
    - The optimization is solved using [CVXPY](https://www.cvxpy.org/) with default convex solvers, returning team ratings sorted in descending order.
    """)
st.markdown("---")