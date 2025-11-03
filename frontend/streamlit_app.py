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
    page_title="CFB Rankings - Entrekin Quadratic Index",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 Entrekin Quadratic Index")
st.caption("A convex quadratic program to optimally rank college football teams based on game outcomes")
st.caption("Updated every Sunday morning at 3:15 AM ET during the college football season")
st.caption("Developed by Wyatt Entrekin")

st.markdown("""
### Project Links

**Data Source:** [📊 College Football Data API](https://collegefootballdata.com/)  
**Source Code:** [💻 GitHub Repository](https://github.com/wmentrekin/cfb-rankings/tree/main)  
**Data Storage:** [🔐 Supabase](https://supabase.com/)  
**Frontend:** [🚀 Streamlit](https://streamlit.io/)  
**Automation:** [⚡ GitHub Actions](https://github.com/features/actions)  
**Documentation:** [📘 Methodology](#methodology)  
**Contact:**  
- ✉️ wmentrekin@gmail.com
- [🔗 LinkedIn](https://linkedin.com/in/wmentrekin)
""")
st.markdown("---")

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
st.markdown("""
### Rankings
            """)
# ---------------------------
# Layout - filters on top
# ---------------------------
st.markdown("""
<style>
    .filters {
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stSelectbox {
        min-width: 120px;
        margin-bottom: 1rem;
    }
    /* Hide last margin on desktop mode */
    @media (min-width: 768px) {
        .stSelectbox {
            margin-bottom: 0;
        }
    }
</style>
""", unsafe_allow_html=True)

seasons = get_available_seasons()
if not seasons:
    st.warning("No seasons found in the rankings table.")
    st.stop()

with st.container():
    st.markdown('<div class="filters">', unsafe_allow_html=True)
    
    # Check if we're likely in desktop mode (will show side by side)
    use_columns = st.session_state.get('_is_desktop', True)
    
    if use_columns:
        col_a, col_b, col_c = st.columns([1, 1, 6])
        with col_a:
            selected_season = st.selectbox("📅 Season", seasons, index=0)
        with col_b:
            weeks = get_weeks_for_season(selected_season)
            if not weeks:
                st.warning("No weeks found for the selected season.")
                st.stop()
            selected_week = st.selectbox("📊 Week", weeks, index=len(weeks)-1)
        with col_c:
            st.write("")
    else:
        # Stack filters vertically on mobile
        selected_season = st.selectbox("📅 Season", seasons, index=0)
        weeks = get_weeks_for_season(selected_season)
        if not weeks:
            st.warning("No weeks found for the selected season.")
            st.stop()
        selected_week = st.selectbox("📊 Week", weeks, index=len(weeks)-1)
    
    st.markdown('</div>', unsafe_allow_html=True)

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
        pd.DataFrame: Formatted DataFrame with columns using responsive column names
    """
    disp = pd.DataFrame()
    disp["<span class='desktop-only'>Rank</span><span class='mobile-only'>#</span>"] = df["rank"]
    # Logo + Team HTML
    def team_html(row):
        logos = row["logos"] if "logos" in row and row["logos"] is not None else []
        logo_html = logo_img_html(safe_parse_logos(logos))
        name = row["team"]
        return f'<div style="display:flex;align-items:center;">{logo_html}<span style="vertical-align:middle">{name}</span></div>'

    disp["Team"] = df.apply(team_html, axis=1)
    disp["<span class='desktop-only'>Record</span><span class='mobile-only'>W-L</span>"] = df["record"]
    disp["<span class='desktop-only'>Rating</span><span class='mobile-only'>RTG</span>"] = df["rating"].map(lambda x: f"{x:.2f}")
    disp["Δ"] = df["delta"].map(format_delta_cell)
    return disp

disp_df = make_display_dataframe(df)

# Add styling for the rankings table
table_style = """
<style>
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
        table-layout: fixed;
    }
    th {
        font-weight: 600;
        text-align: center !important;
        padding: 12px 4px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    td {
        text-align: center !important;
        padding: 8px 4px;
        vertical-align: middle;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Default column widths (mobile) */
    th:nth-child(1), td:nth-child(1) { width: 15%; } /* Rank */
    th:nth-child(2), td:nth-child(2) { width: 40%; } /* Team */
    th:nth-child(3), td:nth-child(3) { width: 15%; } /* Record */
    th:nth-child(4), td:nth-child(4) { width: 18%; } /* Rating */
    th:nth-child(5), td:nth-child(5) { width: 12%; } /* Delta */
    
    /* Team column specific styling */
    td:nth-child(2) {
        padding-right: 8px;
        min-width: 160px; /* Prevent too narrow on mobile */
    }
    td:nth-child(2) div {
        justify-content: flex-start;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Responsive header text */
    .mobile-only { display: inline; }
    .desktop-only { display: none; }
    
    /* Desktop adjustments for team column */
    @media (min-width: 768px) {
        .mobile-only { display: none; }
        .desktop-only { display: inline; }
        
        td:nth-child(2) div {
            justify-content: center;
        }
    }

    /* Desktop adjustments */
    @media (min-width: 768px) {
        /* Center table on desktop */
        table {
            max-width: 1000px;
            margin-left: auto;
            margin-right: auto;
        }
        th:nth-child(1), td:nth-child(1) { width: 10%; } /* Rank */
        th:nth-child(2), td:nth-child(2) { width: 30%; } /* Team */
        th:nth-child(3), td:nth-child(3) { width: 15%; } /* Record */
        th:nth-child(4), td:nth-child(4) { width: 15%; } /* Rating */
        th:nth-child(5), td:nth-child(5) { width: 10%; } /* Delta */
        
        td:nth-child(2) {
            max-width: 300px; /* Prevent too wide on desktop */
        }
    }
</style>
"""

# Render table as HTML
html_table = table_style + disp_df.to_html(escape=False, index=False)
st.markdown(html_table, unsafe_allow_html=True)

st.caption(f"Rankings for Season {selected_season} Week {selected_week}")

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
- Teams that lose to FCS opponents are penalized severely.
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
    """)

    st.markdown("""
    **Parameters**
    - $\\lambda(w)$ : Prior regularization weight where:
      $\\lambda(w) = \\begin{cases} 
      \\frac{7-w}{700} & \\text{if } w < 7 \\\\
      0 & \\text{if } w \\geq 7
      \\end{cases}$
      where $w$ is the week number
    - $\\gamma_{\\text{margin}} = 0.05$ : Small margin penalty coefficient
    - $\\gamma_{\\text{loss}} = 0.5$ : FBS Loss penalty coefficient
    - $\\gamma_{\\text{fcs}} = 5.0$ : FCS Loss penalty coefficient
    - $r_{\\text{min}} = 0.01$ : Lower bound for FBS team rating
    - $r_{\\text{max}} = 100$ : Upper bound for FBS team rating
    - $r_{\\text{fcs-min}} = 5$ : Lower bound for FCS team rating
    - $r_{\\text{fcs-max}} = 15$ : Upper bound for FCS team rating
    - Margin scaling parameters:
      - $\\text{TARGET\\_GAP\\_FOR\\_MEDIAN} = 7.0$ : Rating points that the median margin should represent
      - $\\text{MAX\\_RATING\\_GAP} = 20.0$ : Maximum rating gap any margin can demand
      - $k_{\\text{margin}} = \\frac{\\text{TARGET\\_GAP\\_FOR\\_MEDIAN}}{\\max(\\sqrt{\\text{median\\_margin}}, 10^{-6})}$ : Scaling factor for margins
      - $\\text{margin}_{i,j,k}$ : point differential in game $(i,j,k)$
      - $m_{i,j,k} = \\min(k_{\\text{margin}} \\cdot \\sqrt{\\max(0, \\text{margin}_{i,j,k})}, \\text{MAX\\_RATING\\_GAP})$
    - $\\alpha = \\begin{cases}
      1.0 & \\text{neutral site} \\\\
      0.8 & \\text{home team win} \\\\
      1.2 & \\text{away team win}
      \\end{cases}$
    - $\\text{prior}_i$ : prior rating for team $i$ from previous season's final rankings (default 15 for new FBS teams), model for prior is same as this model, but run on prior season's data
    """)

with st.expander("📐 Objective Function", expanded=False):
    st.markdown("""
    $$
    \\begin{aligned}
    \\text{minimize} \\quad & \\sum_{(i,j,k) \\in \\mathcal{G}} \\gamma_{\\text{margin}} \\cdot [\\max(0, r_{\\text{loser}} + (m_{i,j,k} \\cdot \\alpha_{i,j,k}) - r_{\\text{winner}})]^2 & [\\text{Margin Penalty}] \\\\
    & + \\sum_{(i,j,k) \\in \\mathcal{F}} \\gamma_{\\text{fcs}} \\cdot [\\max(0, r_{\\text{i}} + (m_{i,j,k} \\cdot \\alpha_{i,j,k}) - r_{\\text{fcs}})]^2 & [\\text{FCS Margin Penalty}] \\\\
    & + \\sum_{i \\in \\mathcal{T}} \\gamma_{\\text{loss}} \\cdot (r_i \\cdot \\frac{\\text{losses}_i}{\\text{wins}_i + \\text{losses}_i})^2 & [\\text{Loss Rate Penalty}] \\\\
    & + \\sum_{i \\in \\mathcal{T}} \\lambda \\cdot (r_i - \\text{prior}_i)^2 & [\\text{Prior regularization}]
    \\end{aligned}
    $$
    """)
    
    st.markdown("""
    - Margin penalties give a quadratic penalty for any game where the rating difference does not sufficiently explain the margin of victory. The margin is:
      1. Scaled by the square root to reduce the impact of extreme margins
      2. Multiplied by $k_{\\text{margin}}$ to normalize around the median margin
      3. Capped at $\\text{MAX\\_RATING\\_GAP}$ to prevent blowouts from dominating
      4. Finally adjusted by $\\alpha_{i,j,k}$ for home/away/neutral site effects
    - Loss rate penalties discourage teams with poor win-loss records from having high ratings.
    - Prior regularization ties team ratings to their prior season's ratings early in the season when data is sparse.
    """)

with st.expander("⚖️ Constraints", expanded=False):
    st.markdown("""                
    **FBS rating bounds:**
    $$
    r_{\\text{min}} \\leq r_i \\leq r_{\\text{max}} \\quad \\forall i \\in \\mathcal{T}
    $$          

    **FCS rating bounds:**
    $$
    r_{\\text{fcs-min}} \\leq r_{\\text{fcs}} \\leq r_{\\text{fcs-max}}
    $$
    """)

with st.expander("📝 Implementation Notes", expanded=False):
    st.markdown("""
    - **Prior ratings** are carried over from the previous season's final model output. New FBS teams receive a default prior rating of 15.
    - The $\\lambda$ parameter enforces prior-season influence, linearly dropping to 0 from Week 1 to Week 7, eventually to rely entirely on current season results.
    - **FCS losses** are handled via a dummy FCS team whose rating is constrained between $r_{\\text{fcs-min}}$ and $r_{\\text{fcs-max}}$. This allows the model to penalize FBS teams losing to FCS opponents without needing to explicitly rate every FCS team.
    - **Margin penalties** encourage appropriate rating separation based on game margins.
    - **Loss rate penalties** discourage high ratings for teams with poor win-loss records.
    - **Prior regularization** prevents extreme ratings early in the season when data is sparse.
    - **Home/away adjustments** use $\\alpha$ multipliers to account for game location advantage.
    - The optimization is solved using [CVXPY](https://www.cvxpy.org/) with default convex solvers, returning team ratings sorted in descending order.
    """)
st.markdown("---")