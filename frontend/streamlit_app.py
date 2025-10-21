# streamlit_app.py
import streamlit as st # type: ignore
from supabase import create_client, Client # type: ignore
import pandas as pd
import numpy as np
import os
import ast
from typing import List, Optional

# ---------------------------
# Configuration / Init
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
