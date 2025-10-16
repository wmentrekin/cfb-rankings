import streamlit as st # type: ignore
from supabase import create_client # type: ignore
import pandas as pd
import plotly.express as px # type: ignore
import os

# ------------------------------------------------
# Initialize Supabase client
# ------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_KEY in Streamlit Secrets.")
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(
    page_title="CFB Linear Programming Rankings",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 College Football Linear Programming Rankings")
st.markdown(
    "An objective, data-driven ranking system using convex optimization — automatically updated weekly."
)

# ------------------------------------------------
# Sidebar: season & week selection
# ------------------------------------------------
with st.sidebar:
    st.header("Filters")
    # Fetch available seasons from DB
    seasons_data = supabase.table("ratings").select("season").execute()
    seasons = sorted(list({item["season"] for item in seasons_data.data}), reverse=True)
    selected_season = st.selectbox("Season", seasons, index=0)

    # Fetch available weeks for selected season
    weeks_data = supabase.table("ratings").select("week").eq("season", selected_season).execute()
    weeks = sorted(list({item["week"] for item in weeks_data.data}))
    selected_week = st.selectbox("Week", weeks, index=len(weeks)-1 if weeks else 0)

# ------------------------------------------------
# Load data for selected season/week
# ------------------------------------------------
@st.cache_data(ttl=3600)
def load_rankings(season, week):
    res = supabase.table("ratings") \
        .select("*") \
        .eq("season", season) \
        .eq("week", week) \
        .execute()
    df = pd.DataFrame(res.data)
    if df.empty:
        st.warning("No data found for that week yet.")
        return None
    return df.sort_values("rating", ascending=False).reset_index(drop=True)

df = load_rankings(selected_season, selected_week)

if df is not None:
    st.subheader(f"Top 25 Rankings – Week {selected_week}, {selected_season}")
    st.dataframe(df.head(25)[["team", "rating", "wins", "losses"]], hide_index=True, use_container_width=True)

    # ------------------------------------------------
    # Team lookup & rating trend
    # ------------------------------------------------
    team_list = sorted(df["team"].unique())
    selected_team = st.selectbox("Select a team to view historical trend:", team_list)

    @st.cache_data(ttl=3600)
    def load_team_history(team_name):
        res = supabase.table("ratings").select("*").eq("team", team_name).eq("season", selected_season).execute()
        hist_df = pd.DataFrame(res.data)
        return hist_df.sort_values("week")

    hist_df = load_team_history(selected_team)
    if not hist_df.empty:
        fig = px.line(
            hist_df,
            x="week",
            y="rating",
            markers=True,
            title=f"{selected_team} Rating Trend ({selected_season})"
        )
        fig.update_layout(xaxis_title="Week", yaxis_title="Rating")
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------
    # Summary stats
    # ------------------------------------------------
    avg_rating = df["rating"].mean()
    st.metric("Average Rating", f"{avg_rating:.2f}")
    st.metric("Number of Teams", len(df))
else:
    st.info("Select a valid week to view rankings.")
