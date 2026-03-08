"""
pages/leaderboard.py — Live class leaderboard for The Loopback Initiative.
Instructor projects this page during / after the in-class session.
Auto-refreshes every 15 seconds.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Leaderboard — The Loopback Initiative",
    page_icon="♻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  html, body, [data-testid="stApp"] {
      background-color: #0d1117;
      color: #c9d1d9;
      font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }
  [data-testid="metric-container"] {
      background-color: #161b22 !important;
      border: 1px solid #30363d;
      border-radius: 8px;
  }
  [data-testid="stMetricValue"] { color: #58a6ff; font-size: 1.6rem; font-weight: 700; }
  [data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
  div.stButton > button {
      background-color: #21262d;
      color: #c9d1d9;
      border: 1px solid #30363d;
      border-radius: 6px;
  }
  div.stButton > button:hover { border-color: #58a6ff; color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# Auto-refresh every 15 seconds
st_autorefresh(interval=15_000, key="leaderboard_refresh")

GRADE_COLOR = {
    "S": "#ffd700",
    "A": "#3fb950",
    "B": "#58a6ff",
    "C": "#d29922",
    "D": "#f85149",
}
PLOT_LAYOUT = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(family="monospace", color="#c9d1d9", size=12),
    margin=dict(l=40, r=20, t=50, b=100),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
)
_AXIS = dict(gridcolor="#21262d", zerolinecolor="#30363d")


def _get_supabase():
    try:
        from supabase import create_client
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception:
        return None


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="color:#58a6ff; letter-spacing:0.08em;">♻ THE LOOPBACK INITIATIVE</h1>'
    '<p style="color:#8b949e; margin-top:-0.8rem;">Live Class Leaderboard — refreshes every 15 s</p>',
    unsafe_allow_html=True,
)

supabase = _get_supabase()
if supabase is None:
    st.error(
        "Supabase is not configured. Add `[supabase]` credentials to your Streamlit secrets."
    )
    st.stop()

# ── Fetch all sessions ─────────────────────────────────────────────────────────
try:
    all_rows = supabase.table("scores").select("session").execute()
except Exception as e:
    st.error(f"Could not reach Supabase: {e}")
    st.stop()

sessions = sorted(
    set(r["session"] for r in all_rows.data if r.get("session")),
    reverse=True,
)

if not sessions:
    st.info("No scores have been submitted yet. Waiting for students...")
    st.stop()

# ── Session selector ───────────────────────────────────────────────────────────
selected = st.selectbox(
    "Session",
    options=sessions,
    index=0,
    label_visibility="collapsed",
)

# ── Fetch scores for selected session ─────────────────────────────────────────
try:
    resp = (
        supabase.table("scores")
        .select("*")
        .eq("session", selected)
        .order("score", desc=True)
        .execute()
    )
except Exception as e:
    st.error(f"Query failed: {e}")
    st.stop()

df = pd.DataFrame(resp.data) if resp.data else pd.DataFrame()

if df.empty:
    st.info(f"No scores for session **{selected}** yet.")
    st.stop()

# ── Rank column ────────────────────────────────────────────────────────────────
df = df.reset_index(drop=True)
df.insert(0, "rank", range(1, len(df) + 1))

# ── Summary metrics ────────────────────────────────────────────────────────────
st.divider()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Players", len(df))
m2.metric("Top Score", f"{df['score'].max():.1f}")
m3.metric("Avg Score", f"{df['score'].mean():.1f}")
m4.metric("Avg Circular Mix", f"{df['circular_mix'].mean():.1f}%")
stockout_free = int((df["stockout_rounds"] == 0).sum())
m5.metric("Stockout-Free", f"{stockout_free} / {len(df)}")

st.divider()

# ── Leaderboard table ──────────────────────────────────────────────────────────
st.markdown("### Rankings")

display_df = df[[
    "rank", "nickname", "grade", "score",
    "circular_mix", "stockout_rounds", "cumulative_sap", "game_mode",
]].copy()
display_df.columns = [
    "#", "Nickname", "Grade", "Score (0-100)",
    "Circular Mix %", "Stockout Rounds", "Cumulative SAP ($)", "Mode",
]
display_df["Cumulative SAP ($)"] = display_df["Cumulative SAP ($)"].map(
    lambda x: f"${float(x):,.0f}"
)
display_df["Mode"] = display_df["Mode"].map({
    "free_play": "Free Play",
    "primary_lock": "Primary Lock",
    "circular_lock": "Circular Challenge",
}).fillna(display_df["Mode"])

st.dataframe(display_df, hide_index=True, width="stretch")

# ── Score bar chart ────────────────────────────────────────────────────────────
st.markdown("### Score Distribution")
fig = go.Figure(go.Bar(
    x=df["nickname"],
    y=df["score"],
    marker_color=[GRADE_COLOR.get(g, "#8b949e") for g in df["grade"]],
    text=df["grade"],
    textposition="outside",
    textfont=dict(color="#c9d1d9", size=13, family="monospace"),
    hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
))
fig.update_layout(**PLOT_LAYOUT, title=f"Scores — Session: {selected}", showlegend=False)
fig.update_xaxes(**_AXIS, tickangle=-35)
fig.update_yaxes(**_AXIS, range=[0, 105], title="Score")
st.plotly_chart(fig, width="stretch")

# ── Circular mix vs SAP scatter ────────────────────────────────────────────────
if len(df) >= 3:
    st.markdown("### Circular Mix vs. Cumulative SAP")
    fig2 = go.Figure(go.Scatter(
        x=df["circular_mix"],
        y=df["cumulative_sap"],
        mode="markers+text",
        text=df["nickname"],
        textposition="top center",
        textfont=dict(size=10, color="#8b949e"),
        marker=dict(
            size=14,
            color=[GRADE_COLOR.get(g, "#8b949e") for g in df["grade"]],
            line=dict(width=1, color="#30363d"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Circular Mix: %{x:.1f}%<br>"
            "SAP: $%{y:,.0f}<extra></extra>"
        ),
    ))
    fig2.update_layout(**PLOT_LAYOUT, title="Does more circular sourcing lead to higher SAP?")
    fig2.update_xaxes(**_AXIS, tickangle=0, title="Circular Mix (%)", range=[-5, 105])
    fig2.update_yaxes(**_AXIS, title="Cumulative SAP ($)")
    st.plotly_chart(fig2, width="stretch")

# ── Grade distribution ─────────────────────────────────────────────────────────
st.markdown("### Grade Distribution")
grade_counts = df["grade"].value_counts().reindex(["S", "A", "B", "C", "D"], fill_value=0)
fig3 = go.Figure(go.Bar(
    x=grade_counts.index.tolist(),
    y=grade_counts.values.tolist(),
    marker_color=[GRADE_COLOR[g] for g in grade_counts.index],
    text=grade_counts.values.tolist(),
    textposition="outside",
    textfont=dict(color="#c9d1d9", size=14),
))
fig3.update_layout(**PLOT_LAYOUT, title="Grade Distribution", showlegend=False)
fig3.update_xaxes(**_AXIS, tickangle=0, title="Grade")
fig3.update_yaxes(**_AXIS, title="# Students", dtick=1)
st.plotly_chart(fig3, width="stretch")
