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
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Players", len(df))
m2.metric("Top Score", f"{df['score'].max():.1f}")
m3.metric("Avg Score", f"{df['score'].mean():.1f}")
m4.metric("Avg Circular Mix", f"{df['circular_mix'].mean():.1f}%")
stockout_free = int((df["stockout_rounds"] == 0).sum())
m5.metric("Stockout-Free", f"{stockout_free} / {len(df)}")
if "policy_changes" in df.columns and df["policy_changes"].notna().any():
    m6.metric("Avg Policy Changes", f"{df['policy_changes'].mean():.1f}")
else:
    m6.metric("Avg Policy Changes", "—")

st.divider()

# ── Leaderboard table ──────────────────────────────────────────────────────────
st.markdown("### Rankings")

_table_cols = ["rank", "nickname", "grade", "score",
               "circular_mix", "stockout_rounds", "cumulative_sap", "game_mode"]
_col_names   = ["#", "Nickname", "Grade", "Score (0-100)",
                "Circular Mix %", "Stockout Rounds", "Cumulative SAP ($)", "Mode"]
if "policy_changes" in df.columns:
    _table_cols.append("policy_changes")
    _col_names.append("Policy Changes")
if "scenario" in df.columns:
    _table_cols.append("scenario")
    _col_names.append("Scenario")

display_df = df[_table_cols].copy()
display_df.columns = _col_names
display_df["Cumulative SAP ($)"] = display_df["Cumulative SAP ($)"].map(
    lambda x: f"${float(x):,.0f}"
)
display_df["Mode"] = display_df["Mode"].map({
    "free_play": "Free Play",
    "primary_lock": "Primary Lock",
    "circular_lock": "Circular Challenge",
}).fillna(display_df["Mode"])
if "Scenario" in display_df.columns:
    display_df["Scenario"] = display_df["Scenario"].map({
        "base_game": "Base Game",
        "demand_surge": "Demand Surge",
        "carbon_ratchet": "Carbon Ratchet",
        "supplier_failure": "Supplier Failure",
        "known_shock": "Known Shock",
        "seasonal": "Seasonal",
    }).fillna(display_df["Scenario"])

st.dataframe(display_df, hide_index=True, width="stretch")

# ── Carbon vs SAP scatter ──────────────────────────────────────────────────────
st.markdown("### Total Carbon vs. Cumulative SAP")
carbon_available = "total_carbon" in df.columns and df["total_carbon"].notna().any()
if not carbon_available:
    st.info("Carbon data not available for this session — students may need to resubmit.")
else:
    fig_scatter = go.Figure()
    for grade_val, color in GRADE_COLOR.items():
        subset = df[df["grade"] == grade_val]
        if subset.empty:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=subset["total_carbon"],
            y=subset["cumulative_sap"],
            mode="markers+text",
            name=f"Grade {grade_val}",
            text=subset["nickname"],
            textposition="top center",
            textfont=dict(size=10, color="#8b949e"),
            marker=dict(size=14, color=color, line=dict(width=1, color="#30363d")),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Total Carbon: %{x:,.0f} kg CO₂e<br>"
                "Cumulative SAP: $%{y:,.0f}<extra></extra>"
            ),
        ))
    fig_scatter.update_layout(**PLOT_LAYOUT, title="Lower carbon, higher profit — who got there?")
    fig_scatter.update_layout(legend_title_text="Grade", legend_title_font_color="#8b949e")
    fig_scatter.update_xaxes(**_AXIS, title="Total Carbon (kg CO₂e)")
    fig_scatter.update_yaxes(**_AXIS, title="Cumulative SAP ($)")
    st.plotly_chart(fig_scatter, width="stretch")
