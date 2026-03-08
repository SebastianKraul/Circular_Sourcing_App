"""
app.py — The Loopback Initiative
Streamlit UI + round advancement controller.
"""

import random
import textwrap

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulation import (
    init_game_state,
    run_round,
    run_monte_carlo,
    find_optimal_policy,
    compute_switching_point,
    compute_sustainability_rating,
    TOTAL_ROUNDS,
    SHOCK_ROUND,
    CARBON_PRICE_NORMAL,
    CARBON_PRICE_SHOCK,
    DEFAULT_S,
    DEFAULT_S_UPPER,
    DEFAULT_MIX,
)

# ── Supabase helpers ───────────────────────────────────────────────────────────
_NICKNAME_ADJ = [
    "Swift", "Bold", "Keen", "Calm", "Sharp", "Bright", "Cool", "Dark",
    "Fast", "Wise", "Deep", "High", "Soft", "Pure", "Wild", "Clear",
]
_NICKNAME_NOUN = [
    "Falcon", "Oak", "River", "Stone", "Wolf", "Bear", "Fox", "Hawk",
    "Pine", "Lake", "Star", "Moon", "Wave", "Peak", "Vale", "Crest",
]


def _get_supabase():
    try:
        from supabase import create_client
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception:
        return None


# ── Monte Carlo cache wrappers ─────────────────────────────────────────────────
# Defined at module level so @st.cache_data persists across reruns.

@st.cache_data(show_spinner=False)
def _cached_monte_carlo(s, S, mix_pct, n_runs):
    return run_monte_carlo(s, S, mix_pct, n_runs)


@st.cache_data(show_spinner=False)
def _cached_find_optimal(mix_pct):
    return find_optimal_policy(mix_pct)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Loopback Initiative",
    page_icon="♻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [data-testid="stApp"] {
      background-color: #0d1117;
      color: #c9d1d9;
      font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }
  [data-testid="metric-container"],
  [data-testid="stExpander"],
  div.stDataFrame,
  section[data-testid="stSidebar"] {
      background-color: #161b22 !important;
      border: 1px solid #30363d;
      border-radius: 8px;
  }
  section[data-testid="stSidebar"] { background-color: #161b22 !important; }
  [data-testid="stMetricValue"] { color: #58a6ff; font-size: 1.6rem; font-weight: 700; }
  [data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
  div.stButton > button {
      background-color: #238636; color: #ffffff; border: none;
      border-radius: 6px; font-family: monospace; font-weight: 600;
  }
  div.stButton > button:hover { background-color: #2ea043; }
  div[data-testid="stAlert"] { border-radius: 8px; font-family: monospace; font-weight: 600; }
  h1, h2, h3 { color: #e6edf3; }
  .card {
      background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 1.5rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Round narratives ───────────────────────────────────────────────────────────
ROUND_NARRATIVES = {
    1: {
        "title": "Q1 — First Orders",
        "body": (
            "Your procurement dashboard is live. Inner Mongolia Mining Co. reports full operational "
            "capacity. EcoReclaim Urban Mining has confirmed your account. The board expects a "
            "sustainability strategy within the year. Carbon is priced at **$2/kg CO₂e**."
        ),
        "tip": (
            "Demand averages ~100 units/round (σ=20). You start with 150 units. "
            "Set your reorder point (s) and order-up-to (S) to avoid running dry."
        ),
    },
    2: {
        "title": "Q2 — ESG Reporting Season",
        "body": (
            "Institutional investors flagged NovaPulse's carbon footprint in their annual review. "
            "The CFO requests a cost-carbon breakdown for the board pack. Primary sourcing looks "
            "cheap on the invoice — but factor in the carbon tax."
        ),
        "tip": (
            "True unit cost at $2/kg CO₂e: **Primary = $5 + 8×$2 = $21**. "
            "**Circular = $12 + 0.5×$2 = $13**. Circular is already cheaper on a total-cost basis."
        ),
    },
    3: {
        "title": "Q3 — Yield Variation",
        "body": (
            "EcoReclaim's batch yield came in at **{yield_pct:.0%}** this quarter — {yield_comment}. "
            "Urban mining yield follows N(70%, σ=10%). If you ordered 100 units from circular, "
            "you received ~{received:.0f}. "
            "Yield risk is real — build it into your order quantity."
        ),
        "tip": (
            "Buffer rule: order ~43% more circular units than needed. "
            "To receive 70 units reliably, order ~100 from EcoReclaim."
        ),
    },
    4: {
        "title": "Q4 — Storm Warning",
        "body": (
            "Intelligence reports signal growing labour unrest at the Inner Mongolia primary mine. "
            "A trade analyst briefing warns: *'Any disruption would double lead times and stress "
            "supply chains globally.'* Check your pipeline. Are your buffers adequate?"
        ),
        "tip": (
            "If a disruption hits **next quarter**, primary orders placed NOW arrive in Round 5. "
            "Under-prepared firms will face stockouts at $20/unit penalty."
        ),
    },
    5: {
        "title": "Q5 — SUPPLY SHOCK",
        "body": (
            "A wildcat strike has shut the primary mine indefinitely. Primary lead time doubles "
            "to **2 rounds**. The Carbon Pricing Act takes effect: **$8/kg CO₂e**. Competitors "
            "over-reliant on primary sourcing are scrambling. EcoReclaim is at full capacity."
        ),
        "tip": (
            "Recalculate: **Primary = $5 + 8×$8 = $69/unit**. "
            "**Circular = $12 + 0.5×$8 = $16/unit**. "
            "P* = $0.93/kg CO₂e — the shock price of $8 is 8.6× above the switching point."
        ),
    },
    6: {
        "title": "Q6 — Adapting to the New Normal",
        "body": (
            "The strike continues with no resolution in sight. Circular capacity globally is "
            "tightening as firms pivot away from primary. EcoReclaim is prioritising partners "
            "with established relationships. Primary orders placed now arrive in **Round 8**."
        ),
        "tip": (
            "With 2-round primary lead time, you must plan two rounds ahead. "
            "Circular still arrives in 1 round. Consider raising your (s,S) targets."
        ),
    },
    7: {
        "title": "Q7 — Regulatory Tailwinds",
        "body": (
            "The EU Carbon Border Adjustment Mechanism passes into law, targeting carbon-intensive "
            "imports. Analysts expect $8+/kg CO₂e to persist for years. Circular sourcing is no "
            "longer just ethical — it is the economically rational choice."
        ),
        "tip": (
            "Your cumulative carbon tab is visible in the chart below. "
            "Compare your trajectory against a hypothetical 100% circular scenario."
        ),
    },
    8: {
        "title": "Q8 — Partial Recovery",
        "body": (
            "Mediators report progress at the mine. A partial workforce returned, but full "
            "capacity is months away. Primary reliability remains low. EcoReclaim reports "
            "strong yield conditions. **Two rounds remain** — your final pipeline decisions are critical."
        ),
        "tip": (
            "Primary orders placed in Round 8 (lead time 2) arrive Round 10. "
            "Circular orders arrive Round 9. Map out your final two rounds now."
        ),
    },
    9: {
        "title": "Q9 — Endgame",
        "body": (
            "Primary supply is recovering but carbon pricing stays at **$8/kg CO₂e**. "
            "Your sustainability grade is crystallising. The board will review SAP, "
            "circular mix percentage, and stockout record at the end of Q10."
        ),
        "tip": (
            "Circular orders placed now arrive Round 10. Primary (lead time 2) arrives Round 11 "
            "— after the simulation ends. Circular is your only in-time option."
        ),
    },
    10: {
        "title": "Q10 — Final Quarter",
        "body": (
            "The board meeting is confirmed. Shareholders, ESG auditors, and the press are watching. "
            "Any inventory ordered this round arrives **after the simulation ends** and will not "
            "count toward your score. Focus on fulfilling demand from your current pipeline."
        ),
        "tip": (
            "Avoid over-ordering — holding costs apply to end-of-round inventory. "
            "Your score is now largely determined. Make your final decisions count."
        ),
    },
}


# ── State helpers ──────────────────────────────────────────────────────────────
def _init():
    if "current_round" not in st.session_state:
        state = init_game_state(seed=42)
        for k, v in state.items():
            st.session_state[k] = v
        st.session_state["onboarding_step"] = 0
        st.session_state["onboarding_complete"] = False
        st.session_state["game_mode"] = "free_play"
        st.session_state["shock_banner_shown"] = False


def _restart():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


_init()


# ── Onboarding screens ─────────────────────────────────────────────────────────
def _step_indicator(current_step):
    steps = ["Company Brief", "Supplier Profiles", "Scoring Rubric", "Setup & Begin"]
    cols = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps)):
        if i < current_step:
            color, weight = "#3fb950", "400"
        elif i == current_step:
            color, weight = "#58a6ff", "700"
        else:
            color, weight = "#30363d", "400"
        col.markdown(
            f'<div style="text-align:center; padding:0.4rem; border-bottom:2px solid {color};">'
            f'<span style="color:{color}; font-size:0.82rem; font-weight:{weight};">'
            f'{i+1}. {label}</span></div>',
            unsafe_allow_html=True,
        )


def _nav_buttons(step, back=True):
    labels = ["Company Brief", "Supplier Profiles", "Scoring Rubric", "Setup & Begin"]
    b_col, _, n_col = st.columns([1, 3, 1])
    if back and step > 0:
        with b_col:
            if st.button("← Back", width='stretch'):
                st.session_state["onboarding_step"] = step - 1
                st.session_state["needs_scroll"] = True
                st.session_state["scroll_counter"] = st.session_state.get("scroll_counter", 0) + 1
                st.rerun()
    with n_col:
        if step < 3:
            if st.button(f"Next: {labels[step+1]} →", width='stretch'):
                st.session_state["onboarding_step"] = step + 1
                st.session_state["needs_scroll"] = True
                st.session_state["scroll_counter"] = st.session_state.get("scroll_counter", 0) + 1
                st.rerun()


def _scroll_to_top():
    """Scroll the Streamlit content area back to the top.

    Key design decisions:
    - Target: [data-testid="stMain"] is Streamlit's overflow container — the
      browser window itself does NOT scroll in a Streamlit app, so
      window.parent.scrollTo() is a no-op. We must set scrollTop on the div.
    - height=1: browsers skip script execution in iframes with no layout box.
    - st.empty() at the call site: the slot goes empty→filled→empty between
      scrolls, forcing React to unmount/remount the iframe and re-execute the
      script, rather than patching an existing iframe's srcdoc in place.
    - Multiple selectors: guards against version-to-version DOM changes.
    """
    bust = st.session_state.get("scroll_counter", 0)
    components.html(
        f"""<script>
        (function() {{
            var selectors = [
                '[data-testid="stMain"]',
                '[data-testid="stAppViewContainer"]',
                'section.main',
                '.main'
            ];
            var p = window.parent.document;
            for (var i = 0; i < selectors.length; i++) {{
                var el = p.querySelector(selectors[i]);
                if (el) {{ el.scrollTop = 0; return; }}
            }}
            window.parent.scrollTo(0, 0);
        }})();
        </script><!-- bust={bust} -->""",
        height=1,
    )


def _card(content_html, border_color="#30363d"):
    # dedent + strip removes Python indentation that Streamlit's markdown parser
    # would otherwise interpret as code blocks (4+ leading spaces = code fence).
    clean = textwrap.dedent(content_html).strip()
    st.markdown(
        f'<div style="background:#161b22; border:1px solid {border_color}; '
        f'border-radius:10px; padding:1.5rem; margin-bottom:1rem;">'
        f'{clean}</div>',
        unsafe_allow_html=True,
    )


def _card_row(*cards):
    """Render cards side by side with equal height via CSS flexbox.
    Each card is a tuple: (content_html, border_color, flex_weight).
    flex_weight controls relative width; equal weights = equal-width columns.
    align-items:stretch makes all cards the same height automatically.
    """
    parts = []
    for content_html, border_color, flex in cards:
        clean = textwrap.dedent(content_html).strip()
        parts.append(
            f'<div style="flex:{flex}; background:#161b22; border:1px solid {border_color}; '
            f'border-radius:10px; padding:1.5rem;">'
            f'{clean}</div>'
        )
    st.markdown(
        '<div style="display:flex; gap:1rem; align-items:stretch; margin-bottom:1rem;">'
        + "".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )


def _onboarding_company_brief():
    st.markdown(
        '<h1 style="color:#58a6ff; text-align:center; letter-spacing:0.1em;">'
        '♻ THE LOOPBACK INITIATIVE</h1>'
        '<p style="color:#8b949e; text-align:center; margin-bottom:2rem;">'
        'Rare Earth Magnet Supply Chain Simulation</p>',
        unsafe_allow_html=True,
    )
    _step_indicator(0)
    st.markdown("<br>", unsafe_allow_html=True)

    _card_row(
        ("""
        <h3 style="color:#58a6ff; margin-top:0;">Your Situation</h3>
        <p>You are the newly appointed <strong>Supply Chain Director</strong> at
        <strong>NovaPulse Electronics</strong> — a premium consumer electronics firm
        manufacturing high-performance audio equipment and EV motor assemblies.</p>

        <h4 style="color:#c9d1d9;">Critical Input: NdFeB Rare Earth Magnets</h4>
        <p>Neodymium-Iron-Boron magnets are the heart of your products.
        Without them, the production line stops. You source from two suppliers with
        very different cost, carbon, and risk profiles.</p>

        <h4 style="color:#c9d1d9;">Your Mandate (10 Quarters)</h4>
        <ul>
          <li>Keep production running — avoid stockouts</li>
          <li>Manage costs: procurement, holding, and carbon tax</li>
          <li>Navigate a <em>surprise supply disruption</em> in the middle of the simulation</li>
          <li>Report to an ESG-conscious board at the end</li>
        </ul>
        """, "#58a6ff", 3),
        ("""
        <h3 style="color:#d29922; margin-top:0;">What Makes This Hard</h3>
        <ul style="margin:0 0 1.2rem 0; padding-left:1.2rem;">
          <li style="margin-bottom:0.8rem;"><strong>Demand is uncertain</strong><br>
              <span style="color:#8b949e; font-size:0.9rem;">~N(100, 20²) units/quarter</span></li>
          <li style="margin-bottom:0.8rem;"><strong>You order before you know demand</strong><br>
              <span style="color:#8b949e; font-size:0.9rem;">Inventory policy is your safety net</span></li>
          <li style="margin-bottom:0.8rem;"><strong>Circular yield is variable</strong><br>
              <span style="color:#8b949e; font-size:0.9rem;">~N(70%, 10²%) — order more than you need</span></li>
          <li style="margin-bottom:0.8rem;"><strong>A disruption will occur</strong><br>
              <span style="color:#8b949e; font-size:0.9rem;">Timing and nature: unknown</span></li>
          <li><strong>Carbon pricing is active — and may change</strong><br>
              <span style="color:#8b949e; font-size:0.9rem;">Currently $2/kg CO₂e</span></li>
        </ul>
        <hr style="border-color:#30363d; margin:0 0 1rem 0;">
        <h4 style="color:#c9d1d9; margin:0 0 0.4rem 0;">Objective</h4>
        <p style="font-size:1.05rem; color:#58a6ff; font-weight:700; margin:0;">
        Maximise Sustainability-Adjusted Profit (SAP)</p>
        <p style="color:#8b949e; font-size:0.85rem; margin:0.4rem 0 0 0;">
        SAP = Revenue − Procurement − Holding − Stockout Penalties − Carbon Tax</p>
        """, "#d29922", 2),
    )

    st.markdown("<br>", unsafe_allow_html=True)
    _nav_buttons(0, back=False)


def _onboarding_supplier_cards():
    st.markdown(
        '<h2 style="color:#e6edf3; text-align:center;">Supplier Profiles</h2>'
        '<p style="color:#8b949e; text-align:center; margin-bottom:2rem;">'
        'You have two sourcing options. Each order is split between them by your chosen mix percentage.</p>',
        unsafe_allow_html=True,
    )
    _step_indicator(1)
    st.markdown("<br>", unsafe_allow_html=True)

    _card_row(
        ("""
        <h3 style="color:#58a6ff; margin-top:0;">Primary Mining Co.</h3>
        <p style="color:#8b949e; font-size:0.85rem; margin-top:-0.5rem;">Inner Mongolia, China</p>
        <table style="width:100%; border-collapse:collapse; font-family:monospace;">
          <tr><td style="color:#8b949e; padding:5px 0;">Unit cost</td>
              <td style="color:#c9d1d9; text-align:right; font-weight:700;">$5.00</td></tr>
          <tr><td style="color:#8b949e; padding:5px 0;">Carbon intensity</td>
              <td style="color:#f85149; text-align:right; font-weight:700;">8.0 kg CO₂e/unit</td></tr>
          <tr><td style="color:#8b949e; padding:5px 0;">Lead time (normal)</td>
              <td style="color:#c9d1d9; text-align:right;">1 round</td></tr>
          <tr><td style="color:#8b949e; padding:5px 0;">Yield</td>
              <td style="color:#3fb950; text-align:right;">100% (reliable)</td></tr>
          <tr><td style="color:#8b949e; padding:5px 0;">Geopolitical risk</td>
              <td style="color:#d29922; text-align:right;">HIGH</td></tr>
        </table>
        <hr style="border-color:#30363d; margin:1rem 0;">
        <p style="color:#8b949e; font-size:0.85rem; margin:0;">
        Cheap and reliable under normal conditions.
        Highly carbon-intensive. Exposed to regulatory and geopolitical disruption.
        At <strong style="color:#c9d1d9;">$2/kg CO₂e</strong> carbon price, true cost =
        <strong style="color:#58a6ff;">$21/unit</strong>.</p>
        """, "#58a6ff", 1),
        ("""
        <h3 style="color:#3fb950; margin-top:0;">EcoReclaim Urban Mining</h3>
        <p style="color:#8b949e; font-size:0.85rem; margin-top:-0.5rem;">Distributed EU/NA Facilities</p>
        <table style="width:100%; border-collapse:collapse; font-family:monospace;">
          <tr><td style="color:#8b949e; padding:5px 0;">Unit cost</td>
              <td style="color:#c9d1d9; text-align:right; font-weight:700;">$12.00</td></tr>
          <tr><td style="color:#8b949e; padding:5px 0;">Carbon intensity</td>
              <td style="color:#3fb950; text-align:right; font-weight:700;">0.5 kg CO₂e/unit</td></tr>
          <tr><td style="color:#8b949e; padding:5px 0;">Lead time</td>
              <td style="color:#c9d1d9; text-align:right;">1 round</td></tr>
          <tr><td style="color:#8b949e; padding:5px 0;">Yield</td>
              <td style="color:#d29922; text-align:right;">~N(70%, σ=10%)</td></tr>
          <tr><td style="color:#8b949e; padding:5px 0;">Geopolitical risk</td>
              <td style="color:#3fb950; text-align:right;">LOW</td></tr>
        </table>
        <hr style="border-color:#30363d; margin:1rem 0;">
        <p style="color:#8b949e; font-size:0.85rem; margin:0;">
        Higher unit price but 16× lower carbon. Yield varies each round —
        if you order 100 units, expect to receive ~70 (with variation).
        <strong style="color:#d29922;">Always order more than you need</strong> to buffer yield loss.
        True cost at $2/kg CO₂e = <strong style="color:#3fb950;">$13/unit</strong>.</p>
        """, "#3fb950", 1),
    )

    st.markdown("<br>", unsafe_allow_html=True)
    _card("""
    <h4 style="color:#d29922; margin-top:0;">Key Insight: The Switching Point (P*)</h4>
    <p style="margin:0; color:#c9d1d9;">
    There exists a carbon price P* at which circular sourcing becomes cheaper than primary
    on a <em>total cost</em> basis (procurement + carbon). At any carbon price above P*,
    circular is the economically rational choice — regardless of ESG considerations.
    </p>
    <p style="margin:0.5rem 0 0 0; color:#8b949e; font-size:0.85rem;">
    Can you figure out P* before the simulation reveals it? Hint: it depends only on the
    cost and carbon numbers shown above.
    </p>
    """, border_color="#d29922")

    _nav_buttons(1)


def _onboarding_scoring_rubric():
    st.markdown(
        '<h2 style="color:#e6edf3; text-align:center;">Scoring Rubric</h2>'
        '<p style="color:#8b949e; text-align:center; margin-bottom:2rem;">'
        'How your performance is evaluated at the end of 10 rounds.</p>',
        unsafe_allow_html=True,
    )
    _step_indicator(2)
    st.markdown("<br>", unsafe_allow_html=True)

    _card_row(
        ("""
        <h3 style="color:#58a6ff; margin-top:0;">How Your Score Is Calculated</h3>
        <p style="font-family:monospace; background:#0d1117; padding:0.8rem;
                  border-radius:6px; font-size:0.9rem; color:#c9d1d9;">
          SAP = Revenue<br>
          &nbsp;&nbsp;&nbsp;− Primary Procurement Cost<br>
          &nbsp;&nbsp;&nbsp;− Circular Procurement Cost<br>
          &nbsp;&nbsp;&nbsp;− Holding Cost (end-of-round inventory × $1)<br>
          &nbsp;&nbsp;&nbsp;− Stockout Penalty (shortfall × $20)<br>
          &nbsp;&nbsp;&nbsp;− Carbon Tax (kg CO₂e × carbon price)
        </p>
        <table style="width:100%; border-collapse:collapse; font-family:monospace; margin-top:0.5rem;">
          <tr><td style="color:#8b949e; padding:4px 0;">Revenue</td>
              <td style="color:#3fb950; text-align:right;">$50 / unit sold</td></tr>
          <tr><td style="color:#8b949e; padding:4px 0;">Holding cost</td>
              <td style="color:#c9d1d9; text-align:right;">$1 / unit / round</td></tr>
          <tr><td style="color:#8b949e; padding:4px 0;">Stockout penalty</td>
              <td style="color:#f85149; text-align:right;">$20 / unit short</td></tr>
          <tr><td style="color:#8b949e; padding:4px 0;">Carbon price (now)</td>
              <td style="color:#c9d1d9; text-align:right;">$2 / kg CO₂e</td></tr>
          <tr><td style="color:#8b949e; padding:4px 0;">Carbon price (may change)</td>
              <td style="color:#d29922; text-align:right;">?</td></tr>
        </table>
        <hr style="border-color:#30363d; margin:1rem 0;">
        <h3 style="color:#3fb950; margin-top:0;">Final Grade</h3>
        <table style="width:100%; border-collapse:collapse; font-family:monospace;">
          <tr style="border-bottom:1px solid #30363d;">
            <th style="color:#8b949e; padding:6px 0; text-align:left;">Component</th>
            <th style="color:#8b949e; padding:6px 0; text-align:right;">Weight</th>
          </tr>
          <tr><td style="color:#c9d1d9; padding:6px 0;">Sustainability-Adjusted Profit</td>
              <td style="color:#58a6ff; text-align:right; font-weight:700;">50%</td></tr>
          <tr><td style="color:#c9d1d9; padding:6px 0;">Average Circular Sourcing Mix</td>
              <td style="color:#3fb950; text-align:right; font-weight:700;">30%</td></tr>
          <tr><td style="color:#c9d1d9; padding:6px 0;">Stockout-Free Rounds</td>
              <td style="color:#d29922; text-align:right; font-weight:700;">20%</td></tr>
        </table>
        <hr style="border-color:#30363d; margin:0.8rem 0;">
        <table style="width:100%; border-collapse:collapse; font-family:monospace;">
          <tr><th style="color:#8b949e; text-align:left; padding:3px 0;">Grade</th>
              <th style="color:#8b949e; text-align:right; padding:3px 0;">Score</th></tr>
          <tr><td style="color:#ffd700; font-weight:700; padding:3px 0;">S</td><td style="text-align:right; color:#c9d1d9;">≥ 90</td></tr>
          <tr><td style="color:#3fb950; font-weight:700; padding:3px 0;">A</td><td style="text-align:right; color:#c9d1d9;">75 – 89</td></tr>
          <tr><td style="color:#58a6ff; font-weight:700; padding:3px 0;">B</td><td style="text-align:right; color:#c9d1d9;">60 – 74</td></tr>
          <tr><td style="color:#d29922; font-weight:700; padding:3px 0;">C</td><td style="text-align:right; color:#c9d1d9;">40 – 59</td></tr>
          <tr><td style="color:#f85149; font-weight:700; padding:3px 0;">D</td><td style="text-align:right; color:#c9d1d9;">< 40</td></tr>
        </table>
        """, "#58a6ff", 1),
        ("""
        <h3 style="color:#d29922; margin-top:0;">How to Play Well</h3>
        <h4 style="color:#c9d1d9; margin:0 0 0.4rem 0;">(s,S) Inventory Policy</h4>
        <ul style="color:#c9d1d9; margin:0 0 0.8rem 0; padding-left:1.2rem;">
          <li style="margin-bottom:0.5rem;"><strong style="color:#d29922;">s — Reorder Point</strong>:
              When inventory falls to or below s, you place an order.</li>
          <li><strong style="color:#d29922;">S — Order-Up-To Level</strong>:
              You order enough to bring inventory back up to S, split by your sourcing mix.</li>
        </ul>
        <p style="color:#8b949e; font-size:0.85rem; margin:0 0 1rem 0;">
        Order quantity = max(0, S − inventory), triggered only when inventory ≤ s.
        Manual overrides are available each round.
        </p>
        <hr style="border-color:#30363d; margin:0 0 1rem 0;">
        <h4 style="color:#f85149; margin:0 0 0.6rem 0;">Four Ways to Lose SAP</h4>
        <ul style="color:#c9d1d9; margin:0; padding-left:1.2rem;">
          <li style="margin-bottom:0.6rem;"><strong>Stockouts</strong> — $20/unit penalty plus
              lost revenue. One bad round can erase several good ones.</li>
          <li style="margin-bottom:0.6rem;"><strong>Over-stocking</strong> — Excess inventory
              costs $1/unit/round and ties up capital.</li>
          <li style="margin-bottom:0.6rem;"><strong>Carbon exposure</strong> — Carbon prices
              can change mid-game. A primary-heavy buffer looks very different at a higher tax rate.</li>
          <li><strong>Pipeline blindness</strong> — Orders take 1–2 rounds to arrive.
              What's in transit matters as much as what's on the shelf.</li>
        </ul>
        """, "#d29922", 1),
    )

    st.markdown("<br>", unsafe_allow_html=True)
    _nav_buttons(2)


def _onboarding_setup():
    st.markdown(
        '<h2 style="color:#e6edf3; text-align:center;">Configure Your Strategy</h2>'
        '<p style="color:#8b949e; text-align:center; margin-bottom:2rem;">'
        'Set your inventory policy and sourcing approach before the simulation begins. '
        'These can be adjusted each round, but your starting choices matter.</p>',
        unsafe_allow_html=True,
    )
    _step_indicator(3)
    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("### Game Mode")
        game_mode = st.radio(
            "Select a scenario:",
            options=["free_play", "primary_lock", "circular_lock"],
            format_func=lambda x: {
                "free_play": "Free Play — All decisions are yours",
                "primary_lock": "Primary Lock — 100% primary sourcing (carbon exposure study)",
                "circular_lock": "Circular Challenge — 100% circular sourcing (yield risk study)",
            }[x],
            index=["free_play", "primary_lock", "circular_lock"].index(
                st.session_state.get("game_mode", "free_play")
            ),
            key="setup_game_mode",
        )
        mode_descriptions = {
            "free_play": "You control everything: reorder points, order-up-to levels, and sourcing mix each round.",
            "primary_lock": "Sourcing is locked to 100% primary. Focus on inventory policy and observe your carbon footprint grow — especially if the carbon price changes.",
            "circular_lock": "Sourcing is locked to 100% circular. Learn to manage yield uncertainty and higher per-unit costs while keeping carbon low.",
        }
        st.info(mode_descriptions[game_mode])

        st.markdown("### Sourcing Mix")
        if game_mode == "primary_lock":
            st.markdown(
                '<div style="background:#161b22; border:1px solid #58a6ff; border-radius:6px; '
                'padding:0.8rem; color:#58a6ff; font-weight:700;">LOCKED: 100% Primary</div>',
                unsafe_allow_html=True,
            )
            setup_mix = 100
        elif game_mode == "circular_lock":
            st.markdown(
                '<div style="background:#161b22; border:1px solid #3fb950; border-radius:6px; '
                'padding:0.8rem; color:#3fb950; font-weight:700;">LOCKED: 100% Circular</div>',
                unsafe_allow_html=True,
            )
            setup_mix = 0
        else:
            setup_mix = st.slider(
                "Primary % (remainder goes to Circular)",
                min_value=0, max_value=100, value=DEFAULT_MIX, step=5,
                key="setup_mix",
            )
            c1, c2 = st.columns(2)
            c1.metric("Primary", f"{setup_mix}%")
            c2.metric("Circular", f"{100-setup_mix}%")

    with right:
        st.markdown("### Inventory Policy (s, S)")
        _card("""
        <p style="color:#8b949e; font-size:0.85rem; margin:0;">
        <strong style="color:#d29922;">s</strong> = Reorder Point.
        When inventory hits s or below, you order.<br><br>
        <strong style="color:#d29922;">S</strong> = Order-Up-To Level.
        You order (S − inventory) units, split by your sourcing mix.<br><br>
        <strong>Think carefully:</strong> Demand averages 100 units/round.
        Starting inventory is 150. How low is too low? How high is too high?
        </p>
        """, border_color="#d29922")

        setup_s = st.number_input(
            "s — Reorder Point",
            min_value=0, max_value=500,
            value=DEFAULT_S,
            step=5,
            key="setup_s",
            help="Suggested starting range: 50-120. Too low → stockout risk. Too high → frequent small orders.",
        )
        setup_S = st.number_input(
            "S — Order-Up-To Level",
            min_value=0, max_value=1000,
            value=DEFAULT_S_UPPER,
            step=5,
            key="setup_S",
            help="Suggested starting range: 120-250. Too low → thin buffer. Too high → excess holding costs.",
        )

        if setup_s >= setup_S:
            st.error("s must be strictly less than S.")
            setup_valid = False
        else:
            setup_valid = True
            implied_order = max(0, setup_S - setup_s)
            st.success(
                f"When inventory hits {setup_s}, you'll order up to {implied_order} units "
                f"(split {setup_mix}% / {100-setup_mix}% Primary/Circular)."
            )

    st.markdown("<br>", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        if st.button(
            "BEGIN SIMULATION →",
            disabled=not setup_valid,
            width='stretch',
        ):
            # Apply setup choices to game state
            state = init_game_state(seed=42)
            state["s_reorder_point"] = setup_s
            state["S_order_up_to"] = setup_S
            state["sourcing_mix_pct"] = setup_mix
            for k, v in state.items():
                st.session_state[k] = v
            st.session_state["game_mode"] = game_mode
            st.session_state["onboarding_complete"] = True
            st.session_state["shock_banner_shown"] = False
            st.rerun()

    _nav_buttons(3)


def _show_onboarding():
    _scroll_slot = st.empty()
    if st.session_state.get("needs_scroll"):
        st.session_state["needs_scroll"] = False
        with _scroll_slot:
            _scroll_to_top()
    step = st.session_state["onboarding_step"]
    if step == 0:
        _onboarding_company_brief()
    elif step == 1:
        _onboarding_supplier_cards()
    elif step == 2:
        _onboarding_scoring_rubric()
    elif step == 3:
        _onboarding_setup()


# ── Run onboarding if not complete ────────────────────────────────────────────
if not st.session_state.get("onboarding_complete", False):
    _show_onboarding()
    st.stop()


# ── Sidebar (active game only) ─────────────────────────────────────────────────
game_mode = st.session_state.get("game_mode", "free_play")

with st.sidebar:
    st.markdown("## ♻ The Loopback Initiative")
    st.caption("Rare Earth Magnet Supply Chain Simulation")
    st.divider()

    rnd = st.session_state["current_round"]
    progress_pct = min(1.0, (rnd - 1) / TOTAL_ROUNDS)
    st.markdown(f"**Round {min(rnd, TOTAL_ROUNDS)} / {TOTAL_ROUNDS}**")
    st.progress(progress_pct)

    mode_label = {"free_play": "Free Play", "primary_lock": "Primary Lock",
                  "circular_lock": "Circular Challenge"}.get(game_mode, game_mode)
    st.caption(f"Mode: {mode_label}")
    st.divider()

    st.markdown("### (s,S) Inventory Policy")
    s_val = st.number_input(
        "s — Reorder Point",
        min_value=0, max_value=500,
        value=int(st.session_state["s_reorder_point"]),
        step=5,
        disabled=st.session_state["game_over"],
        key="sidebar_s",
    )
    S_val = st.number_input(
        "S — Order-Up-To Level",
        min_value=0, max_value=1000,
        value=int(st.session_state["S_order_up_to"]),
        step=5,
        disabled=st.session_state["game_over"],
        key="sidebar_S",
    )

    if s_val >= S_val:
        st.error("s must be strictly less than S.")
        inputs_valid = False
    else:
        inputs_valid = True
        st.session_state["s_reorder_point"] = s_val
        st.session_state["S_order_up_to"] = S_val

    st.divider()

    # Sourcing mix — locked for non-free-play modes
    st.markdown("### Sourcing Mix")
    if game_mode == "primary_lock":
        st.markdown(
            '<div style="background:#161b22; border:1px solid #58a6ff; border-radius:6px; '
            'padding:0.5rem 0.8rem; color:#58a6ff; font-size:0.9rem;">LOCKED: 100% Primary</div>',
            unsafe_allow_html=True,
        )
        mix_pct = 100
        st.session_state["sourcing_mix_pct"] = 100
    elif game_mode == "circular_lock":
        st.markdown(
            '<div style="background:#161b22; border:1px solid #3fb950; border-radius:6px; '
            'padding:0.5rem 0.8rem; color:#3fb950; font-size:0.9rem;">LOCKED: 100% Circular</div>',
            unsafe_allow_html=True,
        )
        mix_pct = 0
        st.session_state["sourcing_mix_pct"] = 0
    else:
        mix_pct = st.slider(
            "Primary % (remainder = Circular)",
            min_value=0, max_value=100,
            value=int(st.session_state["sourcing_mix_pct"]),
            step=5,
            disabled=st.session_state["game_over"],
            key="sidebar_mix",
        )
        st.session_state["sourcing_mix_pct"] = mix_pct
        col_p, col_c = st.columns(2)
        col_p.metric("Primary", f"{mix_pct}%")
        col_c.metric("Circular", f"{100-mix_pct}%")

    st.divider()

    with st.expander("Manual Order Override (optional)"):
        st.caption("Leave at 0 to use (s,S) policy.")
        man_primary = st.number_input(
            "Primary units to order", min_value=0, max_value=2000,
            value=0, step=10, key="man_primary",
            disabled=st.session_state["game_over"],
        )
        man_circular = st.number_input(
            "Circular units to order (before yield)", min_value=0, max_value=2000,
            value=0, step=10, key="man_circular",
            disabled=st.session_state["game_over"],
        )
        use_override = st.checkbox(
            "Apply manual override this round",
            value=False, key="use_override",
            disabled=st.session_state["game_over"],
        )

    st.divider()

    if not st.session_state["game_over"]:
        next_label = rnd + 1 if rnd < TOTAL_ROUNDS else "End"
        advance = st.button(
            f"▶  Advance to Round {next_label}",
            disabled=not inputs_valid,
            width='stretch',
        )
    else:
        advance = False

    if st.button("↺  Restart Game", width='stretch'):
        _restart()


# ── Round advancement ──────────────────────────────────────────────────────────
GAME_KEYS = {
    "current_round", "game_over", "shock_triggered", "inventory",
    "pipeline", "s_reorder_point", "S_order_up_to", "sourcing_mix_pct",
    "history", "cumulative_sap", "cumulative_carbon", "rng",
    "manual_primary_override", "manual_circular_override",
}

if advance and inputs_valid and not st.session_state["game_over"]:
    if use_override:
        st.session_state["manual_primary_override"] = man_primary if man_primary > 0 else None
        st.session_state["manual_circular_override"] = man_circular if man_circular > 0 else None
    else:
        st.session_state["manual_primary_override"] = None
        st.session_state["manual_circular_override"] = None

    game_state = {k: st.session_state[k] for k in GAME_KEYS if k in st.session_state}
    new_state = run_round(game_state)
    for k, v in new_state.items():
        st.session_state[k] = v

    if st.session_state["current_round"] > SHOCK_ROUND and not st.session_state.get("shock_banner_shown"):
        st.session_state["shock_banner_shown"] = True

    st.session_state["needs_scroll"] = True
    st.session_state["scroll_counter"] = st.session_state.get("scroll_counter", 0) + 1
    st.rerun()


# ── Plotly layout defaults ─────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(family="monospace", color="#c9d1d9", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", tickmode="linear", dtick=1),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
)


# ── FINAL SCORE SCREEN ─────────────────────────────────────────────────────────
if st.session_state["game_over"]:
    history = st.session_state["history"]
    cumulative_sap = st.session_state["cumulative_sap"]
    score, grade = compute_sustainability_rating(history, cumulative_sap)
    p_star = compute_switching_point()

    grade_color = {"S": "#ffd700", "A": "#3fb950", "B": "#58a6ff",
                   "C": "#d29922", "D": "#f85149"}.get(grade, "#8b949e")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #161b22, #21262d);
                border: 1px solid #58a6ff; border-radius: 12px;
                padding: 2rem; text-align: center; margin-bottom: 1.5rem;">
      <h1 style="color:#58a6ff; margin:0; font-size:2.5rem; letter-spacing:0.1em;">
        SIMULATION COMPLETE
      </h1>
      <p style="color:#8b949e; margin-top:0.5rem; font-size:1rem;">
        NovaPulse Electronics — 10-Quarter Review
      </p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    total_carbon = sum(h["total_carbon"] for h in history)
    m1.metric("Cumulative SAP", f"${cumulative_sap:,.0f}")
    m2.metric("Sustainability Grade", grade)
    m3.metric("Score", f"{score:.1f} / 100")
    m4.metric("Total Carbon", f"{total_carbon:,.0f} kg CO₂e")

    st.divider()

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### P&L Summary")
        total_revenue = sum(h["revenue"] for h in history)
        total_cost_p = sum(h["cost_primary"] for h in history)
        total_cost_c = sum(h["cost_circular"] for h in history)
        total_holding = sum(h["cost_holding"] for h in history)
        total_stockout = sum(h["cost_stockout"] for h in history)
        total_carbon_cost = sum(h["cost_carbon"] for h in history)
        total_costs = total_cost_p + total_cost_c + total_holding + total_stockout + total_carbon_cost

        pl_df = pd.DataFrame({
            "Item": ["Revenue", "  Primary Procurement", "  Circular Procurement",
                     "  Holding Cost", "  Stockout Penalty", "  Carbon Tax",
                     "Total Costs", "Net SAP"],
            "Amount ($)": [total_revenue, -total_cost_p, -total_cost_c,
                           -total_holding, -total_stockout, -total_carbon_cost,
                           -total_costs, cumulative_sap],
        })
        pl_df["Amount ($)"] = pl_df["Amount ($)"].map(lambda x: f"${x:,.0f}")
        st.dataframe(pl_df, hide_index=True, width='stretch')

    with right_col:
        st.markdown("### Sustainability Scorecard")
        total_ordered = sum(h["order_primary"] + h["order_circular"] for h in history)
        total_circ_ordered = sum(h["order_circular"] for h in history)
        circ_pct = (total_circ_ordered / total_ordered * 100) if total_ordered > 0 else 0
        stockout_rounds = sum(1 for h in history if h["stockout_units"] > 0)

        shock_carbon_price = CARBON_PRICE_SHOCK
        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1rem;">
          <table style="width:100%; border-collapse:collapse; font-family:monospace;">
            <tr><td style="color:#8b949e; padding:4px 0;">Sustainability Grade</td>
                <td style="color:{grade_color}; font-weight:700; font-size:1.4rem; text-align:right;">{grade}</td></tr>
            <tr><td style="color:#8b949e; padding:4px 0;">Composite Score</td>
                <td style="color:#c9d1d9; text-align:right;">{score:.1f} / 100</td></tr>
            <tr><td style="color:#8b949e; padding:4px 0;">Avg Circular Mix</td>
                <td style="color:#3fb950; text-align:right;">{circ_pct:.1f}%</td></tr>
            <tr><td style="color:#8b949e; padding:4px 0;">Stockout Rounds</td>
                <td style="color:#{'f85149' if stockout_rounds > 0 else '3fb950'}; text-align:right;">{stockout_rounds} / {TOTAL_ROUNDS}</td></tr>
            <tr><td style="color:#8b949e; padding:4px 0;">Total Carbon</td>
                <td style="color:#c9d1d9; text-align:right;">{total_carbon:,.0f} kg CO₂e</td></tr>
          </table>
          <hr style="border-color:#30363d; margin:0.8rem 0;">
          <p style="color:#8b949e; font-size:0.82rem; margin:0;">
            <strong style="color:#58a6ff;">P* = ${p_star:.2f}/kg CO₂e</strong>
            — the carbon price at which circular becomes cheaper than primary.<br><br>
            The shock carbon price of <strong>${shock_carbon_price:.0f}/kg CO₂e</strong> is
            <strong style="color:#3fb950;">{shock_carbon_price/p_star:.1f}×</strong> above P*.
            From Round {SHOCK_ROUND} onward, circular sourcing was strongly cost-justified
            even before ESG considerations.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### SAP Trajectory")
    df_hist = pd.DataFrame(history)
    fig_sap = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sap.add_trace(
        go.Bar(x=df_hist["round"], y=df_hist["round_profit"], name="Round Profit",
               marker_color=["#f85149" if r >= SHOCK_ROUND else "#58a6ff" for r in df_hist["round"]],
               opacity=0.85),
        secondary_y=False,
    )
    fig_sap.add_trace(
        go.Scatter(x=df_hist["round"], y=df_hist["cumulative_sap"], name="Cumulative SAP",
                   line=dict(color="#3fb950", width=2), mode="lines+markers"),
        secondary_y=True,
    )
    fig_sap.add_vline(x=SHOCK_ROUND - 0.5, line_dash="dash", line_color="#f85149", opacity=0.6,
                      annotation_text="SHOCK", annotation_font_color="#f85149")
    fig_sap.update_layout(**PLOT_LAYOUT, title="Round Profit & Cumulative SAP",
                          xaxis_title="Round")
    fig_sap.update_yaxes(title_text="Round Profit ($)", secondary_y=False,
                          gridcolor="#21262d", zerolinecolor="#30363d")
    fig_sap.update_yaxes(title_text="Cumulative SAP ($)", secondary_y=True,
                          gridcolor="#21262d")
    st.plotly_chart(fig_sap, width='stretch')

    col_donut, col_cost = st.columns(2)
    with col_donut:
        st.markdown("### Carbon Attribution")
        total_carbon_p = sum(h["carbon_primary"] for h in history)
        total_carbon_c = sum(h["carbon_circular"] for h in history)
        fig_donut = go.Figure(go.Pie(
            labels=["Primary", "Circular"],
            values=[total_carbon_p, total_carbon_c],
            hole=0.55,
            marker=dict(colors=["#f85149", "#3fb950"]),
            textfont=dict(family="monospace", color="#c9d1d9"),
        ))
        fig_donut.update_layout(**PLOT_LAYOUT, title="Total Carbon by Source (kg CO₂e)")
        st.plotly_chart(fig_donut, width='stretch')

    with col_cost:
        st.markdown("### Cost Mix")
        cost_labels = ["Primary", "Circular", "Holding", "Stockout", "Carbon Tax"]
        cost_values = [total_cost_p, total_cost_c, total_holding, total_stockout, total_carbon_cost]
        cost_colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#a371f7"]
        fig_cost_pie = go.Figure(go.Pie(
            labels=cost_labels, values=cost_values, hole=0.55,
            marker=dict(colors=cost_colors),
            textfont=dict(family="monospace", color="#c9d1d9"),
        ))
        fig_cost_pie.update_layout(**PLOT_LAYOUT, title="Total Cost Breakdown")
        st.plotly_chart(fig_cost_pie, width='stretch')

    st.divider()
    st.markdown("### Debrief Discussion Questions")
    questions = [
        ("1. The Switching Point",
         f"P* = **${p_star:.2f}/kg CO₂e**. The shock carbon price of ${CARBON_PRICE_SHOCK:.0f}/kg "
         f"is {CARBON_PRICE_SHOCK/p_star:.1f}× above P*. When did circular sourcing become "
         f"economically rational — and did your strategy reflect this in time?"),
        ("2. Pipeline Risk & Lead Times",
         "When the shock hit in Round 5, primary lead time doubled to 2 rounds. How did the "
         "orders already in your pipeline shape your resilience? What would you pre-position "
         "differently if you played again?"),
        ("3. Yield Uncertainty & Over-ordering",
         "EcoReclaim's yield varied ~N(70%, 10%) each round. Did you build a yield buffer into "
         "your circular orders? What systematic approach could reduce the risk of under-receiving?"),
        ("4. (s,S) Policy Design",
         "Reflect on your reorder point (s) and order-up-to (S). Were they calibrated for "
         "normal demand variability? Did you adjust them after the shock changed lead times "
         "and economics? What would an optimal policy look like?"),
        ("5. Carbon vs. Cost Trade-off",
         "Did you prioritise SAP maximisation or carbon minimisation? How would a stricter "
         "carbon budget constraint change your sourcing mix? At what carbon price would you "
         "switch entirely to circular?"),
    ]
    for title, body in questions:
        with st.expander(title):
            st.markdown(body)

    # ── Policy Stress Test ──────────────────────────────────────────────────────
    st.divider()
    with st.expander("Policy Stress Test — Monte Carlo Analysis", expanded=False):
        student_s   = int(st.session_state["s_reorder_point"])
        student_S   = int(st.session_state["S_order_up_to"])
        student_mix = int(st.session_state["sourcing_mix_pct"])

        st.markdown(
            f"Tests your final policy (**s={student_s}, S={student_S}, "
            f"{student_mix}% primary**) across 1,000 random demand/yield scenarios, "
            f"then finds the near-optimal (s,S) for that mix via grid search (~120 "
            f"combinations × 150 runs each)."
        )

        if st.button("Run Analysis", key="run_mc_btn"):
            st.session_state["mc_done"] = True

        if st.session_state.get("mc_done"):
            with st.spinner("Searching for near-optimal policy and running simulations…"):
                opt_s, opt_S, _ = _cached_find_optimal(student_mix)
                student_saps, student_so = _cached_monte_carlo(
                    student_s, student_S, student_mix, 1000)
                optimal_saps, optimal_so = _cached_monte_carlo(
                    opt_s, opt_S, student_mix, 1000)
                default_saps, default_so = _cached_monte_carlo(
                    DEFAULT_S, DEFAULT_S_UPPER, student_mix, 1000)

            # Percentile of actual result within student's own distribution
            actual_pct = float((student_saps < cumulative_sap).mean() * 100)
            sap_gap    = float(optimal_saps.mean() - student_saps.mean())
            so_prob    = float((student_so > 0).mean() * 100)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(
                "Your result vs. your policy",
                f"{actual_pct:.0f}th pct",
                help="Where your actual game result falls within the distribution "
                     "of 1,000 runs with identical (s,S) settings.",
            )
            mc2.metric(
                "P(stockout ≥ 1 round)",
                f"{so_prob:.0f}%",
                help="Fraction of simulated games where your policy produced at "
                     "least one stockout round.",
            )
            mc3.metric(
                "Mean SAP gap to near-optimal",
                f"${sap_gap:,.0f}",
                help="How much more the near-optimal policy earns on average "
                     "compared to your policy.",
            )

            # Histogram
            fig_mc = go.Figure()
            for label, saps, color in [
                (f"Default  (s={DEFAULT_S}, S={DEFAULT_S_UPPER})", default_saps,  "#f85149"),
                (f"Your policy  (s={student_s}, S={student_S})",   student_saps,  "#58a6ff"),
                (f"Near-optimal  (s={opt_s}, S={opt_S})",          optimal_saps,  "#3fb950"),
            ]:
                fig_mc.add_trace(go.Histogram(
                    x=saps, name=label,
                    histnorm="probability",
                    opacity=0.65,
                    marker_color=color,
                    nbinsx=40,
                ))
            fig_mc.add_vline(
                x=cumulative_sap, line_dash="dash", line_color="#d29922",
                annotation_text="Your actual result",
                annotation_font_color="#d29922",
            )
            fig_mc.update_layout(
                **PLOT_LAYOUT,
                barmode="overlay",
                title="SAP Distribution across 1,000 Random Scenarios",
            )
            fig_mc.update_xaxes(title="Cumulative SAP ($)")
            fig_mc.update_yaxes(title="Probability", tickformat=".0%")
            st.plotly_chart(fig_mc, width="stretch")

            # Summary stats table
            def _stats(saps, stockouts):
                return {
                    "Mean SAP ($)":          f"${saps.mean():,.0f}",
                    "10th pct ($)":          f"${np.percentile(saps, 10):,.0f}",
                    "90th pct ($)":          f"${np.percentile(saps, 90):,.0f}",
                    "P(≥1 stockout round)":  f"{(stockouts > 0).mean():.0%}",
                    "Avg stockout rounds":   f"{stockouts.mean():.2f}",
                }

            stats_df = pd.DataFrame({
                f"Default (s={DEFAULT_S}, S={DEFAULT_S_UPPER})": _stats(default_saps, default_so),
                f"Your policy (s={student_s}, S={student_S})":   _stats(student_saps, student_so),
                f"Near-optimal (s={opt_s}, S={opt_S})":          _stats(optimal_saps, optimal_so),
            }).T.reset_index()
            stats_df.columns = ["Policy"] + list(stats_df.columns[1:])
            st.dataframe(stats_df, hide_index=True, width="stretch")

            st.info(
                f"Near-optimal policy for **{student_mix}% primary / "
                f"{100 - student_mix}% circular**: "
                f"**s = {opt_s}**, **S = {opt_S}**  "
                f"(found via grid search — step size 25, so true optimum may differ slightly)"
            )

    # ── Score submission ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Submit Your Score")
    _supabase = _get_supabase()
    if _supabase is None:
        st.info("Score submission is not configured for this deployment.")
    elif st.session_state.get("score_submitted"):
        st.success(
            f"Score submitted as **{st.session_state.get('submitted_nickname', 'anonymous')}**. "
            "Check the leaderboard to see how you rank."
        )
    else:
        if "suggested_nickname" not in st.session_state:
            st.session_state["suggested_nickname"] = (
                random.choice(_NICKNAME_ADJ) + random.choice(_NICKNAME_NOUN)
            )
        sub_l, sub_r = st.columns(2)
        with sub_l:
            session_code = st.text_input(
                "Session code",
                value=st.session_state.get("_session_code", ""),
                placeholder="Ask your instructor",
                key="session_code_input",
            )
            nickname = st.text_input(
                "Your nickname (stays anonymous)",
                value=st.session_state["suggested_nickname"],
                key="nickname_input",
            )
        with sub_r:
            _card(f"""
            <h4 style="color:#58a6ff; margin-top:0;">What will be submitted</h4>
            <table style="width:100%; border-collapse:collapse; font-family:monospace;">
              <tr><td style="color:#8b949e; padding:3px 0;">Nickname</td>
                  <td style="color:#c9d1d9; text-align:right;">{nickname or "—"}</td></tr>
              <tr><td style="color:#8b949e; padding:3px 0;">Grade</td>
                  <td style="color:{grade_color}; font-weight:700; text-align:right;">{grade}</td></tr>
              <tr><td style="color:#8b949e; padding:3px 0;">Score</td>
                  <td style="color:#c9d1d9; text-align:right;">{score:.1f} / 100</td></tr>
              <tr><td style="color:#8b949e; padding:3px 0;">Circular Mix</td>
                  <td style="color:#3fb950; text-align:right;">{circ_pct:.1f}%</td></tr>
              <tr><td style="color:#8b949e; padding:3px 0;">Stockout Rounds</td>
                  <td style="color:#c9d1d9; text-align:right;">{stockout_rounds}</td></tr>
              <tr><td style="color:#8b949e; padding:3px 0;">Cumulative SAP</td>
                  <td style="color:#c9d1d9; text-align:right;">${cumulative_sap:,.0f}</td></tr>
            </table>
            """, border_color="#58a6ff")
        submit_disabled = not session_code.strip() or not nickname.strip()
        if st.button("Submit Score →", disabled=submit_disabled, width='stretch'):
            try:
                _supabase.table("scores").insert({
                    "session": session_code.strip(),
                    "nickname": nickname.strip(),
                    "grade": grade,
                    "score": round(score, 1),
                    "circular_mix": round(circ_pct, 1),
                    "stockout_rounds": stockout_rounds,
                    "cumulative_sap": round(cumulative_sap, 0),
                    "total_carbon": round(total_carbon, 0),
                    "game_mode": st.session_state.get("game_mode", "free_play"),
                }).execute()
                st.session_state["score_submitted"] = True
                st.session_state["submitted_nickname"] = nickname.strip()
                st.session_state["_session_code"] = session_code.strip()
                st.rerun()
            except Exception as e:
                st.error(f"Submission failed: {e}")

    st.divider()
    if st.button("↺  Start New Simulation", width='stretch'):
        _restart()

    st.stop()


# ── ACTIVE GAME AREA ───────────────────────────────────────────────────────────
_scroll_slot = st.empty()
if st.session_state.get("needs_scroll"):
    st.session_state["needs_scroll"] = False
    with _scroll_slot:
        _scroll_to_top()

rnd = st.session_state["current_round"]
history = st.session_state["history"]

# Header
hdr_col, badge_col = st.columns([5, 1])
hdr_col.markdown("# ♻ The Loopback Initiative")
badge_col.markdown(
    f'<div style="background:#161b22; border:1px solid #58a6ff; border-radius:8px; '
    f'padding:0.6rem; text-align:center; margin-top:0.8rem;">'
    f'<span style="color:#58a6ff; font-size:1.2rem; font-weight:700;">Round {rnd}</span><br>'
    f'<span style="color:#8b949e; font-size:0.7rem;">of {TOTAL_ROUNDS}</span></div>',
    unsafe_allow_html=True,
)

# Shock banner
if st.session_state.get("shock_triggered") or rnd > SHOCK_ROUND:
    st.error(
        "MINING STRIKE EVENT — Primary lead time: **2 rounds** | "
        f"Carbon price: **${CARBON_PRICE_SHOCK:.0f}/kg CO₂e** (was ${CARBON_PRICE_NORMAL:.0f})"
    )

# ── Round narrative ────────────────────────────────────────────────────────────
narrative = ROUND_NARRATIVES.get(rnd)
if narrative:
    title = narrative["title"]
    body = narrative["body"]

    # Dynamic substitution for Round 3
    if rnd == 3 and history:
        last_yield = history[-1]["circular_yield"]
        received = history[-1]["circular_received"]
        yield_comment = "above average" if last_yield > 0.70 else ("average" if last_yield > 0.65 else "below average")
        try:
            body = body.format(yield_pct=last_yield, yield_comment=yield_comment, received=received)
        except (KeyError, ValueError):
            pass

    with st.expander(f"Situation Report — {title}", expanded=True):
        st.markdown(body)
        st.markdown(
            f'<div style="background:#0d1117; border-left:3px solid #d29922; '
            f'padding:0.6rem 1rem; border-radius:0 6px 6px 0; font-size:0.85rem; color:#d29922;">'
            f'{narrative["tip"]}</div>',
            unsafe_allow_html=True,
        )

st.divider()

# KPI cards
if history:
    last = history[-1]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Inventory", f"{st.session_state['inventory']:.0f} units",
              delta=f"{last['ending_inventory'] - last['starting_inventory']:.0f}")
    k2.metric("Round Profit", f"${last['round_profit']:,.0f}")
    k3.metric("Cumulative SAP", f"${st.session_state['cumulative_sap']:,.0f}")
    k4.metric("Round Carbon", f"{last['total_carbon']:.1f} kg CO₂e")
else:
    st.info(
        f"Round {rnd} hasn't been played yet. "
        "Configure your policy in the sidebar and click **Advance to Round 2** to begin."
    )

if history:
    df_hist = pd.DataFrame(history)
    st.divider()

    chart_l, chart_r = st.columns(2)

    with chart_l:
        fig_inv = go.Figure()
        fig_inv.add_trace(go.Scatter(
            x=df_hist["round"], y=df_hist["ending_inventory"],
            mode="lines+markers", name="Ending Inventory",
            line=dict(color="#58a6ff", width=2), marker=dict(size=6),
        ))
        fig_inv.add_hline(
            y=st.session_state["s_reorder_point"], line_dash="dash", line_color="#d29922",
            annotation_text=f"s = {st.session_state['s_reorder_point']}",
            annotation_font_color="#d29922",
        )
        fig_inv.update_layout(**PLOT_LAYOUT, title="Inventory Over Rounds",
                              xaxis_title="Round", yaxis_title="Units")
        st.plotly_chart(fig_inv, width='stretch')

    with chart_r:
        fig_carbon = go.Figure()
        fig_carbon.add_trace(go.Bar(
            x=df_hist["round"], y=df_hist["carbon_primary"],
            name="Primary Carbon", marker_color="#f85149",
        ))
        fig_carbon.add_trace(go.Bar(
            x=df_hist["round"], y=df_hist["carbon_circular"],
            name="Circular Carbon", marker_color="#3fb950",
        ))
        fig_carbon.update_layout(**PLOT_LAYOUT, barmode="stack",
                                  title="Carbon per Round (kg CO₂e)",
                                  xaxis_title="Round", yaxis_title="kg CO₂e")
        st.plotly_chart(fig_carbon, width='stretch')

    st.markdown("### Cumulative Cost Breakdown")
    df_hist["cum_cost_primary"] = df_hist["cost_primary"].cumsum()
    df_hist["cum_cost_circular"] = df_hist["cost_circular"].cumsum()
    df_hist["cum_holding"] = df_hist["cost_holding"].cumsum()
    df_hist["cum_stockout"] = df_hist["cost_stockout"].cumsum()
    df_hist["cum_carbon"] = df_hist["cost_carbon"].cumsum()

    fig_costs = go.Figure()
    for label, col, color in [
        ("Procurement — Primary", "cum_cost_primary", "#58a6ff"),
        ("Procurement — Circular", "cum_cost_circular", "#3fb950"),
        ("Holding", "cum_holding", "#d29922"),
        ("Stockout", "cum_stockout", "#f85149"),
        ("Carbon Tax", "cum_carbon", "#a371f7"),
    ]:
        fig_costs.add_trace(go.Scatter(
            x=df_hist["round"], y=df_hist[col], stackgroup="costs",
            name=label, mode="lines",
            line=dict(width=0.5, color=color), fillcolor=color, opacity=0.7,
        ))
    fig_costs.update_layout(**PLOT_LAYOUT, title="Cumulative Cost Breakdown ($)",
                             xaxis_title="Round", yaxis_title="Cumulative Cost ($)")
    st.plotly_chart(fig_costs, width='stretch')

    st.markdown("### Round History")
    display_cols = [
        "round", "demand", "units_sold", "stockout_units",
        "starting_inventory", "ending_inventory",
        "order_primary", "order_circular", "circular_yield",
        "primary_received", "circular_received",
        "revenue", "cost_primary", "cost_circular",
        "cost_holding", "cost_stockout", "cost_carbon",
        "round_profit", "cumulative_sap", "total_carbon",
    ]
    display_df = df_hist[display_cols].copy()
    for col in display_df.select_dtypes(include="float").columns:
        display_df[col] = display_df[col].round(1)
    st.dataframe(display_df, hide_index=True, width='stretch')
