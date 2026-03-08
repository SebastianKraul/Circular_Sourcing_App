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
    SCENARIOS,
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
def _cached_monte_carlo(s, S, mix_pct, n_runs, scenario):
    return run_monte_carlo(s, S, mix_pct, n_runs, scenario=scenario)


@st.cache_data(show_spinner=False)
def _cached_find_optimal(mix_pct, scenario):
    return find_optimal_policy(mix_pct, scenario=scenario)


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


# ── Scenario narratives ────────────────────────────────────────────────────────
# SCENARIO_NARRATIVES[scenario_key][round_num] = {title, body, tip}
# Round 3 body may contain {yield_pct}, {yield_comment}, {received} placeholders.

SCENARIO_NARRATIVES = {

    # ── Base Game ──────────────────────────────────────────────────────────────
    "base_game": {
        1: {
            "title": "Q1 — First Orders",
            "body": (
                "Your procurement dashboard is live. Inner Mongolia Mining Co. reports full "
                "operational capacity. EcoReclaim Urban Mining has confirmed your account. "
                "The board expects a sustainability strategy within the year. "
                "Carbon is priced at **$2/kg CO₂e**."
            ),
            "tip": (
                "Demand averages ~100 units/round (σ=20). You start with 150 units. "
                "Set your reorder point (s) and order-up-to (S) to avoid running dry."
            ),
        },
        2: {
            "title": "Q2 — ESG Reporting Season",
            "body": (
                "Institutional investors flagged NovaPulse's carbon footprint in their annual "
                "review. The CFO requests a cost-carbon breakdown for the board pack. Primary "
                "sourcing looks cheap on the invoice — but factor in the carbon tax."
            ),
            "tip": (
                "True unit cost at $2/kg CO₂e: **Primary = $5 + 8×$2 = $21**. "
                "**Circular = $12 + 0.5×$2 = $13**. Circular is already cheaper on total-cost basis."
            ),
        },
        3: {
            "title": "Q3 — Yield Variation",
            "body": (
                "EcoReclaim's batch yield came in at **{yield_pct:.0%}** this quarter — "
                "{yield_comment}. Urban mining yield follows N(70%, σ=10%). If you ordered "
                "100 units from circular, you received ~{received:.0f}. "
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
                "Intelligence reports signal growing labour unrest at the Inner Mongolia primary "
                "mine. A trade analyst briefing warns: *'Any disruption would double lead times "
                "and stress supply chains globally.'* Check your pipeline. Are your buffers adequate?"
            ),
            "tip": (
                "If a disruption hits **next quarter**, primary orders placed NOW arrive in "
                "Round 5. Under-prepared firms will face stockouts at $20/unit penalty."
            ),
        },
        5: {
            "title": "Q5 — SUPPLY SHOCK",
            "body": (
                "A wildcat strike has shut the primary mine indefinitely. Primary lead time "
                "doubles to **2 rounds**. The Carbon Pricing Act takes effect: **$8/kg CO₂e**. "
                "Competitors over-reliant on primary sourcing are scrambling. "
                "EcoReclaim is at full capacity."
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
                "The strike continues with no resolution in sight. Circular capacity is "
                "tightening globally as firms pivot away from primary. EcoReclaim is "
                "prioritising established partners. Primary orders placed now arrive in **Round 8**."
            ),
            "tip": (
                "With 2-round primary lead time, you must plan two rounds ahead. "
                "Circular still arrives in 1 round. Consider raising your (s,S) targets."
            ),
        },
        7: {
            "title": "Q7 — Regulatory Tailwinds",
            "body": (
                "The EU Carbon Border Adjustment Mechanism passes into law, targeting "
                "carbon-intensive imports. Analysts expect $8+/kg CO₂e to persist for years. "
                "Circular sourcing is no longer just ethical — it is the economically rational choice."
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
                "capacity is months away. EcoReclaim reports strong yield conditions. "
                "**Two rounds remain** — your final pipeline decisions are critical."
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
                "Circular orders placed now arrive Round 10. Primary (lead time 2) arrives "
                "Round 11 — after the simulation ends. Circular is your only in-time option."
            ),
        },
        10: {
            "title": "Q10 — Final Quarter",
            "body": (
                "The board meeting is confirmed. Shareholders, ESG auditors, and the press are "
                "watching. Any inventory ordered this round arrives **after the simulation ends** "
                "and will not count toward your score."
            ),
            "tip": (
                "Avoid over-ordering — holding costs apply to end-of-round inventory. "
                "Your score is now largely determined. Make your final decisions count."
            ),
        },
    },

    # ── Demand Surge ───────────────────────────────────────────────────────────
    "demand_surge": {
        1: {
            "title": "Q1 — Steady State",
            "body": (
                "Operations are running smoothly. Both suppliers are fully available. "
                "Carbon stays at **$2/kg CO₂e** throughout this scenario — no regulatory shock. "
                "Demand is averaging 100 units/round and the outlook appears stable."
            ),
            "tip": (
                "With no supply disruption, the focus is on demand management. "
                "Calibrate your (s,S) policy for normal demand before conditions change."
            ),
        },
        2: {
            "title": "Q2 — Early Signals",
            "body": (
                "The NovaPulse EX-7 is generating positive buzz in early reviews. "
                "Retail sell-through is slightly above forecast. Channel inventory across "
                "distributors is thinning. Nothing alarming — but worth watching."
            ),
            "tip": (
                "Demand has nudged up slightly. If the trend continues, your current "
                "(s,S) policy may not provide adequate buffer. "
                "Consider whether S is set high enough."
            ),
        },
        3: {
            "title": "Q3 — Growing Momentum",
            "body": (
                "Two tech influencers with combined reach of 40 million subscribers "
                "featured the EX-7. Retailer reorder rates are up 20%. "
                "The sales team is upgrading forecasts. Demand is averaging ~105 units."
            ),
            "tip": (
                "Demand is creeping up. Yield uncertainty on circular orders remains. "
                "If you're planning to build inventory buffer, act now — "
                "orders take at least 1 round to arrive."
            ),
        },
        4: {
            "title": "Q4 — Analyst Alert",
            "body": (
                "A viral unboxing video crossed 10 million views overnight. "
                "Three major retailers have placed emergency pre-orders. "
                "An industry analyst report warns: *'EX-7 demand could double by Q5 "
                "if social momentum holds.'* Your pipeline — is it ready?"
            ),
            "tip": (
                "Average demand next round may reach **~185 units** — nearly double normal. "
                "Orders placed this round arrive in Round 5 (lead time 1). "
                "This is your last chance to pre-position inventory."
            ),
        },
        5: {
            "title": "Q5 — DEMAND SURGE",
            "body": (
                "The EX-7 has gone viral. Demand has roughly doubled to **~185 units/round** "
                "and volatility has spiked (σ=40). Both suppliers remain fully available — "
                "this is a pure demand-side event. "
                "Stockout penalties are now your primary threat."
            ),
            "tip": (
                "At demand ~185 and inventory below S, your (s,S) order will be "
                "(S − inventory) — split by your sourcing mix. "
                "If S is too low, you will chronically under-order and face repeated stockouts."
            ),
        },
        6: {
            "title": "Q6 — Sustaining the Peak",
            "body": (
                "Demand remains elevated at ~185 units. The supply chain team is working "
                "overtime. EcoReclaim has confirmed they can accommodate increased circular "
                "order volumes. Your (s,S) policy was designed for 100-unit demand — "
                "does it still hold?"
            ),
            "tip": (
                "Consider raising S to match the new demand regime. "
                "Remember: (s,S) changes take effect from the *next* round. "
                "Each round of under-ordering compounds your stockout exposure."
            ),
        },
        7: {
            "title": "Q7 — Demand Plateau",
            "body": (
                "Social buzz has stabilised. Demand is still ~180 units — well above "
                "the original baseline. The CFO is asking for a revised inventory strategy "
                "that accounts for the new demand level. Carbon costs remain low at $2/kg CO₂e."
            ),
            "tip": (
                "You're operating in a higher-demand regime now. "
                "The optimal (s,S) for demand ~180 is materially different from "
                "what was optimal for demand ~100. Has your policy kept pace?"
            ),
        },
        8: {
            "title": "Q8 — Early Signs of Cooling",
            "body": (
                "A competing product has launched, drawing some attention away from the EX-7. "
                "Demand is ~175 units — still elevated but trending down slightly. "
                "Retailer pre-orders are moderating. **Two rounds remain.**"
            ),
            "tip": (
                "If demand is declining, over-ordering now creates excess inventory "
                "you'll carry into Q10 with holding costs. "
                "Balance service level against end-of-game inventory position."
            ),
        },
        9: {
            "title": "Q9 — Gradual Normalisation",
            "body": (
                "The competitor has gained market share. Demand is pulling back toward "
                "~170 units. Your sustainability grade is crystallising — review your "
                "circular mix and stockout record before the final quarter."
            ),
            "tip": (
                "Circular orders placed now arrive Round 10. "
                "Avoid over-ordering — end-of-game inventory earns no revenue "
                "but still incurs holding costs."
            ),
        },
        10: {
            "title": "Q10 — Final Quarter",
            "body": (
                "The board meeting is set. The EX-7 story — from baseline to viral surge "
                "to gradual normalisation — is complete. Any inventory ordered this round "
                "arrives **after the simulation ends** and will not count toward your score."
            ),
            "tip": (
                "Focus on demand ~165 from your current pipeline. "
                "Holding costs apply to end-of-round inventory. Make your final decisions count."
            ),
        },
    },

    # ── Carbon Ratchet ─────────────────────────────────────────────────────────
    "carbon_ratchet": {
        1: {
            "title": "Q1 — Regulatory Calendar Published",
            "body": (
                "The government has published a binding carbon pricing schedule. "
                "The trajectory is fixed and public: **$2 → $4 → $6 → $8 → $10/kg CO₂e**, "
                "stepping up every two rounds. Current price: **$2/kg CO₂e**. "
                "No supply disruptions are expected in this scenario."
            ),
            "tip": (
                "At $2/kg: Primary total cost = $21/unit, Circular = $13/unit. "
                "The gap widens with every price step. "
                "Proactive sourcing mix shifts now avoid painful carbon bills later."
            ),
        },
        2: {
            "title": "Q2 — Final Quarter at $2",
            "body": (
                "Operations are stable. The carbon price steps up to $4/kg CO₂e **next round**. "
                "This is your last quarter at the lowest carbon rate. "
                "Primary lead time remains 1 round throughout this scenario."
            ),
            "tip": (
                "Next round: Primary total cost = $5 + 8×$4 = **$37/unit**. "
                "Circular = $12 + 0.5×$4 = **$14/unit**. "
                "The cost gap has already more than doubled. Is your sourcing mix ready?"
            ),
        },
        3: {
            "title": "Q3 — Carbon Steps to $4",
            "body": (
                "As scheduled, the carbon price has increased to **$4/kg CO₂e**. "
                "Your carbon cost this round is double what it was in Q1–Q2. "
                "Firms that shifted to circular early are already ahead on total cost."
            ),
            "tip": (
                "At $4/kg: Primary = $37/unit, Circular = $14/unit. "
                "Carbon gap between suppliers: 7.5 kg × $4 = **$30/unit**. "
                "Next step to $6/kg is in two rounds."
            ),
        },
        4: {
            "title": "Q4 — Holding at $4",
            "body": (
                "Carbon is stable at $4/kg this round before the next scheduled step. "
                "ESG analysts are flagging NovaPulse's carbon trajectory in their reports. "
                "Investors are asking questions about the sourcing strategy."
            ),
            "tip": (
                "Next round: carbon steps to $6/kg. "
                "Primary cost will be $5 + 8×$6 = **$53/unit**. "
                "Circular = $12 + 0.5×$6 = **$15/unit**. "
                "Waiting to switch is becoming increasingly expensive."
            ),
        },
        5: {
            "title": "Q5 — Carbon Steps to $6",
            "body": (
                "The third price step has taken effect: **$6/kg CO₂e**. "
                "At this level, the carbon tax on a single primary unit ($48) now exceeds "
                "the circular premium ($7) by a factor of nearly 7. "
                "The economics of primary sourcing are deteriorating sharply."
            ),
            "tip": (
                "Primary = $53/unit. Circular = $15/unit. "
                "The switching point P* = $0.93/kg was crossed back in Q1 — "
                "but at $6, primary is now **3.5× more expensive than circular per unit**."
            ),
        },
        6: {
            "title": "Q6 — Holding at $6",
            "body": (
                "Carbon holds at $6/kg. The regulatory trajectory has proven credible — "
                "no policy reversals, no delays. Firms that deferred circular investment "
                "are now facing a painful correction. "
                "EcoReclaim reports strong yield conditions this quarter."
            ),
            "tip": (
                "Next round: carbon steps to $8/kg — matching the Base Game shock price. "
                "Primary will cost $5 + 8×$8 = **$69/unit**. "
                "Circular = $12 + 0.5×$8 = **$16/unit**. "
                "Are your (s,S) buffers sized for the new cost reality?"
            ),
        },
        7: {
            "title": "Q7 — Carbon Steps to $8",
            "body": (
                "The carbon price has reached **$8/kg CO₂e** — the same level as the "
                "Base Game shock, but reached gradually over 6 rounds. "
                "Firms with early circular commitments have built a compounding cost advantage."
            ),
            "tip": (
                "Primary = $69/unit. Circular = $16/unit. "
                "One more price step remains. "
                "Circular orders placed now arrive next round (lead time 1)."
            ),
        },
        8: {
            "title": "Q8 — Holding at $8",
            "body": (
                "Carbon holds at $8/kg. The final step to $10/kg is **next round**. "
                "**Two rounds remain.** Your end-of-game carbon footprint is being "
                "watched closely by the ESG committee. "
            ),
            "tip": (
                "Next round: Primary = $5 + 8×$10 = **$85/unit**. "
                "Circular = $12 + 0.5×$10 = **$17/unit**. "
                "At this point, the sourcing mix is the dominant driver of your final score."
            ),
        },
        9: {
            "title": "Q9 — Carbon Steps to $10",
            "body": (
                "The final step: **$10/kg CO₂e**. Primary sourcing now costs $85/unit — "
                "17× more expensive than the original $5 sticker price once carbon is included. "
                "This is the highest carbon rate in any scenario. One round remains."
            ),
            "tip": (
                "Circular orders placed now arrive Round 10 (lead time 1). "
                "Primary orders also arrive Round 10 (lead time 1 in this scenario). "
                "Avoid over-ordering — end-of-game inventory earns no additional revenue."
            ),
        },
        10: {
            "title": "Q10 — Final Quarter at Peak Carbon",
            "body": (
                "The simulation concludes under maximum carbon pricing. "
                "The board will assess whether NovaPulse's sourcing strategy kept pace "
                "with the regulatory trajectory — or chased it reactively. "
                "Any inventory ordered this round arrives **after the simulation ends**."
            ),
            "tip": (
                "Holding costs apply to end-of-round inventory. "
                "Your circular mix percentage over the full 10 rounds is a key score component. "
                "Make your final decisions count."
            ),
        },
    },

    # ── Supplier Failure ───────────────────────────────────────────────────────
    "supplier_failure": {
        1: {
            "title": "Q1 — Strong Partnerships",
            "body": (
                "Both suppliers are performing well. EcoReclaim Urban Mining has reported "
                "record throughput at its EU facilities. Inner Mongolia Mining Co. is at "
                "full capacity. Carbon is **$2/kg CO₂e**."
            ),
            "tip": (
                "Both suppliers are available. Build your sourcing mix and (s,S) policy "
                "with the assumption that this could change — "
                "single-supplier dependency is a hidden risk."
            ),
        },
        2: {
            "title": "Q2 — Rumours in the Market",
            "body": (
                "Trade press is reporting that EcoReclaim's parent company missed a debt "
                "covenant last quarter. Management has denied any liquidity concerns. "
                "Your account manager at EcoReclaim assures you everything is fine."
            ),
            "tip": (
                "Early warning signs in supply chain risk are often dismissed. "
                "Consider what your sourcing strategy looks like if circular "
                "becomes unavailable at short notice."
            ),
        },
        3: {
            "title": "Q3 — {yield_comment} Yield",
            "body": (
                "EcoReclaim's batch yield came in at **{yield_pct:.0%}** this quarter — "
                "{yield_comment}. You received ~{received:.0f} units from your circular order. "
                "The parent company has hired a restructuring advisor, "
                "according to a leaked filing."
            ),
            "tip": (
                "Yield risk is always present with circular sourcing — "
                "order ~43% more than you need to receive reliably. "
                "The financial situation at EcoReclaim bears watching."
            ),
        },
        4: {
            "title": "Q4 — Distress Signals",
            "body": (
                "EcoReclaim's account manager has not responded in two weeks. "
                "A creditor filing has appeared in the public registry. "
                "Industry contacts confirm the company is in emergency talks with lenders. "
                "Inner Mongolia Mining Co. remains fully operational."
            ),
            "tip": (
                "If EcoReclaim fails **next round**, circular sourcing will be permanently "
                "unavailable. Circular orders placed this round arrive in Round 5 — "
                "potentially your last. Primary orders also arrive in Round 5 (lead time 1)."
            ),
        },
        5: {
            "title": "Q5 — SUPPLIER FAILURE",
            "body": (
                "EcoReclaim Urban Mining has filed for insolvency. All facilities are "
                "immediately suspended. **Circular sourcing is permanently unavailable** "
                "from this round forward. Simultaneously, the Carbon Pricing Act takes "
                "effect: **$8/kg CO₂e**. You are now entirely dependent on primary sourcing "
                "at elevated carbon cost."
            ),
            "tip": (
                "Primary total cost: $5 + 8×$8 = **$69/unit**. "
                "There is no circular alternative. "
                "Your only levers are (s,S) policy and managing holding vs. stockout costs."
            ),
        },
        6: {
            "title": "Q6 — Single-Source Reality",
            "body": (
                "Inner Mongolia Mining Co. is your only supplier. They are meeting demand "
                "but your carbon footprint is accumulating rapidly at $8/kg CO₂e. "
                "Competitors who retained circular capacity are posting lower carbon costs."
            ),
            "tip": (
                "Primary lead time is still 1 round in this scenario. "
                "Focus on (s,S) optimisation — it's the only policy lever you have left. "
                "Stockouts at $20/unit penalty are your main financial risk."
            ),
        },
        7: {
            "title": "Q7 — Carbon Accumulation",
            "body": (
                "Three rounds on primary sourcing at $8/kg CO₂e. "
                "Your carbon cost per unit is $69 — well above any competitor "
                "that maintained circular access. The ESG scorecard is looking unfavourable."
            ),
            "tip": (
                "Reflect on the counterfactual: if you had maintained 50% circular "
                "in rounds 1–4, how much carbon exposure would you have avoided? "
                "This is the cost of concentration risk."
            ),
        },
        8: {
            "title": "Q8 — Searching for Alternatives",
            "body": (
                "The procurement team is exploring secondary urban mining contacts in Asia "
                "but nothing is contractually available within the simulation window. "
                "**Two rounds remain.** Carbon cost stays at $8/kg CO₂e."
            ),
            "tip": (
                "Primary orders placed this round arrive Round 9 (lead time 1). "
                "Manage your buffer to avoid stockouts in the final two rounds "
                "without excessive end-of-game inventory."
            ),
        },
        9: {
            "title": "Q9 — Endgame on Primary",
            "body": (
                "One round remains. You have operated on primary-only sourcing since "
                "Round 5. The board's ESG committee has flagged NovaPulse's carbon "
                "trajectory as a material risk in the annual report."
            ),
            "tip": (
                "Primary orders placed now arrive Round 10. "
                "Avoid over-ordering — end-of-game inventory has no value "
                "but still incurs holding costs."
            ),
        },
        10: {
            "title": "Q10 — Final Quarter",
            "body": (
                "The board meeting is confirmed. The story of concentration risk — "
                "from promising partnership to permanent loss — will be central to the debrief. "
                "Any inventory ordered this round arrives **after the simulation ends**."
            ),
            "tip": (
                "Your circular mix percentage over 10 rounds is a key score component — "
                "it reflects the full game, including the rounds before failure. "
                "Make your final decisions count."
            ),
        },
    },

    # ── Known Shock ────────────────────────────────────────────────────────────
    "known_shock": {
        1: {
            "title": "Q1 — Strike Confirmed for Q5",
            "body": (
                "Intelligence is unambiguous: labour action at the Inner Mongolia primary mine "
                "is confirmed to begin in **Round 5**. Lead time will double to 2 rounds. "
                "The Carbon Pricing Act will simultaneously raise the rate to **$8/kg CO₂e**. "
                "You have four rounds to prepare. Carbon is currently **$2/kg CO₂e**."
            ),
            "tip": (
                "This is a planning problem with perfect information. "
                "Primary orders placed in Rounds 3–4 arrive in Rounds 4–5 (lead time 1 now). "
                "After Round 5, primary orders take 2 rounds. "
                "Design your inventory pipeline for the transition."
            ),
        },
        2: {
            "title": "Q2 — Preparation Window: 3 Rounds",
            "body": (
                "The strike timeline holds. Union negotiators report no progress. "
                "EcoReclaim has confirmed they can scale up order volumes if needed. "
                "Three rounds remain before conditions change permanently."
            ),
            "tip": (
                "Consider what inventory level you want entering Round 5. "
                "At demand ~100/round and lead time doubling to 2 rounds, "
                "your safety stock calculation changes materially."
            ),
        },
        3: {
            "title": "Q3 — {yield_comment} Circular Yield",
            "body": (
                "EcoReclaim's yield came in at **{yield_pct:.0%}** — {yield_comment}. "
                "You received ~{received:.0f} units from circular. "
                "The strike timeline is unchanged: Round 5. Two preparation rounds remain."
            ),
            "tip": (
                "Circular yield variability matters even when you're planning ahead. "
                "If you intend to build buffer via circular orders, "
                "order ~43% more than you need to receive reliably."
            ),
        },
        4: {
            "title": "Q4 — Final Preparation Round",
            "body": (
                "One round before the strike. Primary orders placed **this round** "
                "arrive in Round 5 with normal lead time (1 round). "
                "From Round 5 onward, primary lead time doubles to 2 rounds. "
                "This is your last chance to place fast primary orders."
            ),
            "tip": (
                "Primary orders placed in Round 5 arrive Round 7 (lead time 2). "
                "Circular orders placed in Round 5 arrive Round 6 (lead time 1). "
                "What does your inventory pipeline look like entering Round 5?"
            ),
        },
        5: {
            "title": "Q5 — SUPPLY SHOCK (as forecast)",
            "body": (
                "The wildcat strike has begun as predicted. Primary lead time has doubled "
                "to **2 rounds**. The Carbon Pricing Act is in effect: **$8/kg CO₂e**. "
                "The question now: did your preparation match the theory?"
            ),
            "tip": (
                "Recalculate: **Primary = $5 + 8×$8 = $69/unit**. "
                "**Circular = $12 + 0.5×$8 = $16/unit**. "
                "Firms that pre-positioned inventory and shifted mix will feel this differently "
                "from those in the Base Game who were surprised."
            ),
        },
        6: {
            "title": "Q6 — Executing the Plan",
            "body": (
                "The disruption is unfolding exactly as forecast. Primary lead time is 2 rounds. "
                "Carbon is $8/kg CO₂e. Your pipeline position now reflects the quality "
                "of preparation made in Rounds 1–4."
            ),
            "tip": (
                "Primary orders placed in Round 6 arrive Round 8. "
                "Circular orders arrive Round 7. "
                "Adjust (s,S) if your buffers are not sized for the new regime."
            ),
        },
        7: {
            "title": "Q7 — Regulatory Tailwinds",
            "body": (
                "The EU Carbon Border Adjustment Mechanism passes into law. "
                "Analysts confirm $8+/kg CO₂e will persist. "
                "Firms without pre-existing circular relationships are now scrambling "
                "to build them under capacity constraints."
            ),
            "tip": (
                "Your circular mix and carbon footprint trajectory are visible in the charts. "
                "How does your post-shock performance compare to your pre-shock planning?"
            ),
        },
        8: {
            "title": "Q8 — Partial Mine Recovery",
            "body": (
                "Mediators report progress. A partial workforce returned to the mine, "
                "but full capacity is months away. Primary lead time remains 2 rounds. "
                "**Two rounds remain** — your final pipeline decisions are critical."
            ),
            "tip": (
                "Primary orders placed in Round 8 arrive Round 10. "
                "Circular orders arrive Round 9. "
                "End-of-game inventory earns no revenue — size your final orders carefully."
            ),
        },
        9: {
            "title": "Q9 — Endgame",
            "body": (
                "Carbon pricing holds at $8/kg CO₂e. Your sustainability grade is "
                "crystallising. The board will compare your result against the Base Game "
                "cohort — who faced the same shock without advance notice."
            ),
            "tip": (
                "Circular orders placed now arrive Round 10 (lead time 1). "
                "Primary orders (lead time 2) arrive Round 11 — after the simulation ends. "
                "Circular is your only in-time option."
            ),
        },
        10: {
            "title": "Q10 — Final Quarter",
            "body": (
                "The board meeting is confirmed. You had perfect foresight of the disruption "
                "— the debrief will examine whether foreknowledge translated into better "
                "preparation and outcomes vs. the Base Game scenario."
            ),
            "tip": (
                "Any inventory ordered this round arrives after the simulation ends. "
                "Holding costs apply to end-of-round inventory. Make your final decisions count."
            ),
        },
    },

    # ── Seasonal ───────────────────────────────────────────────────────────────
    "seasonal": {
        1: {
            "title": "Q1 — Off Season",
            "body": (
                "It is the start of the fiscal year. Consumer electronics demand is at its "
                "seasonal low — averaging **~65 units/round** this quarter. "
                "Carbon stays at **$2/kg CO₂e** throughout this scenario. "
                "The peak season runs Q5–Q6 when demand reaches ~160 units."
            ),
            "tip": (
                "Off-season demand is well below your starting inventory of 150 units. "
                "Use these quiet rounds to build buffer stock ahead of the peak — "
                "but don't over-invest in holding costs if you build too early."
            ),
        },
        2: {
            "title": "Q2 — Pre-Season Build",
            "body": (
                "Demand is rising to **~80 units/round** as distributors begin restocking "
                "ahead of the peak season. This is the ideal window to increase inventory "
                "levels proactively. Lead times remain 1 round for both suppliers."
            ),
            "tip": (
                "Orders placed this round arrive Round 3. Orders placed in Round 3 "
                "arrive Round 4. You have two more rounds before demand reaches ~130 units. "
                "Size your order-up-to (S) for the peak, not the current demand."
            ),
        },
        3: {
            "title": "Q3 — Season Building",
            "body": (
                "Demand has reached **~100 units/round** — the long-run average, "
                "but still well below the upcoming peak. Retailer orders are accelerating. "
                "EcoReclaim's yield came in at **{yield_pct:.0%}** this quarter — "
                "{yield_comment}. You received ~{received:.0f} units from circular."
            ),
            "tip": (
                "Peak demand (Rounds 5–6) averages ~160 units. "
                "Orders placed NOW arrive Round 4. "
                "If your inventory entering Round 5 is below ~200 units, "
                "stockouts are likely."
            ),
        },
        4: {
            "title": "Q4 — Final Build Round",
            "body": (
                "Demand is ramping to **~130 units/round**. Holiday season purchasing "
                "begins next quarter. Orders placed this round arrive in Round 5 — "
                "the start of peak. This is your final opportunity to build buffer "
                "before demand surges."
            ),
            "tip": (
                "Round 5 demand averages ~160 units (σ=30). "
                "Inventory entering Round 5 should be at least 160–200 units to safely "
                "cover demand plus uncertainty. What does your pipeline look like?"
            ),
        },
        5: {
            "title": "Q5 — PEAK SEASON",
            "body": (
                "Holiday season demand has arrived: **~160 units/round** with elevated "
                "volatility (σ=30). Both suppliers are fully available and lead times "
                "remain 1 round. This is the most critical quarter — stockouts now "
                "cost $20/unit in penalty plus lost revenue of $50/unit."
            ),
            "tip": (
                "Peak demand continues into Round 6. "
                "Circular orders placed now arrive Round 6. "
                "Ensure your pipeline covers both peak rounds before ordering less."
            ),
        },
        6: {
            "title": "Q6 — Peak Continues",
            "body": (
                "Demand holds near peak at **~155 units/round**. Retailers are fulfilling "
                "pre-committed orders. EcoReclaim is operating at high utilisation. "
                "The peak will begin fading from Round 7 onward."
            ),
            "tip": (
                "Round 7 demand drops to ~130 units. "
                "Start tapering your (s,S) policy downward — "
                "over-ordering now creates excess inventory you'll carry through "
                "the low season at $1/unit/round holding cost."
            ),
        },
        7: {
            "title": "Q7 — Post-Peak Decline",
            "body": (
                "The peak has passed. Demand is declining to **~130 units/round**. "
                "Retailer channel inventory is replenished. "
                "The procurement team's focus shifts to controlled drawdown — "
                "running inventory down without triggering stockouts in the off-season."
            ),
            "tip": (
                "Consider lowering your order-up-to (S) to match declining demand. "
                "A high S in low-demand rounds forces unnecessary orders and holding costs. "
                "Remember: (s,S) changes take effect the *following* round."
            ),
        },
        8: {
            "title": "Q8 — Demand Cooling",
            "body": (
                "Demand is returning to the seasonal baseline: **~100 units/round**. "
                "Distributors are managing their own channel inventory down. "
                "**Two rounds remain** — your end-of-game inventory position matters."
            ),
            "tip": (
                "Inventory held at Round 10's end-of-game earns no revenue but incurs "
                "holding costs. Size your orders in Rounds 8–9 to minimise surplus stock. "
                "Orders placed this round arrive Round 9."
            ),
        },
        9: {
            "title": "Q9 — Off Season Returns",
            "body": (
                "Demand has dropped to **~80 units/round** — back to pre-season levels. "
                "Your sustainability grade is crystallising. "
                "The board will review how well the seasonal pattern was anticipated "
                "and whether stockouts occurred during the peak."
            ),
            "tip": (
                "Circular orders placed now arrive Round 10 (lead time 1). "
                "Demand in Round 10 averages ~65 units. "
                "Avoid over-ordering — end-of-game surplus has no value."
            ),
        },
        10: {
            "title": "Q10 — Year End",
            "body": (
                "The fiscal year closes. Demand is at its seasonal low: **~65 units/round**. "
                "Any inventory ordered this round arrives **after the simulation ends** "
                "and will not count toward your score. "
                "The board will assess whether the seasonal cycle was anticipated and managed well."
            ),
            "tip": (
                "End-of-game inventory earns holding costs but no revenue. "
                "Aim to end with minimal surplus. "
                "Your score reflects the full seasonal cycle — peak service level and off-season discipline."
            ),
        },
    },
}

# ── Scenario banners (shown persistently after shock_triggered = True) ─────────
# None = no persistent banner for this scenario (narrative handles it instead).
SCENARIO_BANNERS = {
    "base_game":       (
        "MINING STRIKE — Primary lead time: **2 rounds** | "
        f"Carbon price: **${CARBON_PRICE_SHOCK:.0f}/kg CO₂e** (was ${CARBON_PRICE_NORMAL:.0f})"
    ),
    "demand_surge":    "DEMAND SURGE — Market demand has roughly doubled | Volatility elevated",
    "carbon_ratchet":  None,
    "supplier_failure": (
        "SUPPLIER FAILURE — EcoReclaim has ceased operations | "
        f"Circular sourcing unavailable | Carbon price: **${CARBON_PRICE_SHOCK:.0f}/kg CO₂e**"
    ),
    "known_shock":     (
        "MINING STRIKE (as forecast) — Primary lead time: **2 rounds** | "
        f"Carbon price: **${CARBON_PRICE_SHOCK:.0f}/kg CO₂e**"
    ),
    "seasonal":        None,
}

# ── Incompatible scenario / game-mode combinations ─────────────────────────────
INCOMPATIBLE_COMBOS = {
    ("supplier_failure", "circular_lock"),
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
        st.session_state["policy_changes"] = 0


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
          <li style="margin-bottom:0.8rem;"><strong>Carbon pricing is active — and may change</strong><br>
              <span style="color:#8b949e; font-size:0.9rem;">Currently $2/kg CO₂e</span></li>
          <li><strong>Policy changes are not immediate</strong><br>
              <span style="color:#8b949e; font-size:0.9rem;">Adjustments to s and S take effect the <em>following</em> quarter — design your policy upfront, not reactively</span></li>
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
        'Choose a scenario, set your inventory policy, and select a sourcing mode '
        'before the simulation begins.</p>',
        unsafe_allow_html=True,
    )
    _step_indicator(3)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scenario selector (full width) ────────────────────────────────────────
    scenario_keys = list(SCENARIOS.keys())
    scenario_labels = [SCENARIOS[k]["label"] for k in scenario_keys]
    default_idx = scenario_keys.index(st.session_state.get("_setup_scenario", "base_game"))

    selected_scenario = st.selectbox(
        "Scenario",
        options=scenario_keys,
        format_func=lambda k: SCENARIOS[k]["label"],
        index=default_idx,
        key="setup_scenario_select",
    )
    st.session_state["_setup_scenario"] = selected_scenario
    _card(
        f'<p style="color:#8b949e; margin:0; font-size:0.9rem;">'
        f'{SCENARIOS[selected_scenario]["description"]}</p>',
        border_color="#30363d",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)

    with left:
        st.markdown("### Game Mode")
        game_mode = st.radio(
            "Select a mode:",
            options=["free_play", "primary_lock", "circular_lock"],
            format_func=lambda x: {
                "free_play": "Free Play — All decisions are yours",
                "primary_lock": "Primary Lock — 100% primary sourcing",
                "circular_lock": "Circular Challenge — 100% circular sourcing",
            }[x],
            index=["free_play", "primary_lock", "circular_lock"].index(
                st.session_state.get("game_mode", "free_play")
            ),
            key="setup_game_mode",
        )
        mode_descriptions = {
            "free_play": "You control everything: reorder points, order-up-to levels, and sourcing mix each round.",
            "primary_lock": "Sourcing is locked to 100% primary. Focus on inventory policy and observe your carbon footprint grow.",
            "circular_lock": "Sourcing is locked to 100% circular. Learn to manage yield uncertainty and higher per-unit costs while keeping carbon low.",
        }
        st.info(mode_descriptions[game_mode])

        # Incompatibility check
        combo_invalid = (selected_scenario, game_mode) in INCOMPATIBLE_COMBOS
        if combo_invalid:
            st.error(
                f"**{SCENARIOS[selected_scenario]['label']}** is not compatible with "
                f"**Circular Challenge** mode — the circular supplier becomes unavailable "
                f"mid-game. Choose Free Play or Primary Lock."
            )

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
        Starting inventory is 150. How low is too low? How high is too high?<br><br>
        <strong style="color:#f85149;">Note:</strong> Your initial settings apply immediately.
        Any changes made <em>during</em> the simulation take effect from the
        <em>following</em> quarter — not the current one. The number of policy
        changes you make is tracked and shown on the leaderboard.
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
            disabled=not setup_valid or combo_invalid,
            width='stretch',
        ):
            # Apply setup choices to game state
            state = init_game_state(seed=42, scenario=selected_scenario)
            state["s_reorder_point"] = setup_s
            state["S_order_up_to"] = setup_S
            state["sourcing_mix_pct"] = setup_mix
            for k, v in state.items():
                st.session_state[k] = v
            st.session_state["game_mode"] = game_mode
            st.session_state["onboarding_complete"] = True
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
    scenario_key = st.session_state.get("scenario", "base_game")
    st.caption(f"Scenario: {SCENARIOS[scenario_key]['label']}")
    st.caption(f"Mode: {mode_label}")
    st.caption(f"Policy changes: {st.session_state.get('policy_changes', 0)}")
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
        _eff_s = st.session_state["s_reorder_point"]
        _eff_S = st.session_state["S_order_up_to"]
        if s_val != _eff_s or S_val != _eff_S:
            st.caption(
                f"Active this round: s={_eff_s}, S={_eff_S} — "
                "changes apply from the next round."
            )

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
    "scenario",
}

if advance and inputs_valid and not st.session_state["game_over"]:
    # Capture pending (s,S) from sidebar widgets before running the round
    _pend_s = st.session_state.get("sidebar_s", st.session_state["s_reorder_point"])
    _pend_S = st.session_state.get("sidebar_S", st.session_state["S_order_up_to"])

    # Count policy change if sidebar values differ from currently effective values
    if _pend_s != st.session_state["s_reorder_point"] or _pend_S != st.session_state["S_order_up_to"]:
        st.session_state["policy_changes"] = st.session_state.get("policy_changes", 0) + 1

    if use_override:
        st.session_state["manual_primary_override"] = man_primary if man_primary > 0 else None
        st.session_state["manual_circular_override"] = man_circular if man_circular > 0 else None
    else:
        st.session_state["manual_primary_override"] = None
        st.session_state["manual_circular_override"] = None

    # Run round with currently effective (s,S) — pending values apply after
    game_state = {k: st.session_state[k] for k in GAME_KEYS if k in st.session_state}
    new_state = run_round(game_state)
    for k, v in new_state.items():
        st.session_state[k] = v

    # Now promote pending (s,S) to effective — takes effect next round
    st.session_state["s_reorder_point"] = _pend_s
    st.session_state["S_order_up_to"] = _pend_S

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

        _fin_scenario_key = st.session_state.get("scenario", "base_game")
        _sc_prices = SCENARIOS[_fin_scenario_key]["carbon_prices"]
        _max_carbon = max(_sc_prices)
        _min_carbon = min(_sc_prices)
        _carbon_changed = _max_carbon > _min_carbon
        _event_round_fin = SCENARIOS[_fin_scenario_key].get("event_round")

        if _carbon_changed:
            _carbon_note = (
                f'The peak carbon price in this scenario was '
                f'<strong>${_max_carbon:.0f}/kg CO₂e</strong>, which is '
                f'<strong style="color:#3fb950;">{_max_carbon/p_star:.1f}×</strong> above P*. '
                f'From Round {_event_round_fin} onward, circular sourcing was strongly '
                f'cost-justified even before ESG considerations.'
            )
        else:
            _carbon_note = (
                f'Carbon pricing was constant at <strong>${_max_carbon:.0f}/kg CO₂e</strong> '
                f'throughout this scenario — already above P* (${p_star:.2f}/kg CO₂e), '
                f'so circular sourcing was cost-justified from Round 1.'
            )

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
            <tr><td style="color:#8b949e; padding:4px 0;">Policy Changes</td>
                <td style="color:#{'d29922' if st.session_state.get('policy_changes', 0) > 0 else '3fb950'}; text-align:right;">{st.session_state.get('policy_changes', 0)}</td></tr>
          </table>
          <hr style="border-color:#30363d; margin:0.8rem 0;">
          <p style="color:#8b949e; font-size:0.82rem; margin:0;">
            <strong style="color:#58a6ff;">P* = ${p_star:.2f}/kg CO₂e</strong>
            — the carbon price at which circular becomes cheaper than primary on a total-cost basis.<br><br>
            {_carbon_note}
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### SAP Trajectory")
    df_hist = pd.DataFrame(history)
    fig_sap = make_subplots(specs=[[{"secondary_y": True}]])
    _sap_scenario_key = st.session_state.get("scenario", "base_game")
    _sap_event_round = SCENARIOS[_sap_scenario_key].get("event_round")
    _bar_colors = [
        "#f85149" if (_sap_event_round and r >= _sap_event_round) else "#58a6ff"
        for r in df_hist["round"]
    ]
    fig_sap.add_trace(
        go.Bar(x=df_hist["round"], y=df_hist["round_profit"], name="Round Profit",
               marker_color=_bar_colors, opacity=0.85),
        secondary_y=False,
    )
    fig_sap.add_trace(
        go.Scatter(x=df_hist["round"], y=df_hist["cumulative_sap"], name="Cumulative SAP",
                   line=dict(color="#3fb950", width=2), mode="lines+markers"),
        secondary_y=True,
    )
    if _sap_event_round is not None:
        fig_sap.add_vline(
            x=_sap_event_round - 0.5, line_dash="dash", line_color="#f85149", opacity=0.6,
            annotation_text="EVENT", annotation_font_color="#f85149",
        )
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
    _dq_scenario = st.session_state.get("scenario", "base_game")
    _dq_event = SCENARIOS[_dq_scenario].get("event_round")
    _dq_max_carbon = max(SCENARIOS[_dq_scenario]["carbon_prices"])
    _dq_carbon_changed = _dq_max_carbon > min(SCENARIOS[_dq_scenario]["carbon_prices"])

    # Q1: Switching Point — tailor to whether carbon changed
    if _dq_carbon_changed:
        _q1_body = (
            f"P* = **${p_star:.2f}/kg CO₂e**. The peak carbon price in this scenario was "
            f"**${_dq_max_carbon:.0f}/kg CO₂e** — {_dq_max_carbon/p_star:.1f}× above P*. "
            f"When did circular sourcing become economically rational, and did your strategy reflect this in time?"
        )
    else:
        _q1_body = (
            f"P* = **${p_star:.2f}/kg CO₂e**. The carbon price was constant at "
            f"**${_dq_max_carbon:.0f}/kg CO₂e** — already above P* from Round 1. "
            f"Did your sourcing mix reflect this cost advantage of circular throughout the game?"
        )

    # Q2: Pipeline / Lead Times — only relevant for scenarios with lead-time change
    _lead_times = SCENARIOS[_dq_scenario]["lead_times_primary"]
    _lead_time_changed = len(set(_lead_times)) > 1
    if _lead_time_changed:
        _q2 = ("2. Pipeline Risk & Lead Times",
               f"When the disruption hit in Round {_dq_event}, primary lead time doubled to 2 rounds. "
               "How did the orders already in your pipeline shape your resilience? "
               "What would you pre-position differently if you played again?")
    else:
        _q2 = ("2. Pipeline & Demand Planning",
               "Primary lead time was constant at 1 round throughout this scenario. "
               "How did the timing of your orders relative to demand changes affect your inventory position? "
               "What would a better pipeline planning approach look like?")

    questions = [
        ("1. The Switching Point", _q1_body),
        _q2,
        ("3. Yield Uncertainty & Over-ordering",
         "EcoReclaim's yield varied ~N(70%, 10%) each round. Did you build a yield buffer into "
         "your circular orders? What systematic approach could reduce the risk of under-receiving?"),
        ("4. (s,S) Policy Design",
         "Reflect on your reorder point (s) and order-up-to (S). Were they well-calibrated for "
         "the demand pattern in this scenario? Did you adjust them proactively or reactively? "
         "What would an optimal policy look like?"),
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
            _mc_scenario = st.session_state.get("scenario", "base_game")
            with st.spinner("Searching for near-optimal policy and running simulations…"):
                opt_s, opt_S, _ = _cached_find_optimal(student_mix, _mc_scenario)
                student_saps, student_so = _cached_monte_carlo(
                    student_s, student_S, student_mix, 1000, _mc_scenario)
                optimal_saps, optimal_so = _cached_monte_carlo(
                    opt_s, opt_S, student_mix, 1000, _mc_scenario)
                default_saps, default_so = _cached_monte_carlo(
                    DEFAULT_S, DEFAULT_S_UPPER, student_mix, 1000, _mc_scenario)

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
              <tr><td style="color:#8b949e; padding:3px 0;">Policy Changes</td>
                  <td style="color:#c9d1d9; text-align:right;">{st.session_state.get('policy_changes', 0)}</td></tr>
            </table>
            """, border_color="#58a6ff")
        submit_disabled = (
            not session_code.strip()
            or not nickname.strip()
            or len(session_code.strip()) > 50
            or len(nickname.strip()) > 40
        )
        if len(session_code.strip()) > 50:
            st.error("Session code must be 50 characters or fewer.")
        if len(nickname.strip()) > 40:
            st.error("Nickname must be 40 characters or fewer.")
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
                    "policy_changes": st.session_state.get("policy_changes", 0),
                    "game_mode": st.session_state.get("game_mode", "free_play"),
                    "scenario": st.session_state.get("scenario", "base_game"),
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
_scenario_key = st.session_state.get("scenario", "base_game")
_banner_text = SCENARIO_BANNERS.get(_scenario_key)
if _banner_text and st.session_state.get("shock_triggered"):
    st.error(_banner_text)

# ── Round narrative ────────────────────────────────────────────────────────────
_scenario_narratives = SCENARIO_NARRATIVES.get(_scenario_key, SCENARIO_NARRATIVES["base_game"])
narrative = _scenario_narratives.get(rnd)
if narrative:
    title = narrative["title"]
    body = narrative["body"]

    # Dynamic substitution where {yield_pct}, {yield_comment}, {received} appear
    if "{yield_pct}" in body and history:
        last = history[-1]
        if last["order_circular"] > 0:
            last_yield = last["circular_yield"]
            received = last["circular_received"]
            yield_comment = (
                "above average" if last_yield > 0.70
                else ("average" if last_yield > 0.65 else "below average")
            )
            try:
                body = body.format(
                    yield_pct=last_yield, yield_comment=yield_comment, received=received
                )
            except (KeyError, ValueError):
                pass
        else:
            # No circular orders placed last round — show a static fallback
            body = (
                "No circular orders were placed last quarter, so EcoReclaim's yield data "
                "is not available for your account. Urban mining yield follows N(70%, σ=10%) — "
                "if you order circular units, expect to receive ~70% of the quantity ordered, "
                "with variation. Consider whether a mixed sourcing strategy would strengthen resilience."
            )

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
    df_costs = df_hist.copy()
    df_costs["cum_cost_primary"] = df_costs["cost_primary"].cumsum()
    df_costs["cum_cost_circular"] = df_costs["cost_circular"].cumsum()
    df_costs["cum_holding"] = df_costs["cost_holding"].cumsum()
    df_costs["cum_stockout"] = df_costs["cost_stockout"].cumsum()
    df_costs["cum_carbon"] = df_costs["cost_carbon"].cumsum()

    fig_costs = go.Figure()
    for label, col, color in [
        ("Procurement — Primary", "cum_cost_primary", "#58a6ff"),
        ("Procurement — Circular", "cum_cost_circular", "#3fb950"),
        ("Holding", "cum_holding", "#d29922"),
        ("Stockout", "cum_stockout", "#f85149"),
        ("Carbon Tax", "cum_carbon", "#a371f7"),
    ]:
        fig_costs.add_trace(go.Scatter(
            x=df_costs["round"], y=df_costs[col], stackgroup="costs",
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
