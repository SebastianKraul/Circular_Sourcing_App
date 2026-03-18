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
        st.session_state["shock_announcement_shown"] = False
        st.session_state["show_round_result"] = False
        st.session_state["round_result_data"] = {}


def _restart():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


_init()



# ── Scroll helpers ─────────────────────────────────────────────────────────────
def _scroll_to_top():
    bust = st.session_state.get("scroll_counter", 0)
    components.html(
        f"""<script>
        (function() {{
            var selectors = [
                '[data-testid="stMain"]',
                '[data-testid="stAppViewContainer"]',
                'section.main', '.main'
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
    clean = textwrap.dedent(content_html).strip()
    st.markdown(
        f'<div style="background:#161b22; border:1px solid {border_color}; '
        f'border-radius:10px; padding:1.5rem; margin-bottom:1rem;">'
        f'{clean}</div>',
        unsafe_allow_html=True,
    )


def _card_row(*cards):
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


# ── Onboarding: Screen 0 — Briefing Room ──────────────────────────────────────
def _briefing_room():
    st.markdown(
        '<div style="text-align:center; margin-bottom:2.5rem;">'
        '<h1 style="color:#58a6ff; letter-spacing:0.12em; margin-bottom:0.3rem;">'
        'THE LOOPBACK INITIATIVE</h1>'
        '<p style="color:#8b949e; font-size:0.95rem; margin:0;">'
        'CONFIDENTIAL &mdash; NovaPulse Electronics | Supply Chain Directorate</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    _card_row(
        ("""
        <h3 style="color:#58a6ff; margin-top:0; letter-spacing:0.06em;">MISSION BRIEF</h3>
        <p style="color:#8b949e; font-size:0.82rem; margin-bottom:1rem;">
        TO: Incoming Supply Chain Director<br>
        FROM: Chief Operations Officer<br>
        RE: Rare Earth Magnet Sourcing &mdash; 10-Quarter Mandate
        </p>
        <p>You have been appointed to lead sourcing of <strong>NdFeB Rare Earth Magnets</strong>
        &mdash; the critical input for NovaPulse's premium audio and EV motor product lines.
        Without them, the production line stops.</p>

        <p>Your mandate covers <strong>10 quarters</strong>. You will manage two suppliers
        with very different cost, carbon, and risk profiles. The board expects a
        <strong>Sustainability-Adjusted Profit (SAP)</strong> report at the end.</p>

        <h4 style="color:#c9d1d9; margin-bottom:0.4rem;">What you control each quarter:</h4>
        <ul style="color:#c9d1d9; margin:0; padding-left:1.2rem;">
          <li><strong style="color:#d29922;">Reorder Trigger (s)</strong> &mdash; inventory level that triggers a new order</li>
          <li><strong style="color:#d29922;">Target Stock Level (S)</strong> &mdash; how much inventory to order up to</li>
          <li><strong style="color:#d29922;">Sourcing Mix</strong> &mdash; share of circular vs. primary in each order</li>
        </ul>
        """, "#58a6ff", 3),
        ("""
        <h3 style="color:#d29922; margin-top:0; letter-spacing:0.06em;">OPERATING CONDITIONS</h3>
        <table style="width:100%; border-collapse:collapse; font-family:monospace; font-size:0.88rem;">
          <tr><td style="color:#8b949e; padding:5px 0;">Demand</td>
              <td style="color:#c9d1d9; text-align:right;">~N(100, &sigma;=20) units/qtr</td></tr>
          <tr><td style="color:#8b949e; padding:5px 0;">Starting inventory</td>
              <td style="color:#c9d1d9; text-align:right;">150 units</td></tr>
          <tr><td style="color:#8b949e; padding:5px 0;">Carbon price (current)</td>
              <td style="color:#c9d1d9; text-align:right;">$2/kg CO&#x2082;e</td></tr>
          <tr><td style="color:#8b949e; padding:5px 0;">Revenue per unit sold</td>
              <td style="color:#3fb950; text-align:right;">$50</td></tr>
          <tr><td style="color:#8b949e; padding:5px 0;">Stockout penalty</td>
              <td style="color:#f85149; text-align:right;">$20/unit short</td></tr>
          <tr><td style="color:#8b949e; padding:5px 0;">Holding cost</td>
              <td style="color:#c9d1d9; text-align:right;">$1/unit/quarter</td></tr>
        </table>
        <hr style="border-color:#30363d; margin:1rem 0;">
        <p style="color:#8b949e; font-size:0.82rem; margin:0;">
        Policy changes to (s, S) take effect the <strong style="color:#c9d1d9;">following quarter</strong>.
        Design your strategy upfront &mdash; reactive adjustments are costly.
        Intelligence suggests a <em>supply or market event</em> may occur mid-simulation.
        </p>
        """, "#d29922", 2),
    )

    _card_row(
        ("""
        <h4 style="color:#58a6ff; margin-top:0;">Primary Mining Co.</h4>
        <p style="color:#8b949e; font-size:0.82rem; margin-top:-0.4rem;">Inner Mongolia, China</p>
        <table style="width:100%; border-collapse:collapse; font-family:monospace; font-size:0.85rem;">
          <tr><td style="color:#8b949e; padding:3px 0;">Unit cost</td>
              <td style="color:#c9d1d9; text-align:right; font-weight:700;">$5.00</td></tr>
          <tr><td style="color:#8b949e; padding:3px 0;">Carbon intensity</td>
              <td style="color:#f85149; text-align:right;">8.0 kg CO&#x2082;e/unit</td></tr>
          <tr><td style="color:#8b949e; padding:3px 0;">Lead time (normal)</td>
              <td style="color:#c9d1d9; text-align:right;">1 quarter</td></tr>
          <tr><td style="color:#8b949e; padding:3px 0;">Yield</td>
              <td style="color:#3fb950; text-align:right;">100% reliable</td></tr>
          <tr><td style="color:#8b949e; padding:3px 0;">Geopolitical exposure</td>
              <td style="color:#d29922; text-align:right;">HIGH</td></tr>
        </table>
        <hr style="border-color:#30363d; margin:0.8rem 0;">
        <p style="color:#8b949e; font-size:0.82rem; margin:0;">
        At $2/kg CO&#x2082;e: true cost = <strong style="color:#58a6ff;">$21/unit</strong>
        </p>
        """, "#58a6ff", 1),
        ("""
        <h4 style="color:#3fb950; margin-top:0;">EcoReclaim Urban Mining</h4>
        <p style="color:#8b949e; font-size:0.82rem; margin-top:-0.4rem;">Distributed EU/NA Facilities</p>
        <table style="width:100%; border-collapse:collapse; font-family:monospace; font-size:0.85rem;">
          <tr><td style="color:#8b949e; padding:3px 0;">Unit cost</td>
              <td style="color:#c9d1d9; text-align:right; font-weight:700;">$12.00</td></tr>
          <tr><td style="color:#8b949e; padding:3px 0;">Carbon intensity</td>
              <td style="color:#3fb950; text-align:right;">0.5 kg CO&#x2082;e/unit</td></tr>
          <tr><td style="color:#8b949e; padding:3px 0;">Lead time</td>
              <td style="color:#c9d1d9; text-align:right;">1 quarter</td></tr>
          <tr><td style="color:#8b949e; padding:3px 0;">Yield</td>
              <td style="color:#d29922; text-align:right;">~N(70%, &sigma;=10%)</td></tr>
          <tr><td style="color:#8b949e; padding:3px 0;">Geopolitical exposure</td>
              <td style="color:#3fb950; text-align:right;">LOW</td></tr>
        </table>
        <hr style="border-color:#30363d; margin:0.8rem 0;">
        <p style="color:#8b949e; font-size:0.82rem; margin:0;">
        At $2/kg CO&#x2082;e: true cost = <strong style="color:#3fb950;">$13/unit</strong>.
        Order ~43% more than needed to buffer yield loss.
        </p>
        """, "#3fb950", 1),
    )

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        if st.button("Proceed to Strategy Configuration \u2192", width='stretch', key="ob_briefing_next"):
            st.session_state["onboarding_step"] = 1
            st.session_state["needs_scroll"] = True
            st.session_state["scroll_counter"] = st.session_state.get("scroll_counter", 0) + 1
            st.rerun()


# ── Onboarding: Screen 1 — Mission Dossier (setup) ────────────────────────────
def _mission_dossier():
    st.markdown(
        '<div style="text-align:center; margin-bottom:2rem;">'
        '<h2 style="color:#e6edf3; letter-spacing:0.06em; margin-bottom:0.3rem;">'
        'STRATEGY CONFIGURATION</h2>'
        '<p style="color:#8b949e; font-size:0.9rem; margin:0;">'
        'Define your operating mandate before the simulation begins.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    scenario_keys = list(SCENARIOS.keys())
    default_idx = scenario_keys.index(st.session_state.get("_setup_scenario", "base_game"))
    selected_scenario = st.selectbox(
        "Scenario",
        options=scenario_keys,
        format_func=lambda k: SCENARIOS[k]["label"],
        index=default_idx,
        key="md_scenario_select",
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
        st.markdown(
            '<h3 style="color:#e6edf3; margin-bottom:0.8rem;">Board Mandate</h3>',
            unsafe_allow_html=True,
        )
        game_mode = st.radio(
            "Select mandate:",
            options=["free_play", "primary_lock", "circular_lock"],
            format_func=lambda x: {
                "free_play":     "Full Discretion \u2014 all sourcing decisions are yours",
                "primary_lock":  "Primary Only \u2014 board has locked sourcing to primary",
                "circular_lock": "Circular Challenge \u2014 board mandate: 100% circular",
            }[x],
            index=["free_play", "primary_lock", "circular_lock"].index(
                st.session_state.get("game_mode", "free_play")
            ),
            key="md_game_mode",
        )
        mandate_notes = {
            "free_play":     "You control reorder trigger, target level, and sourcing mix each quarter.",
            "primary_lock":  "Sourcing locked to 100% primary. Observe your carbon footprint accumulate under full geopolitical exposure.",
            "circular_lock": "Sourcing locked to 100% circular. Manage yield uncertainty and higher unit cost while keeping carbon near zero.",
        }
        st.info(mandate_notes[game_mode])

        combo_invalid = (selected_scenario, game_mode) in INCOMPATIBLE_COMBOS
        if combo_invalid:
            st.error(
                f"**{SCENARIOS[selected_scenario]['label']}** is incompatible with Circular Challenge "
                f"\u2014 the circular supplier becomes unavailable mid-game. "
                f"Choose Full Discretion or Primary Only."
            )

        st.markdown(
            '<h3 style="color:#e6edf3; margin-top:1.5rem; margin-bottom:0.8rem;">Sourcing Mix</h3>',
            unsafe_allow_html=True,
        )
        if game_mode == "primary_lock":
            st.markdown(
                '<div style="background:#161b22; border:1px solid #58a6ff; border-radius:6px; '
                'padding:0.8rem; color:#58a6ff; font-weight:700;">LOCKED: 100% Primary</div>',
                unsafe_allow_html=True,
            )
            setup_mix_primary = 100
        elif game_mode == "circular_lock":
            st.markdown(
                '<div style="background:#161b22; border:1px solid #3fb950; border-radius:6px; '
                'padding:0.8rem; color:#3fb950; font-weight:700;">LOCKED: 100% Circular</div>',
                unsafe_allow_html=True,
            )
            setup_mix_primary = 0
        else:
            circ_slider = st.slider(
                "Circular % (remainder goes to Primary)",
                min_value=0, max_value=100,
                value=100 - int(st.session_state.get("sourcing_mix_pct", DEFAULT_MIX)),
                step=5,
                key="md_mix_slider",
            )
            setup_mix_primary = 100 - circ_slider
            c1, c2 = st.columns(2)
            c1.metric("Primary", f"{setup_mix_primary}%")
            c2.metric("Circular", f"{circ_slider}%")

    with right:
        st.markdown(
            '<h3 style="color:#e6edf3; margin-bottom:0.8rem;">Inventory Policy</h3>',
            unsafe_allow_html=True,
        )
        _card("""
        <p style="color:#8b949e; font-size:0.85rem; margin:0;">
        <strong style="color:#d29922;">Reorder Trigger (s)</strong> &mdash; when inventory
        falls to or below s, you place an order.<br><br>
        <strong style="color:#d29922;">Target Stock Level (S)</strong> &mdash; you order
        (S &minus; inventory) units, split by your sourcing mix.<br><br>
        <strong>Think carefully:</strong> demand averages 100 units/quarter. Starting
        inventory is 150 units. How low is too low? How high is too high?<br><br>
        <strong style="color:#f85149;">Note:</strong> your initial settings apply from
        Q1. Changes made <em>during</em> the simulation take effect from the
        <em>following</em> quarter.
        </p>
        """, border_color="#d29922")

        setup_s = st.number_input(
            "Reorder Trigger (s)",
            min_value=0, max_value=500,
            value=DEFAULT_S,
            step=5,
            key="md_setup_s",
            help="When inventory hits s or below, an order is triggered. Too low \u2192 stockout risk.",
        )
        setup_S = st.number_input(
            "Target Stock Level (S)",
            min_value=0, max_value=1000,
            value=DEFAULT_S_UPPER,
            step=5,
            key="md_setup_S",
            help="Order-up-to level. Too low \u2192 thin buffer. Too high \u2192 excess holding costs.",
        )

        if setup_s >= setup_S:
            st.error("Reorder Trigger (s) must be strictly less than Target Stock Level (S).")
            setup_valid = False
        else:
            setup_valid = True
            implied_order = max(0, setup_S - setup_s)
            st.success(
                f"When inventory hits {setup_s}, you'll order up to {implied_order} units "
                f"({setup_mix_primary}% primary / {100 - setup_mix_primary}% circular)."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        _sc = SCENARIOS[selected_scenario]
        _event_r = _sc.get("event_round")
        _max_cp = max(_sc["carbon_prices"])
        _min_cp = min(_sc["carbon_prices"])
        _lead_max = max(_sc["lead_times_primary"])
        _risk_items = []
        if _max_cp > _min_cp:
            _risk_items.append('<li style="color:#d29922;">Carbon pricing will escalate during this scenario</li>')
        elif _max_cp > 2:
            _risk_items.append(f'<li style="color:#d29922;">Carbon pricing is elevated throughout (${_max_cp:.0f}/kg CO&#x2082;e)</li>')
        if _lead_max > 1:
            _risk_items.append('<li style="color:#f85149;">Primary lead time may extend \u2014 pipeline management is critical</li>')
        if selected_scenario == "supplier_failure":
            _risk_items.append('<li style="color:#f85149;">Supplier availability is not guaranteed for the full 10 quarters</li>')
        if selected_scenario == "demand_surge":
            _risk_items.append('<li style="color:#d29922;">Demand conditions may shift materially during the simulation</li>')
        if selected_scenario == "seasonal":
            _risk_items.append('<li style="color:#d29922;">Demand follows a seasonal pattern \u2014 buffers must be timed carefully</li>')
        if selected_scenario == "known_shock":
            _risk_items.append('<li style="color:#58a6ff;">A confirmed supply event is expected mid-simulation</li>')
        if not _risk_items:
            _risk_items.append('<li style="color:#3fb950;">Standard operating conditions \u2014 no known disruptions</li>')

        _card(
            '<h4 style="color:#c9d1d9; margin-top:0;">Risk Indicators</h4>'
            '<ul style="margin:0; padding-left:1.2rem; font-size:0.85rem;">'
            + "".join(_risk_items)
            + '</ul>',
            border_color="#30363d",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    b_col, _, n_col = st.columns([1, 3, 1])
    with b_col:
        if st.button("\u2190 Back", width='stretch', key="md_back"):
            st.session_state["onboarding_step"] = 0
            st.session_state["needs_scroll"] = True
            st.session_state["scroll_counter"] = st.session_state.get("scroll_counter", 0) + 1
            st.rerun()
    with n_col:
        if st.button(
            "BEGIN SIMULATION \u2192",
            disabled=not setup_valid or combo_invalid,
            width='stretch',
            key="md_begin",
        ):
            state = init_game_state(seed=42, scenario=selected_scenario)
            state["s_reorder_point"] = setup_s
            state["S_order_up_to"] = setup_S
            state["sourcing_mix_pct"] = setup_mix_primary
            for k, v in state.items():
                st.session_state[k] = v
            st.session_state["game_mode"] = game_mode
            st.session_state["onboarding_complete"] = True
            st.session_state["shock_announcement_shown"] = False
            st.session_state["show_round_result"] = False
            st.session_state["round_result_data"] = {}
            st.rerun()


def _show_onboarding():
    _scroll_slot = st.empty()
    if st.session_state.get("needs_scroll"):
        st.session_state["needs_scroll"] = False
        with _scroll_slot:
            _scroll_to_top()
    if st.session_state["onboarding_step"] == 0:
        _briefing_room()
    else:
        _mission_dossier()


# ── Run onboarding if not complete ────────────────────────────────────────────
if not st.session_state.get("onboarding_complete", False):
    _show_onboarding()
    st.stop()


# ── Post-onboarding state ──────────────────────────────────────────────────────
game_mode = st.session_state.get("game_mode", "free_play")
_scenario_key = st.session_state.get("scenario", "base_game")
_sc_meta = SCENARIOS[_scenario_key]
event_round = _sc_meta.get("event_round")

# Force-lock sourcing mix for locked modes on every render
if game_mode == "primary_lock":
    st.session_state["sourcing_mix_pct"] = 100
elif game_mode == "circular_lock":
    st.session_state["sourcing_mix_pct"] = 0

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

# ── GAME_KEYS ──────────────────────────────────────────────────────────────────
GAME_KEYS = {
    "current_round", "game_over", "shock_triggered", "inventory",
    "pipeline", "s_reorder_point", "S_order_up_to", "sourcing_mix_pct",
    "history", "cumulative_sap", "cumulative_carbon", "rng",
    "manual_primary_override", "manual_circular_override",
    "scenario",
}


# ── Board Pulse ────────────────────────────────────────────────────────────────
def _board_pulse(history, gmode, s_reorder_point):
    if not history:
        return
    last = history[-1]
    recent = history[-3:] if len(history) >= 3 else history

    avg_profit = sum(h["round_profit"] for h in recent) / len(recent)
    if avg_profit > 1500:
        profit_color, profit_label = "#3fb950", "HEALTHY"
    elif avg_profit > 0:
        profit_color, profit_label = "#d29922", "MODERATE"
    else:
        profit_color, profit_label = "#f85149", "UNDER PRESSURE"

    if gmode == "primary_lock":
        esg_color, esg_label = "#f85149", "CRITICAL"
    elif gmode == "circular_lock":
        esg_color, esg_label = "#3fb950", "STRONG"
    else:
        recent_circ = [
            h["order_circular"] / (h["order_circular"] + h["order_primary"])
            if (h["order_circular"] + h["order_primary"]) > 0 else 0
            for h in recent
        ]
        avg_circ = sum(recent_circ) / len(recent_circ) if recent_circ else 0
        if avg_circ >= 0.5:
            esg_color, esg_label = "#3fb950", "STRONG"
        elif avg_circ >= 0.2:
            esg_color, esg_label = "#d29922", "MODERATE"
        else:
            esg_color, esg_label = "#f85149", "WEAK"

    recent_stockouts = sum(1 for h in recent if h["stockout_units"] > 0)
    inv = last["ending_inventory"]
    if recent_stockouts > 0:
        sec_color, sec_label = "#f85149", "AT RISK"
    elif inv <= s_reorder_point * 0.75:
        sec_color, sec_label = "#d29922", "WATCH"
    else:
        sec_color, sec_label = "#3fb950", "SECURE"

    def _dot(color, label, title):
        return (
            f'<div style="flex:1; text-align:center; padding:0.6rem;">'
            f'<div style="width:14px; height:14px; border-radius:50%; background:{color}; '
            f'margin:0 auto 0.4rem auto;"></div>'
            f'<div style="color:#8b949e; font-size:0.7rem; text-transform:uppercase; '
            f'letter-spacing:0.06em;">{title}</div>'
            f'<div style="color:{color}; font-size:0.78rem; font-weight:700;">{label}</div>'
            f'</div>'
        )

    st.markdown(
        '<div style="background:#161b22; border:1px solid #30363d; border-radius:8px; '
        'display:flex; margin-bottom:1rem;">'
        + _dot(profit_color, profit_label, "Profitability")
        + '<div style="width:1px; background:#30363d; margin:0.5rem 0;"></div>'
        + _dot(esg_color, esg_label, "ESG")
        + '<div style="width:1px; background:#30363d; margin:0.5rem 0;"></div>'
        + _dot(sec_color, sec_label, "Supply Security")
        + '</div>',
        unsafe_allow_html=True,
    )


# ── Round observation ──────────────────────────────────────────────────────────
def _round_observation(round_data, completed_round, scenario_key):
    stockout = round_data.get("stockout_units", 0)
    profit = round_data.get("round_profit", 0)
    circ_ordered = round_data.get("order_circular", 0)
    prim_ordered = round_data.get("order_primary", 0)
    inv = round_data.get("ending_inventory", 0)
    obs = []
    if stockout > 0:
        obs.append(f"Stockout of {stockout:.0f} units \u2014 ${stockout * 20:,.0f} in penalties.")
    if profit < 0:
        obs.append("Round profit was negative.")
    elif profit > 3000:
        obs.append("Strong round profit.")
    total_ord = circ_ordered + prim_ordered
    if total_ord > 0:
        circ_share = circ_ordered / total_ord
        if circ_share < 0.1 and scenario_key != "supplier_failure":
            obs.append("Low circular share this quarter \u2014 carbon exposure is elevated.")
        elif circ_share > 0.8:
            obs.append("High circular share \u2014 strong ESG trajectory.")
    if inv < 30:
        obs.append("Inventory is critically low entering the next quarter.")
    return " ".join(obs) if obs else "Quarter completed."


# ── Round result interstitial ──────────────────────────────────────────────────
def _show_round_result():
    _scroll_slot = st.empty()
    if st.session_state.get("needs_scroll"):
        st.session_state["needs_scroll"] = False
        with _scroll_slot:
            _scroll_to_top()

    rdata = st.session_state.get("round_result_data", {})
    completed = rdata.get("round", 0)
    scenario_key = st.session_state.get("scenario", "base_game")

    st.markdown(
        f'<div style="text-align:center; margin-bottom:1.5rem;">'
        f'<h2 style="color:#58a6ff; letter-spacing:0.08em;">Q{completed} RESULT</h2>'
        f'<p style="color:#8b949e; margin:0;">Quarter completed \u2014 review before proceeding</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Round Profit", f"${rdata.get('round_profit', 0):,.0f}")
    m2.metric("Cumulative SAP", f"${st.session_state.get('cumulative_sap', 0):,.0f}")
    m3.metric("Ending Inventory", f"{rdata.get('ending_inventory', 0):.0f} units")
    m4.metric("Carbon This Qtr", f"{rdata.get('total_carbon', 0):.1f} kg CO\u2082e")

    obs = _round_observation(rdata, completed, scenario_key)
    if obs:
        _card(
            f'<p style="color:#c9d1d9; margin:0; font-size:0.95rem;">{obs}</p>',
            border_color="#d29922",
        )

    next_round = st.session_state.get("current_round", 1)
    is_game_over = st.session_state.get("game_over", False)

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        label = "View Final Board Report \u2192" if is_game_over else f"Proceed to Q{next_round} \u2192"
        if st.button(label, width='stretch', key="rr_proceed"):
            st.session_state["show_round_result"] = False
            st.session_state["needs_scroll"] = True
            st.session_state["scroll_counter"] = st.session_state.get("scroll_counter", 0) + 1
            st.rerun()


# ── Shock announcement ────────────────────────────────────────────────────────
def _show_shock_announcement():
    _scroll_slot = st.empty()
    if st.session_state.get("needs_scroll"):
        st.session_state["needs_scroll"] = False
        with _scroll_slot:
            _scroll_to_top()

    scenario_key = st.session_state.get("scenario", "base_game")
    sc_meta = SCENARIOS[scenario_key]
    ev_round = sc_meta.get("event_round")
    banner_text = SCENARIO_BANNERS.get(scenario_key)

    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a0a0a,#21080a); '
        'border:2px solid #f85149; border-radius:12px; '
        'padding:2rem; text-align:center; margin-bottom:1.5rem;">'
        f'<h1 style="color:#f85149; letter-spacing:0.12em; margin:0;">EVENT ALERT</h1>'
        f'<p style="color:#8b949e; margin-top:0.5rem;">Q{ev_round} \u2014 conditions have changed</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if banner_text:
        st.error(banner_text)

    narratives = SCENARIO_NARRATIVES.get(scenario_key, SCENARIO_NARRATIVES["base_game"])
    if ev_round and ev_round in narratives:
        n = narratives[ev_round]
        _card(
            f'<h3 style="color:#f85149; margin-top:0;">{n["title"]}</h3>'
            f'<p style="color:#c9d1d9;">{n["body"]}</p>'
            f'<div style="background:#0d1117; border-left:3px solid #d29922; '
            f'padding:0.6rem 1rem; border-radius:0 6px 6px 0; font-size:0.85rem; color:#d29922;">'
            f'{n["tip"]}</div>',
            border_color="#f85149",
        )

    st.markdown("---")
    st.markdown(
        '<h2 style="color:#e6edf3; text-align:center;">Board Scorecard \u2014 Now Revealed</h2>'
        '<p style="color:#8b949e; text-align:center; margin-bottom:1.5rem;">'
        'The board has formalised its evaluation criteria for the end-of-year report.</p>',
        unsafe_allow_html=True,
    )

    _card_row(
        ("""
        <h4 style="color:#58a6ff; margin-top:0;">Final Grade Weights</h4>
        <table style="width:100%; border-collapse:collapse; font-family:monospace;">
          <tr><td style="color:#c9d1d9; padding:5px 0;">Sustainability-Adjusted Profit</td>
              <td style="color:#58a6ff; text-align:right; font-weight:700;">50%</td></tr>
          <tr><td style="color:#c9d1d9; padding:5px 0;">Average Circular Sourcing Mix</td>
              <td style="color:#3fb950; text-align:right; font-weight:700;">30%</td></tr>
          <tr><td style="color:#c9d1d9; padding:5px 0;">Stockout-Free Quarters</td>
              <td style="color:#d29922; text-align:right; font-weight:700;">20%</td></tr>
        </table>
        <hr style="border-color:#30363d; margin:0.8rem 0;">
        <table style="width:100%; border-collapse:collapse; font-family:monospace;">
          <tr><th style="color:#8b949e; text-align:left; padding:2px 0;">Grade</th>
              <th style="color:#8b949e; text-align:right; padding:2px 0;">Score</th></tr>
          <tr><td style="color:#ffd700; font-weight:700;">S</td><td style="text-align:right; color:#c9d1d9;">&ge; 90</td></tr>
          <tr><td style="color:#3fb950; font-weight:700;">A</td><td style="text-align:right; color:#c9d1d9;">75&ndash;89</td></tr>
          <tr><td style="color:#58a6ff; font-weight:700;">B</td><td style="text-align:right; color:#c9d1d9;">60&ndash;74</td></tr>
          <tr><td style="color:#d29922; font-weight:700;">C</td><td style="text-align:right; color:#c9d1d9;">40&ndash;59</td></tr>
          <tr><td style="color:#f85149; font-weight:700;">D</td><td style="text-align:right; color:#c9d1d9;">&lt; 40</td></tr>
        </table>
        """, "#58a6ff", 1),
        ("""
        <h4 style="color:#d29922; margin-top:0;">Remaining Quarters</h4>
        <p style="color:#8b949e; font-size:0.85rem; margin:0 0 0.8rem 0;">
        You now know what the board is measuring. Use the remaining quarters
        to maximise your composite score.
        </p>
        <ul style="color:#c9d1d9; font-size:0.85rem; margin:0; padding-left:1.2rem;">
          <li style="margin-bottom:0.5rem;">Shift sourcing mix toward circular to improve ESG score</li>
          <li style="margin-bottom:0.5rem;">Maintain adequate buffer to avoid stockouts (20% weight)</li>
          <li>SAP is cumulative &mdash; every quarter counts toward the 50% SAP component</li>
        </ul>
        """, "#d29922", 1),
    )

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        if st.button("Acknowledge & Continue \u2192", width='stretch', key="shock_ack"):
            st.session_state["shock_announcement_shown"] = True
            st.session_state["needs_scroll"] = True
            st.session_state["scroll_counter"] = st.session_state.get("scroll_counter", 0) + 1
            st.rerun()


# ── Final board presentation ───────────────────────────────────────────────────
def _show_board_presentation():
    _scroll_slot = st.empty()
    if st.session_state.get("needs_scroll"):
        st.session_state["needs_scroll"] = False
        with _scroll_slot:
            _scroll_to_top()

    history = st.session_state["history"]
    cumulative_sap = st.session_state["cumulative_sap"]
    score, grade = compute_sustainability_rating(history, cumulative_sap)
    p_star = compute_switching_point()

    grade_color = {"S": "#ffd700", "A": "#3fb950", "B": "#58a6ff",
                   "C": "#d29922", "D": "#f85149"}.get(grade, "#8b949e")
    verdict_map = {
        "S": "OUTSTANDING", "A": "COMMENDABLE", "B": "SATISFACTORY",
        "C": "NEEDS IMPROVEMENT", "D": "UNDER EXPECTATION",
    }
    verdict = verdict_map.get(grade, "COMPLETE")

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#161b22,#21262d); '
        f'border:1px solid {grade_color}; border-radius:12px; '
        f'padding:2rem; text-align:center; margin-bottom:1.5rem;">'
        f'<p style="color:#8b949e; margin:0; font-size:0.85rem; letter-spacing:0.1em;">'
        f'BOARD REPORT &mdash; NOVAPULSE ELECTRONICS &mdash; 10-QUARTER REVIEW</p>'
        f'<h1 style="color:{grade_color}; font-size:3rem; margin:0.5rem 0; letter-spacing:0.1em;">{grade}</h1>'
        f'<h2 style="color:{grade_color}; margin:0; letter-spacing:0.08em;">{verdict}</h2>'
        f'<p style="color:#8b949e; margin-top:0.5rem;">Composite score: {score:.1f} / 100</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    total_carbon = sum(h["total_carbon"] for h in history)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cumulative SAP", f"${cumulative_sap:,.0f}")
    m2.metric("Sustainability Grade", grade)
    m3.metric("Score", f"{score:.1f} / 100")
    m4.metric("Total Carbon", f"{total_carbon:,.0f} kg CO\u2082e")

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

        _sc_prices = _sc_meta["carbon_prices"]
        _max_carbon = max(_sc_prices)
        _min_carbon = min(_sc_prices)
        _carbon_changed = _max_carbon > _min_carbon
        _event_round_fin = _sc_meta.get("event_round")

        if _carbon_changed and _event_round_fin:
            _carbon_note = (
                f'Peak carbon price was <strong>${_max_carbon:.0f}/kg CO\u2082e</strong> &mdash; '
                f'<strong style="color:#3fb950;">{_max_carbon/p_star:.1f}&times;</strong> above P*. '
                f'From Q{_event_round_fin} onward, circular was strongly cost-justified.'
            )
        else:
            _carbon_note = (
                f'Carbon pricing was constant at <strong>${_max_carbon:.0f}/kg CO\u2082e</strong> '
                f'&mdash; already above P* (${p_star:.2f}/kg CO\u2082e) from Q1.'
            )

        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1rem;">
          <table style="width:100%; border-collapse:collapse; font-family:monospace;">
            <tr><td style="color:#8b949e; padding:4px 0;">Grade</td>
                <td style="color:{grade_color}; font-weight:700; font-size:1.4rem; text-align:right;">{grade}</td></tr>
            <tr><td style="color:#8b949e; padding:4px 0;">Composite Score</td>
                <td style="color:#c9d1d9; text-align:right;">{score:.1f} / 100</td></tr>
            <tr><td style="color:#8b949e; padding:4px 0;">Avg Circular Mix</td>
                <td style="color:#3fb950; text-align:right;">{circ_pct:.1f}%</td></tr>
            <tr><td style="color:#8b949e; padding:4px 0;">Stockout Quarters</td>
                <td style="color:#{'f85149' if stockout_rounds > 0 else '3fb950'}; text-align:right;">{stockout_rounds} / {TOTAL_ROUNDS}</td></tr>
            <tr><td style="color:#8b949e; padding:4px 0;">Total Carbon</td>
                <td style="color:#c9d1d9; text-align:right;">{total_carbon:,.0f} kg CO\u2082e</td></tr>
            <tr><td style="color:#8b949e; padding:4px 0;">Policy Changes</td>
                <td style="color:#{'d29922' if st.session_state.get('policy_changes', 0) > 0 else '3fb950'}; text-align:right;">{st.session_state.get('policy_changes', 0)}</td></tr>
          </table>
          <hr style="border-color:#30363d; margin:0.8rem 0;">
          <p style="color:#8b949e; font-size:0.82rem; margin:0;">
            <strong style="color:#58a6ff;">P* = ${p_star:.2f}/kg CO\u2082e</strong>
            &mdash; the carbon price at which circular becomes cheaper than primary on total cost.<br><br>
            {_carbon_note}
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Performance Exhibits")

    df_hist = pd.DataFrame(history)
    fig_sap = make_subplots(specs=[[{"secondary_y": True}]])
    _bar_colors = [
        "#f85149" if (event_round and r >= event_round) else "#58a6ff"
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
    if event_round is not None:
        fig_sap.add_vline(
            x=event_round - 0.5, line_dash="dash", line_color="#f85149", opacity=0.6,
            annotation_text="EVENT", annotation_font_color="#f85149",
        )
    fig_sap.update_layout(**PLOT_LAYOUT, title="Round Profit & Cumulative SAP")
    fig_sap.update_xaxes(title_text="Quarter")
    fig_sap.update_yaxes(title_text="Round Profit ($)", secondary_y=False,
                          gridcolor="#21262d", zerolinecolor="#30363d")
    fig_sap.update_yaxes(title_text="Cumulative SAP ($)", secondary_y=True,
                          gridcolor="#21262d")
    st.plotly_chart(fig_sap, width='stretch')

    col_donut, col_cost = st.columns(2)
    with col_donut:
        total_carbon_p = sum(h["carbon_primary"] for h in history)
        total_carbon_c = sum(h["carbon_circular"] for h in history)
        fig_donut = go.Figure(go.Pie(
            labels=["Primary", "Circular"],
            values=[total_carbon_p, total_carbon_c],
            hole=0.55,
            marker=dict(colors=["#f85149", "#3fb950"]),
            textfont=dict(family="monospace", color="#c9d1d9"),
        ))
        fig_donut.update_layout(**PLOT_LAYOUT, title="Carbon Attribution (kg CO\u2082e)")
        st.plotly_chart(fig_donut, width='stretch')

    with col_cost:
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
    _dq_max_carbon = max(_sc_meta["carbon_prices"])
    _dq_carbon_changed = _dq_max_carbon > min(_sc_meta["carbon_prices"])
    _lead_times = _sc_meta["lead_times_primary"]
    _lead_time_changed = len(set(_lead_times)) > 1

    if _dq_carbon_changed:
        _q1_body = (
            f"P* = **${p_star:.2f}/kg CO\u2082e**. The peak carbon price was "
            f"**${_dq_max_carbon:.0f}/kg CO\u2082e** \u2014 {_dq_max_carbon/p_star:.1f}\u00d7 above P*. "
            f"When did circular become economically rational, and did your strategy reflect this in time?"
        )
    else:
        _q1_body = (
            f"P* = **${p_star:.2f}/kg CO\u2082e**. Carbon was constant at **${_dq_max_carbon:.0f}/kg CO\u2082e** "
            f"\u2014 above P* from Q1. Did your sourcing mix reflect this cost advantage throughout?"
        )

    if _lead_time_changed and event_round:
        _q2 = ("2. Pipeline Risk & Lead Times",
               f"When the disruption hit in Q{event_round}, primary lead time doubled to 2 rounds. "
               "How did orders already in your pipeline shape your resilience? "
               "What would you pre-position differently if you played again?")
    else:
        _q2 = ("2. Pipeline & Demand Planning",
               "Primary lead time was constant at 1 round. "
               "How did the timing of your orders relative to demand changes affect inventory? "
               "What would a better pipeline planning approach look like?")

    questions = [
        ("1. The Switching Point", _q1_body),
        _q2,
        ("3. Yield Uncertainty & Over-ordering",
         "EcoReclaim's yield varied ~N(70%, 10%) each quarter. Did you build a yield buffer? "
         "What systematic approach could reduce the risk of under-receiving?"),
        ("4. Inventory Policy Design",
         "Reflect on your Reorder Trigger (s) and Target Stock Level (S). Were they well-calibrated "
         "for the demand pattern? Did you adjust proactively or reactively?"),
        ("5. Carbon vs. Cost Trade-off",
         "Did you prioritise SAP maximisation or carbon minimisation? How would a stricter "
         "carbon budget change your sourcing mix? At what carbon price would you switch entirely to circular?"),
    ]
    for title, body in questions:
        with st.expander(title):
            st.markdown(body)

    st.divider()
    with st.expander("Deep Analysis \u2014 Full Round History & Cost Breakdown", expanded=False):
        df_costs = df_hist.copy()
        for cname, csrc in [
            ("cum_p", "cost_primary"), ("cum_c", "cost_circular"),
            ("cum_h", "cost_holding"), ("cum_so", "cost_stockout"),
            ("cum_ct", "cost_carbon"),
        ]:
            df_costs[cname] = df_costs[csrc].cumsum()

        fig_cc = go.Figure()
        for lbl, col, clr in [
            ("Primary", "cum_p", "#58a6ff"),
            ("Circular", "cum_c", "#3fb950"),
            ("Holding", "cum_h", "#d29922"),
            ("Stockout", "cum_so", "#f85149"),
            ("Carbon Tax", "cum_ct", "#a371f7"),
        ]:
            fig_cc.add_trace(go.Scatter(
                x=df_costs["round"], y=df_costs[col], stackgroup="costs",
                name=lbl, mode="lines",
                line=dict(width=0.5, color=clr), fillcolor=clr, opacity=0.7,
            ))
        fig_cc.update_layout(**PLOT_LAYOUT, title="Cumulative Cost Breakdown ($)")
        fig_cc.update_xaxes(title_text="Quarter")
        fig_cc.update_yaxes(title_text="Cumulative Cost ($)")
        st.plotly_chart(fig_cc, width='stretch')

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

    st.divider()
    with st.expander("Policy Stress Test \u2014 Monte Carlo Analysis", expanded=False):
        student_s   = int(st.session_state["s_reorder_point"])
        student_S   = int(st.session_state["S_order_up_to"])
        student_mix = int(st.session_state["sourcing_mix_pct"])

        st.markdown(
            f"Tests your final policy (**s={student_s}, S={student_S}, "
            f"{student_mix}% primary**) across 1,000 random demand/yield scenarios, "
            f"then finds the near-optimal (s,S) via grid search (~120 combinations \u00d7 150 runs each)."
        )

        if st.button("Run Analysis", key="run_mc_btn"):
            st.session_state["mc_done"] = True

        if st.session_state.get("mc_done"):
            _mc_scenario = st.session_state.get("scenario", "base_game")
            with st.spinner("Running simulations\u2026"):
                opt_s, opt_S, _ = _cached_find_optimal(student_mix, _mc_scenario)
                student_saps, student_so = _cached_monte_carlo(
                    student_s, student_S, student_mix, 1000, _mc_scenario)
                optimal_saps, optimal_so = _cached_monte_carlo(
                    opt_s, opt_S, student_mix, 1000, _mc_scenario)
                default_saps, default_so = _cached_monte_carlo(
                    DEFAULT_S, DEFAULT_S_UPPER, student_mix, 1000, _mc_scenario)

            actual_pct = float((student_saps < cumulative_sap).mean() * 100)
            sap_gap    = float(optimal_saps.mean() - student_saps.mean())
            so_prob    = float((student_so > 0).mean() * 100)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Your result vs. your policy", f"{actual_pct:.0f}th pct",
                help="Where your actual game result falls within 1,000 simulated runs.")
            mc2.metric("P(stockout \u2265 1 quarter)", f"{so_prob:.0f}%",
                help="Fraction of simulated games with at least one stockout quarter.")
            mc3.metric("Mean SAP gap to near-optimal", f"${sap_gap:,.0f}",
                help="How much more the near-optimal policy earns on average.")

            fig_mc = go.Figure()
            for label, saps, color in [
                (f"Default  (s={DEFAULT_S}, S={DEFAULT_S_UPPER})", default_saps,  "#f85149"),
                (f"Your policy  (s={student_s}, S={student_S})",   student_saps,  "#58a6ff"),
                (f"Near-optimal  (s={opt_s}, S={opt_S})",          optimal_saps,  "#3fb950"),
            ]:
                fig_mc.add_trace(go.Histogram(
                    x=saps, name=label, histnorm="probability",
                    opacity=0.65, marker_color=color, nbinsx=40,
                ))
            fig_mc.add_vline(
                x=cumulative_sap, line_dash="dash", line_color="#d29922",
                annotation_text="Your actual result", annotation_font_color="#d29922",
            )
            fig_mc.update_layout(**PLOT_LAYOUT, barmode="overlay",
                                  title="SAP Distribution across 1,000 Random Scenarios")
            fig_mc.update_xaxes(title_text="Cumulative SAP ($)")
            fig_mc.update_yaxes(title_text="Probability", tickformat=".0%")
            st.plotly_chart(fig_mc, width="stretch")

            def _stats(saps, stockouts):
                return {
                    "Mean SAP ($)":         f"${saps.mean():,.0f}",
                    "10th pct ($)":         f"${np.percentile(saps, 10):,.0f}",
                    "90th pct ($)":         f"${np.percentile(saps, 90):,.0f}",
                    "P(\u22651 stockout qtr)": f"{(stockouts > 0).mean():.0%}",
                    "Avg stockout qtrs":    f"{stockouts.mean():.2f}",
                }

            stats_df = pd.DataFrame({
                f"Default (s={DEFAULT_S}, S={DEFAULT_S_UPPER})": _stats(default_saps, default_so),
                f"Your policy (s={student_s}, S={student_S})":   _stats(student_saps, student_so),
                f"Near-optimal (s={opt_s}, S={opt_S})":          _stats(optimal_saps, optimal_so),
            }).T.reset_index()
            stats_df.columns = ["Policy"] + list(stats_df.columns[1:])
            st.dataframe(stats_df, hide_index=True, width="stretch")

            st.info(
                f"Near-optimal for **{student_mix}% primary / {100 - student_mix}% circular**: "
                f"**s = {opt_s}**, **S = {opt_S}** (grid search step 25)"
            )

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
                  <td style="color:#c9d1d9; text-align:right;">{nickname or "&mdash;"}</td></tr>
              <tr><td style="color:#8b949e; padding:3px 0;">Grade</td>
                  <td style="color:{grade_color}; font-weight:700; text-align:right;">{grade}</td></tr>
              <tr><td style="color:#8b949e; padding:3px 0;">Score</td>
                  <td style="color:#c9d1d9; text-align:right;">{score:.1f} / 100</td></tr>
              <tr><td style="color:#8b949e; padding:3px 0;">Circular Mix</td>
                  <td style="color:#3fb950; text-align:right;">{circ_pct:.1f}%</td></tr>
              <tr><td style="color:#8b949e; padding:3px 0;">Stockout Quarters</td>
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
        if st.button("Submit Score \u2192", disabled=submit_disabled, width='stretch'):
            try:
                _supabase.table("scores").insert({
                    "session":         session_code.strip(),
                    "nickname":        nickname.strip(),
                    "grade":           grade,
                    "score":           round(score, 1),
                    "circular_mix":    round(circ_pct, 1),
                    "stockout_rounds": stockout_rounds,
                    "cumulative_sap":  round(cumulative_sap, 0),
                    "total_carbon":    round(total_carbon, 0),
                    "policy_changes":  st.session_state.get("policy_changes", 0),
                    "game_mode":       st.session_state.get("game_mode", "free_play"),
                    "scenario":        st.session_state.get("scenario", "base_game"),
                }).execute()
                st.session_state["score_submitted"] = True
                st.session_state["submitted_nickname"] = nickname.strip()
                st.session_state["_session_code"] = session_code.strip()
                st.rerun()
            except Exception as e:
                st.error(f"Submission failed: {e}")

    st.divider()
    if st.button("\u21ba  Start New Simulation", width='stretch'):
        _restart()


# ── Active game ────────────────────────────────────────────────────────────────
def _show_active_game():
    _scroll_slot = st.empty()
    if st.session_state.get("needs_scroll"):
        st.session_state["needs_scroll"] = False
        with _scroll_slot:
            _scroll_to_top()

    rnd = st.session_state["current_round"]
    history = st.session_state["history"]
    scenario_key = st.session_state.get("scenario", "base_game")
    sc_meta = SCENARIOS[scenario_key]
    ev_round = sc_meta.get("event_round")

    if st.session_state.get("shock_triggered"):
        narrative_border = "#f85149"
    elif ev_round and rnd >= ev_round - 1:
        narrative_border = "#d29922"
    else:
        narrative_border = "#30363d"

    hdr_col, badge_col = st.columns([5, 1])
    hdr_col.markdown(
        '<h2 style="color:#58a6ff; letter-spacing:0.06em; margin-bottom:0;">'
        'THE LOOPBACK INITIATIVE</h2>'
        '<p style="color:#8b949e; margin:0; font-size:0.85rem;">'
        'NovaPulse Electronics \u2014 Supply Chain Simulation</p>',
        unsafe_allow_html=True,
    )
    badge_col.markdown(
        f'<div style="background:#161b22; border:1px solid {narrative_border}; border-radius:8px; '
        f'padding:0.6rem; text-align:center; margin-top:0.6rem;">'
        f'<span style="color:{narrative_border}; font-size:1.3rem; font-weight:700;">Q{rnd}</span><br>'
        f'<span style="color:#8b949e; font-size:0.7rem;">of {TOTAL_ROUNDS}</span></div>',
        unsafe_allow_html=True,
    )

    banner_text = SCENARIO_BANNERS.get(scenario_key)
    if banner_text and st.session_state.get("shock_triggered"):
        st.error(banner_text)

    _narratives = SCENARIO_NARRATIVES.get(scenario_key, SCENARIO_NARRATIVES["base_game"])
    narrative = _narratives.get(rnd)
    if narrative:
        title = narrative["title"]
        body = narrative["body"]
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
                        yield_pct=last_yield, yield_comment=yield_comment, received=received)
                except (KeyError, ValueError):
                    pass
            else:
                body = (
                    "No circular orders were placed last quarter, so EcoReclaim's yield data "
                    "is not available. Urban mining yield follows N(70%, \u03c3=10%) \u2014 "
                    "if you order circular units, expect to receive ~70% of the quantity ordered."
                )

        with st.expander(f"Situation Report \u2014 {title}", expanded=True):
            st.markdown(body)
            st.markdown(
                f'<div style="background:#0d1117; border-left:3px solid #d29922; '
                f'padding:0.6rem 1rem; border-radius:0 6px 6px 0; font-size:0.85rem; color:#d29922;">'
                f'{narrative["tip"]}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    sit_col, dec_col = st.columns([4, 6])

    with sit_col:
        st.markdown(
            '<h3 style="color:#e6edf3; font-size:1rem; letter-spacing:0.08em; '
            'text-transform:uppercase; margin-bottom:0.8rem;">Situation Room</h3>',
            unsafe_allow_html=True,
        )

        if history:
            last = history[-1]
            k1, k2 = st.columns(2)
            k1.metric("Inventory", f"{st.session_state['inventory']:.0f} u",
                      delta=f"{last['ending_inventory'] - last['starting_inventory']:.0f}")
            k2.metric("Cumulative SAP", f"${st.session_state['cumulative_sap']:,.0f}")
            k3, k4 = st.columns(2)
            k3.metric("Last Round Profit", f"${last['round_profit']:,.0f}")
            k4.metric("Round Carbon", f"{last['total_carbon']:.1f} kg")

            _board_pulse(history, game_mode, st.session_state["s_reorder_point"])

            df_h = pd.DataFrame(history)
            fig_inv = go.Figure()
            fig_inv.add_trace(go.Scatter(
                x=df_h["round"], y=df_h["ending_inventory"],
                mode="lines+markers", name="Inventory",
                line=dict(color="#58a6ff", width=2), marker=dict(size=5),
            ))
            fig_inv.add_hline(
                y=st.session_state["s_reorder_point"], line_dash="dash", line_color="#d29922",
                annotation_text=f"s={st.session_state['s_reorder_point']}",
                annotation_font_color="#d29922",
            )
            fig_inv.update_layout(**PLOT_LAYOUT, title="Inventory")
            fig_inv.update_layout(margin=dict(l=40, r=10, t=35, b=30))
            fig_inv.update_xaxes(title_text="Quarter")
            fig_inv.update_yaxes(title_text="Units")
            st.plotly_chart(fig_inv, width='stretch')

            with st.expander("Carbon & Cost Charts", expanded=False):
                fig_carbon = go.Figure()
                fig_carbon.add_trace(go.Bar(
                    x=df_h["round"], y=df_h["carbon_primary"],
                    name="Primary", marker_color="#f85149",
                ))
                fig_carbon.add_trace(go.Bar(
                    x=df_h["round"], y=df_h["carbon_circular"],
                    name="Circular", marker_color="#3fb950",
                ))
                fig_carbon.update_layout(**PLOT_LAYOUT, barmode="stack",
                                          title="Carbon / Quarter (kg CO\u2082e)")
                fig_carbon.update_xaxes(title_text="Quarter")
                fig_carbon.update_yaxes(title_text="kg CO\u2082e")
                st.plotly_chart(fig_carbon, width='stretch')

                df_costs = df_h.copy()
                for cname, csrc in [
                    ("cum_p", "cost_primary"), ("cum_c", "cost_circular"),
                    ("cum_h", "cost_holding"), ("cum_so", "cost_stockout"),
                    ("cum_ct", "cost_carbon"),
                ]:
                    df_costs[cname] = df_costs[csrc].cumsum()

                fig_costs = go.Figure()
                for lbl, col, clr in [
                    ("Primary", "cum_p", "#58a6ff"),
                    ("Circular", "cum_c", "#3fb950"),
                    ("Holding", "cum_h", "#d29922"),
                    ("Stockout", "cum_so", "#f85149"),
                    ("Carbon Tax", "cum_ct", "#a371f7"),
                ]:
                    fig_costs.add_trace(go.Scatter(
                        x=df_costs["round"], y=df_costs[col], stackgroup="costs",
                        name=lbl, mode="lines",
                        line=dict(width=0.5, color=clr), fillcolor=clr, opacity=0.7,
                    ))
                fig_costs.update_layout(**PLOT_LAYOUT, title="Cumulative Costs ($)")
                fig_costs.update_xaxes(title_text="Quarter")
                fig_costs.update_yaxes(title_text="Cumulative Cost ($)")
                st.plotly_chart(fig_costs, width='stretch')
        else:
            st.info(
                f"Q{rnd} hasn't been played yet. "
                "Configure your policy and click Advance to begin."
            )

    with dec_col:
        st.markdown(
            '<h3 style="color:#e6edf3; font-size:1rem; letter-spacing:0.08em; '
            'text-transform:uppercase; margin-bottom:0.8rem;">Decision Console</h3>',
            unsafe_allow_html=True,
        )

        _eff_s = st.session_state["s_reorder_point"]
        _eff_S = st.session_state["S_order_up_to"]
        _mode_label = {
            "free_play": "Full Discretion",
            "primary_lock": "Primary Only",
            "circular_lock": "Circular Challenge",
        }.get(game_mode, game_mode)

        st.markdown(
            f'<div style="background:#161b22; border:1px solid #30363d; border-radius:6px; '
            f'padding:0.6rem 1rem; font-size:0.82rem; color:#8b949e; margin-bottom:1rem;">'
            f'Active policy: <strong style="color:#c9d1d9;">s={_eff_s}, S={_eff_S}</strong>'
            f' &nbsp;|&nbsp; Mandate: <strong style="color:#c9d1d9;">{_mode_label}</strong>'
            f' &nbsp;|&nbsp; Policy changes: '
            f'<strong style="color:#d29922;">{st.session_state.get("policy_changes", 0)}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )

        pol_l, pol_r = st.columns(2)
        with pol_l:
            st.markdown(
                '<p style="color:#d29922; font-size:0.85rem; font-weight:700; margin-bottom:0.3rem;">'
                'Reorder Trigger (s)</p>',
                unsafe_allow_html=True,
            )
            dc_s = st.number_input(
                "s", min_value=0, max_value=500, value=_eff_s, step=5,
                disabled=st.session_state["game_over"],
                key="dc_s", label_visibility="collapsed",
            )
        with pol_r:
            st.markdown(
                '<p style="color:#d29922; font-size:0.85rem; font-weight:700; margin-bottom:0.3rem;">'
                'Target Stock Level (S)</p>',
                unsafe_allow_html=True,
            )
            dc_S = st.number_input(
                "S", min_value=0, max_value=1000, value=_eff_S, step=5,
                disabled=st.session_state["game_over"],
                key="dc_S", label_visibility="collapsed",
            )

        inputs_valid = True
        if dc_s >= dc_S:
            st.error("Reorder Trigger (s) must be less than Target Stock Level (S).")
            inputs_valid = False
        elif dc_s != _eff_s or dc_S != _eff_S:
            st.caption(f"Active this quarter: s={_eff_s}, S={_eff_S} \u2014 changes apply from Q{rnd + 1}.")

        st.markdown(
            '<p style="color:#d29922; font-size:0.85rem; font-weight:700; '
            'margin:0.8rem 0 0.3rem 0;">Sourcing Mix</p>',
            unsafe_allow_html=True,
        )
        if game_mode == "primary_lock":
            st.markdown(
                '<div style="background:#161b22; border:1px solid #58a6ff; border-radius:6px; '
                'padding:0.5rem 0.8rem; color:#58a6ff; font-size:0.88rem; font-weight:700;">'
                'LOCKED: 100% Primary</div>',
                unsafe_allow_html=True,
            )
            mix_pct_primary = 100
        elif game_mode == "circular_lock":
            st.markdown(
                '<div style="background:#161b22; border:1px solid #3fb950; border-radius:6px; '
                'padding:0.5rem 0.8rem; color:#3fb950; font-size:0.88rem; font-weight:700;">'
                'LOCKED: 100% Circular</div>',
                unsafe_allow_html=True,
            )
            mix_pct_primary = 0
        else:
            dc_circ_slider = st.slider(
                "Circular %",
                min_value=0, max_value=100,
                value=100 - int(st.session_state["sourcing_mix_pct"]),
                step=5,
                disabled=st.session_state["game_over"],
                key="dc_mix",
                label_visibility="collapsed",
            )
            mix_pct_primary = 100 - dc_circ_slider
            mx1, mx2 = st.columns(2)
            mx1.metric("Primary", f"{mix_pct_primary}%")
            mx2.metric("Circular", f"{dc_circ_slider}%")

        with st.expander("Manual Order Override (optional)", expanded=False):
            st.caption("Leave at 0 to use (s,S) policy.")
            dc_man_primary = st.number_input(
                "Primary units to order", min_value=0, max_value=2000,
                value=0, step=10, key="dc_man_primary",
                disabled=st.session_state["game_over"],
            )
            dc_man_circular = st.number_input(
                "Circular units to order (before yield)", min_value=0, max_value=2000,
                value=0, step=10, key="dc_man_circular",
                disabled=st.session_state["game_over"],
            )
            dc_use_override = st.checkbox(
                "Apply manual override this quarter",
                value=False, key="dc_use_override",
                disabled=st.session_state["game_over"],
            )

        st.markdown("<br>", unsafe_allow_html=True)
        advance = False
        if not st.session_state["game_over"]:
            next_q = rnd + 1 if rnd < TOTAL_ROUNDS else "End"
            advance = st.button(
                f"\u25b6  Advance to Q{next_q}",
                disabled=not inputs_valid,
                width='stretch',
                key="dc_advance",
            )

        if st.button("\u21ba  Restart Simulation", width='stretch', key="dc_restart"):
            _restart()

    return advance, inputs_valid, mix_pct_primary


# ── Top-level routing ──────────────────────────────────────────────────────────
if st.session_state.get("show_round_result"):
    _show_round_result()
    st.stop()

if st.session_state.get("game_over"):
    _show_board_presentation()
    st.stop()

# Shock announcement: shown once when player enters event_round after shock triggered
if (
    event_round is not None
    and st.session_state.get("current_round") == event_round
    and st.session_state.get("shock_triggered")
    and not st.session_state.get("shock_announcement_shown")
):
    _show_shock_announcement()
    st.stop()

# Active game
advance, inputs_valid, mix_pct_primary = _show_active_game()

# ── Round advancement ──────────────────────────────────────────────────────────
if advance and inputs_valid and not st.session_state["game_over"]:
    _pend_s = st.session_state.get("dc_s", st.session_state["s_reorder_point"])
    _pend_S = st.session_state.get("dc_S", st.session_state["S_order_up_to"])

    if _pend_s != st.session_state["s_reorder_point"] or _pend_S != st.session_state["S_order_up_to"]:
        st.session_state["policy_changes"] = st.session_state.get("policy_changes", 0) + 1

    _use_ov = st.session_state.get("dc_use_override", False)
    if _use_ov:
        _mp = st.session_state.get("dc_man_primary", 0)
        _mc = st.session_state.get("dc_man_circular", 0)
        st.session_state["manual_primary_override"] = _mp if _mp > 0 else None
        st.session_state["manual_circular_override"] = _mc if _mc > 0 else None
    else:
        st.session_state["manual_primary_override"] = None
        st.session_state["manual_circular_override"] = None

    # Apply mix from slider
    if game_mode == "free_play":
        st.session_state["sourcing_mix_pct"] = mix_pct_primary

    game_state = {k: st.session_state[k] for k in GAME_KEYS if k in st.session_state}
    new_state = run_round(game_state)

    _history = new_state.get("history", [])
    _round_data = _history[-1] if _history else {}
    st.session_state["round_result_data"] = _round_data

    for k, v in new_state.items():
        st.session_state[k] = v

    # Promote pending (s,S)
    st.session_state["s_reorder_point"] = _pend_s
    st.session_state["S_order_up_to"] = _pend_S

    st.session_state["show_round_result"] = True
    st.session_state["needs_scroll"] = True
    st.session_state["scroll_counter"] = st.session_state.get("scroll_counter", 0) + 1
    st.rerun()
