"""
simulation.py — Pure game engine for The Loopback Initiative
No Streamlit dependency.
"""

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
PRIMARY_COST = 5.0          # $/unit
PRIMARY_CARBON = 8.0        # kg CO2e/unit
PRIMARY_LEAD_TIME_NORMAL = 1
PRIMARY_LEAD_TIME_SHOCK = 2

CIRCULAR_COST = 12.0        # $/unit
CIRCULAR_CARBON = 0.5       # kg CO2e/unit
CIRCULAR_LEAD_TIME = 1      # yield applied at order placement

DEMAND_MEAN = 100.0
DEMAND_STD = 20.0

CARBON_PRICE_NORMAL = 2.0   # $/kg CO2e
CARBON_PRICE_SHOCK = 8.0    # $/kg CO2e (round 5+)

HOLDING_COST = 1.0          # $/unit/round
STOCKOUT_PENALTY = 20.0     # $/unit short
REVENUE_PER_UNIT = 50.0     # $/unit sold

STARTING_INVENTORY = 150.0
SHOCK_ROUND = 5
TOTAL_ROUNDS = 10

DEFAULT_S = 60              # reorder point (deliberately low — causes stockout risk)
DEFAULT_S_UPPER = 130       # order-up-to level (deliberately tight)
DEFAULT_MIX = 80            # % primary (deliberately high — carbon-heavy)

# P* switching point: (12 - 5) / (8 - 0.5) = 7 / 7.5
P_STAR = (CIRCULAR_COST - PRIMARY_COST) / (PRIMARY_CARBON - CIRCULAR_CARBON)


# ── State initialisation ───────────────────────────────────────────────────────
def init_game_state(seed=None):
    """Return a fresh state dict."""
    rng = np.random.default_rng(seed)
    return {
        "current_round": 1,
        "game_over": False,
        "shock_triggered": False,
        "inventory": STARTING_INVENTORY,
        "pipeline": [],          # [{"arrive_round": int, "units": float, "source": str}]
        "s_reorder_point": DEFAULT_S,
        "S_order_up_to": DEFAULT_S_UPPER,
        "sourcing_mix_pct": DEFAULT_MIX,  # % from Primary
        "history": [],
        "cumulative_sap": 0.0,
        "cumulative_carbon": 0.0,
        "rng": rng,
    }


# ── Random generators ──────────────────────────────────────────────────────────
def generate_demand(rng):
    return float(max(0.0, rng.normal(DEMAND_MEAN, DEMAND_STD)))


def generate_circular_yield(rng):
    return float(np.clip(rng.normal(0.70, 0.10), 0.0, 1.0))


# ── Round parameters ───────────────────────────────────────────────────────────
def get_round_params(round_num):
    shock_active = round_num >= SHOCK_ROUND
    return {
        "lead_time_primary": PRIMARY_LEAD_TIME_SHOCK if shock_active else PRIMARY_LEAD_TIME_NORMAL,
        "carbon_price": CARBON_PRICE_SHOCK if shock_active else CARBON_PRICE_NORMAL,
        "shock_active": shock_active,
    }


# ── Order quantities ───────────────────────────────────────────────────────────
def compute_order_quantities(inventory, s, S, mix_pct,
                              manual_primary=None, manual_circular=None):
    """
    Apply (s,S) policy.  Manual overrides (non-None) take precedence.
    Returns (order_primary, order_circular) as floats.
    """
    if manual_primary is not None and manual_circular is not None:
        return float(manual_primary), float(manual_circular)

    if inventory <= s:
        total_order = max(0.0, S - inventory)
        pct_primary = mix_pct / 100.0
        order_primary = total_order * pct_primary
        order_circular = total_order * (1.0 - pct_primary)
    else:
        order_primary = 0.0
        order_circular = 0.0

    if manual_primary is not None:
        order_primary = float(manual_primary)
    if manual_circular is not None:
        order_circular = float(manual_circular)

    return order_primary, order_circular


# ── Pipeline management ────────────────────────────────────────────────────────
def process_arrivals(pipeline, current_round):
    """
    Split pipeline into arriving and remaining.
    Returns (arriving_units, remaining_pipeline).
    arriving_units: total units arriving this round (float).
    """
    arriving = [e for e in pipeline if e["arrive_round"] <= current_round]
    remaining = [e for e in pipeline if e["arrive_round"] > current_round]
    arriving_units = sum(e["units"] for e in arriving)
    return arriving_units, remaining


def place_orders(pipeline, current_round, order_primary, order_circular,
                 lead_time_primary, rng):
    """
    Apply circular yield, append new entries to pipeline.
    Returns (updated_pipeline, yield_factor).
    """
    yield_factor = generate_circular_yield(rng)
    new_pipeline = list(pipeline)

    if order_primary > 0:
        new_pipeline.append({
            "arrive_round": current_round + lead_time_primary,
            "units": order_primary,
            "source": "primary",
        })

    if order_circular > 0:
        effective_circular = order_circular * yield_factor
        new_pipeline.append({
            "arrive_round": current_round + CIRCULAR_LEAD_TIME,
            "units": effective_circular,
            "source": "circular",
        })

    return new_pipeline, yield_factor


# ── Core round runner ──────────────────────────────────────────────────────────
def run_round(state):
    """
    Pure function: takes state dict, returns updated state dict.
    Does NOT mutate the input.
    """
    state = dict(state)  # shallow copy; pipeline/history replaced below
    state["pipeline"] = list(state["pipeline"])
    state["history"] = list(state["history"])

    rng = state["rng"]
    round_num = state["current_round"]

    # 1. Round parameters
    params = get_round_params(round_num)
    lead_time_primary = params["lead_time_primary"]
    carbon_price = params["carbon_price"]

    # 2. Process pipeline arrivals
    arriving_units, remaining_pipeline = process_arrivals(state["pipeline"], round_num)

    # Track what arrived by source for cost attribution
    arriving_primary = sum(
        e["units"] for e in state["pipeline"]
        if e["arrive_round"] <= round_num and e["source"] == "primary"
    )
    arriving_circular = sum(
        e["units"] for e in state["pipeline"]
        if e["arrive_round"] <= round_num and e["source"] == "circular"
    )

    starting_inventory = state["inventory"] + arriving_units

    # 3. Generate demand
    demand = generate_demand(rng)

    # 4. Fulfil demand
    units_sold = min(starting_inventory, demand)
    stockout_units = max(0.0, demand - starting_inventory)
    ending_inventory = starting_inventory - units_sold

    # 5. Revenue
    revenue = units_sold * REVENUE_PER_UNIT

    # 6. Procurement costs (on units that arrived this round)
    cost_primary = arriving_primary * PRIMARY_COST
    cost_circular = arriving_circular * CIRCULAR_COST

    # 7. Carbon costs (on units that arrived)
    carbon_primary = arriving_primary * PRIMARY_CARBON
    carbon_circular = arriving_circular * CIRCULAR_CARBON
    total_carbon = carbon_primary + carbon_circular
    cost_carbon = total_carbon * carbon_price

    # 8. Holding cost (end-of-round inventory)
    cost_holding = ending_inventory * HOLDING_COST

    # 9. Stockout penalty
    cost_stockout = stockout_units * STOCKOUT_PENALTY

    # 10. Round SAP
    round_profit = (revenue
                    - cost_primary - cost_circular
                    - cost_holding - cost_stockout
                    - cost_carbon)

    new_cumulative_sap = state["cumulative_sap"] + round_profit
    new_cumulative_carbon = state["cumulative_carbon"] + total_carbon

    # 11. Compute orders for next round
    s = state["s_reorder_point"]
    S = state["S_order_up_to"]
    mix_pct = state["sourcing_mix_pct"]
    manual_primary = state.get("manual_primary_override")
    manual_circular = state.get("manual_circular_override")

    order_primary, order_circular = compute_order_quantities(
        ending_inventory, s, S, mix_pct, manual_primary, manual_circular
    )

    # 12. Place orders into pipeline
    new_pipeline, yield_factor = place_orders(
        remaining_pipeline, round_num,
        order_primary, order_circular,
        lead_time_primary, rng
    )

    # 13. Build history entry
    history_entry = {
        "round": round_num,
        "demand": demand,
        "units_sold": units_sold,
        "stockout_units": stockout_units,
        "starting_inventory": starting_inventory,
        "ending_inventory": ending_inventory,
        "order_primary": order_primary,
        "order_circular": order_circular,
        "circular_yield": yield_factor,
        "primary_received": arriving_primary,
        "circular_received": arriving_circular,
        "revenue": revenue,
        "cost_primary": cost_primary,
        "cost_circular": cost_circular,
        "cost_holding": cost_holding,
        "cost_stockout": cost_stockout,
        "cost_carbon": cost_carbon,
        "round_profit": round_profit,
        "cumulative_sap": new_cumulative_sap,
        "carbon_primary": carbon_primary,
        "carbon_circular": carbon_circular,
        "total_carbon": total_carbon,
        "lead_time_primary": lead_time_primary,
        "carbon_price": carbon_price,
    }

    new_history = state["history"] + [history_entry]
    next_round = round_num + 1
    game_over = next_round > TOTAL_ROUNDS

    state.update({
        "current_round": next_round,
        "game_over": game_over,
        "shock_triggered": next_round > SHOCK_ROUND or state["shock_triggered"],
        "inventory": ending_inventory,
        "pipeline": new_pipeline,
        "history": new_history,
        "cumulative_sap": new_cumulative_sap,
        "cumulative_carbon": new_cumulative_carbon,
        # clear manual overrides after use
        "manual_primary_override": None,
        "manual_circular_override": None,
    })
    return state


# ── Analytics ──────────────────────────────────────────────────────────────────
def compute_switching_point():
    """Carbon price P* at which circular becomes cheaper than primary."""
    return P_STAR


def compute_sustainability_rating(history, cumulative_sap):
    """
    Composite score (0-100):
      - SAP component    50%  (scaled vs. theoretical max)
      - Avg circular mix 30%
      - Stockout rounds  20%  (fewer is better)
    Returns (score, grade) where grade in S/A/B/C/D.
    """
    if not history:
        return 0.0, "D"

    # SAP component — normalise against a rough max of $30k
    sap_max = 30_000.0
    sap_score = min(100.0, max(0.0, (cumulative_sap / sap_max) * 100.0))

    # Circular mix component
    total_ordered = sum(
        h["order_primary"] + h["order_circular"] for h in history
    )
    total_circular = sum(h["order_circular"] for h in history)
    circular_pct = (total_circular / total_ordered * 100.0) if total_ordered > 0 else 0.0

    # Stockout component — % rounds with zero stockout
    stockout_rounds = sum(1 for h in history if h["stockout_units"] > 0)
    stockout_score = max(0.0, (1.0 - stockout_rounds / len(history)) * 100.0)

    composite = (
        0.50 * sap_score
        + 0.30 * circular_pct
        + 0.20 * stockout_score
    )

    if composite >= 90:
        grade = "S"
    elif composite >= 75:
        grade = "A"
    elif composite >= 60:
        grade = "B"
    elif composite >= 40:
        grade = "C"
    else:
        grade = "D"

    return composite, grade
