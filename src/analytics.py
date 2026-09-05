"""
Analytics Engine for the RM Intelligence Workbench.

Loads all dataset files and computes portfolio summaries, LTV analysis,
mandate drift, liquidity coverage, concentration risk, unrealised P&L,
and event attribution across the 5 snapshots.
"""

import json
import os
from datetime import datetime, timedelta

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SNAPSHOT_DATES = ["2025-12-31", "2026-02-27", "2026-03-31", "2026-06-30", "2026-08-26"]
TODAY = "2026-08-26"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_all_data() -> dict:
    """Load all dataset files into a unified data context dictionary."""
    data = {}
    data["clients"] = pd.read_csv(os.path.join(DATA_DIR, "clients.csv"))
    data["portfolios"] = pd.read_csv(os.path.join(DATA_DIR, "portfolios.csv"))
    data["holdings"] = pd.read_csv(os.path.join(DATA_DIR, "holdings.csv"))
    data["instruments"] = pd.read_csv(os.path.join(DATA_DIR, "instruments.csv"))
    data["mandates"] = pd.read_csv(os.path.join(DATA_DIR, "mandates.csv"))
    data["transactions"] = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    data["credit_facilities"] = pd.read_csv(os.path.join(DATA_DIR, "credit_facilities.csv"))
    data["commitments"] = pd.read_csv(os.path.join(DATA_DIR, "commitments.csv"))
    data["planned_cash_needs"] = pd.read_csv(os.path.join(DATA_DIR, "planned_cash_needs.csv"))
    data["market_context"] = pd.read_csv(os.path.join(DATA_DIR, "market_context.csv"))
    data["event_log"] = pd.read_csv(os.path.join(DATA_DIR, "event_log.csv"))

    with open(os.path.join(DATA_DIR, "rm_notes.json"), encoding="utf-8") as f:
        data["rm_notes"] = json.load(f)

    return data


# ---------------------------------------------------------------------------
# Portfolio Summaries
# ---------------------------------------------------------------------------

def compute_portfolio_summary(data: dict) -> pd.DataFrame:
    """
    Compute total AUM per client at each snapshot, plus YTD change.
    Returns a DataFrame indexed by client_id with AUM at each snapshot and returns.
    """
    portfolios = data["portfolios"]
    clients = data["clients"]

    aum_cols = [c for c in portfolios.columns if c.startswith("aum_20")]
    summary_rows = []

    for _, client in clients.iterrows():
        cid = client["client_id"]
        client_pfs = portfolios[portfolios["client_id"] == cid]

        row = {"client_id": cid, "client_name": client["client_name"]}
        for col in aum_cols:
            date_str = col.replace("aum_", "")
            row[f"aum_{date_str}"] = client_pfs[col].sum()

        row["aum_usd_current"] = client_pfs["aum_usd_current"].sum()

        # YTD change
        baseline = row.get("aum_2025-12-31", 0)
        current = row.get("aum_2026-08-26", 0)
        if baseline and baseline > 0:
            row["ytd_change_pct"] = ((current - baseline) / baseline) * 100
            row["ytd_change_abs"] = current - baseline
        else:
            row["ytd_change_pct"] = 0
            row["ytd_change_abs"] = 0

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def get_client_portfolio_timeseries(data: dict, client_id: str) -> pd.DataFrame:
    """Get AUM timeseries for all portfolios of a client across 5 snapshots."""
    portfolios = data["portfolios"]
    client_pfs = portfolios[portfolios["client_id"] == client_id]

    rows = []
    aum_cols = [c for c in portfolios.columns if c.startswith("aum_20")]
    for _, pf in client_pfs.iterrows():
        for col in aum_cols:
            date_str = col.replace("aum_", "")
            rows.append({
                "portfolio_id": pf["portfolio_id"],
                "portfolio_name": pf["portfolio_name"],
                "mandate_name": pf["mandate_name"],
                "service_model": pf["service_model"],
                "snapshot_date": date_str,
                "aum": pf[col],
                "base_currency": pf["base_currency"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LTV Analysis
# ---------------------------------------------------------------------------

def compute_ltv_analysis(data: dict) -> pd.DataFrame:
    """
    Analyze Loan-to-Value ratios across snapshots for all credit facilities.
    Returns a DataFrame with LTV history, breach flags, and headroom.
    """
    facilities = data["credit_facilities"]
    clients = data["clients"]

    results = []
    for _, fac in facilities.iterrows():
        client_name = clients.loc[
            clients["client_id"] == fac["client_id"], "client_name"
        ].iloc[0] if len(clients[clients["client_id"] == fac["client_id"]]) > 0 else "Unknown"

        row = {
            "facility_id": fac["facility_id"],
            "client_id": fac["client_id"],
            "client_name": client_name,
            "facility_type": fac["facility_type"],
            "facility_ccy": fac["facility_ccy"],
            "credit_limit": fac["credit_limit"],
            "margin_call_ltv_pct": fac["margin_call_ltv_pct"],
            "utilisation_pct_current": fac["utilisation_pct_current"],
        }

        # LTV at each snapshot
        breached = False
        max_ltv = 0
        for date in SNAPSHOT_DATES:
            ltv_col = f"ltv_pct_{date}"
            headroom_col = f"headroom_{date}"
            drawn_col = f"drawn_{date}"
            if ltv_col in fac.index:
                ltv_val = fac[ltv_col]
                row[f"ltv_{date}"] = ltv_val
                row[f"headroom_{date}"] = fac.get(headroom_col, 0)
                row[f"drawn_{date}"] = fac.get(drawn_col, 0)
                if ltv_val > max_ltv:
                    max_ltv = ltv_val
                if ltv_val >= fac["margin_call_ltv_pct"]:
                    breached = True

        row["max_ltv"] = max_ltv
        row["ever_breached"] = breached
        row["current_ltv"] = fac.get(f"ltv_pct_{TODAY}", 0)
        row["ltv_headroom_pct"] = fac["margin_call_ltv_pct"] - row["current_ltv"]
        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Mandate Drift Detection
# ---------------------------------------------------------------------------

def compute_mandate_drift(data: dict, portfolio_id: str = None) -> pd.DataFrame:
    """
    Compare actual asset class weights vs mandate bands for all portfolios
    (or a specific one). Flags breaches.
    """
    holdings = data["holdings"]
    portfolios = data["portfolios"]
    mandates = data["mandates"]

    # Filter to today's snapshot
    current_holdings = holdings[holdings["snapshot_date"] == TODAY]
    if portfolio_id:
        current_holdings = current_holdings[current_holdings["portfolio_id"] == portfolio_id]

    results = []
    pf_ids = current_holdings["portfolio_id"].unique()

    for pid in pf_ids:
        pf_info = portfolios[portfolios["portfolio_id"] == pid]
        if pf_info.empty:
            continue
        pf_info = pf_info.iloc[0]

        # Skip custody accounts (no mandate)
        if pf_info["service_model"] == "Custody":
            continue

        mandate_code = pf_info["mandate_code"]
        mandate_bands = mandates[mandates["mandate_code"] == mandate_code]

        pf_holdings = current_holdings[current_holdings["portfolio_id"] == pid]
        total_mv = pf_holdings["market_value_usd"].sum()
        if total_mv == 0:
            continue

        # Aggregate weights by asset class
        ac_weights = pf_holdings.groupby("asset_class")["weight_pct"].sum()

        for _, band in mandate_bands.iterrows():
            ac = band["asset_class"]
            actual_weight = ac_weights.get(ac, 0)
            min_pct = band["min_pct"]
            max_pct = band["max_pct"]
            target_pct = band["target_pct"]

            breach = "None"
            breach_amount = 0
            if actual_weight < min_pct:
                breach = "Under"
                breach_amount = min_pct - actual_weight
            elif actual_weight > max_pct:
                breach = "Over"
                breach_amount = actual_weight - max_pct

            results.append({
                "portfolio_id": pid,
                "client_id": pf_info["client_id"],
                "portfolio_name": pf_info["portfolio_name"],
                "mandate_code": mandate_code,
                "mandate_name": pf_info["mandate_name"],
                "asset_class": ac,
                "actual_weight_pct": round(actual_weight, 2),
                "min_pct": min_pct,
                "target_pct": target_pct,
                "max_pct": max_pct,
                "breach": breach,
                "breach_amount_pct": round(breach_amount, 2),
            })

    return pd.DataFrame(results)


def get_mandate_breaches(data: dict) -> pd.DataFrame:
    """Return only the asset classes that breach their mandate bands."""
    drift = compute_mandate_drift(data)
    if drift.empty:
        return drift
    return drift[drift["breach"] != "None"].sort_values("breach_amount_pct", ascending=False)


# ---------------------------------------------------------------------------
# Liquidity Coverage
# ---------------------------------------------------------------------------

def compute_liquidity_coverage(data: dict) -> pd.DataFrame:
    """
    Match planned cash needs + commitments against Daily-liquid holdings.
    Returns a DataFrame per client with liquidity gap analysis.
    """
    clients = data["clients"]
    holdings = data["holdings"]
    cash_needs = data["planned_cash_needs"]
    commitments = data["commitments"]

    current_holdings = holdings[holdings["snapshot_date"] == TODAY]

    results = []
    for _, client in clients.iterrows():
        cid = client["client_id"]

        # Daily liquid assets (USD)
        client_holdings = current_holdings[current_holdings["client_id"] == cid]
        daily_liquid = client_holdings[
            client_holdings["liquidity_tier"] == "Daily"
        ]["market_value_usd"].sum()

        weekly_liquid = client_holdings[
            client_holdings["liquidity_tier"].isin(["Daily", "Weekly"])
        ]["market_value_usd"].sum()

        monthly_liquid = client_holdings[
            client_holdings["liquidity_tier"].isin(["Daily", "Weekly", "Monthly"])
        ]["market_value_usd"].sum()

        total_aum = client_holdings["market_value_usd"].sum()

        # Cash needs within 6 months
        client_needs = cash_needs[cash_needs["client_id"] == cid]
        near_term_needs = 0
        all_needs = 0
        for _, need in client_needs.iterrows():
            amount = need["amount"]
            # Rough USD conversion for non-USD currencies
            ccy = need["currency"]
            fx = _get_fx_rate(ccy, data)
            amount_usd = amount / fx if fx else amount
            all_needs += amount_usd

            due_from = need.get("due_from", "")
            if isinstance(due_from, str) and due_from:
                try:
                    due_dt = datetime.strptime(due_from, "%Y-%m-%d")
                    if due_dt <= datetime(2026, 9, 25):
                        near_term_needs += amount_usd
                except (ValueError, TypeError):
                    pass

        # Uncalled commitments
        client_comms = commitments[commitments["client_id"] == cid]
        uncalled = 0
        for _, comm in client_comms.iterrows():
            uncalled += comm["uncalled"]

        total_obligations = near_term_needs + uncalled
        liquidity_gap = daily_liquid - total_obligations

        results.append({
            "client_id": cid,
            "client_name": client["client_name"],
            "total_aum_usd": total_aum,
            "daily_liquid_usd": daily_liquid,
            "weekly_liquid_usd": weekly_liquid,
            "monthly_liquid_usd": monthly_liquid,
            "near_term_needs_usd": near_term_needs,
            "uncalled_commitments_usd": uncalled,
            "total_obligations_usd": total_obligations,
            "liquidity_gap_usd": liquidity_gap,
            "daily_liquid_pct": (daily_liquid / total_aum * 100) if total_aum > 0 else 0,
            "coverage_ratio": (daily_liquid / total_obligations) if total_obligations > 0 else 999,
            "is_stressed": liquidity_gap < 0,
        })

    return pd.DataFrame(results)


def _get_fx_rate(ccy: str, data: dict) -> float:
    """Get approximate USD conversion rate for a currency using market_context."""
    if ccy == "USD":
        return 1.0
    market = data["market_context"]
    current = market[market["snapshot_date"] == TODAY]

    fx_map = {
        "SGD": ("USDSGD", True),    # SGD per USD → divide
        "HKD": ("USDHKD", True),
        "EUR": ("EURUSD", False),    # USD per EUR → multiply
        "CHF": ("USDCHF", True),
        "JPY": ("USDJPY", True),
        "GBP": ("GBPUSD", False),
        "CNH": ("USDCNH", True),
        "IDR": ("USDIDR", True),
        "THB": ("USDTHB", True),
        "INR": ("USDINR", True),
    }

    if ccy in fx_map:
        series_id, is_per_usd = fx_map[ccy]
        rate_row = current[current["series_id"] == series_id]
        if not rate_row.empty:
            rate = rate_row.iloc[0]["value"]
            return rate if is_per_usd else (1 / rate)
    return 1.0


# ---------------------------------------------------------------------------
# Concentration Risk
# ---------------------------------------------------------------------------

def compute_concentration_risk(data: dict) -> pd.DataFrame:
    """
    Identify single-name and sector concentration risks, including
    structured product look-through via underlying_reference.
    """
    holdings = data["holdings"]
    portfolios = data["portfolios"]
    instruments = data["instruments"]
    mandates = data["mandates"]

    current = holdings[holdings["snapshot_date"] == TODAY]
    results = []

    for pid in current["portfolio_id"].unique():
        pf_info = portfolios[portfolios["portfolio_id"] == pid]
        if pf_info.empty:
            continue
        pf_info = pf_info.iloc[0]

        # Skip custody accounts
        if pf_info["service_model"] == "Custody":
            continue

        mandate_code = pf_info["mandate_code"]
        mandate_limit = mandates[mandates["mandate_code"] == mandate_code]
        max_single = mandate_limit["max_single_position_pct"].iloc[0] if not mandate_limit.empty else 15.0

        pf_holdings = current[current["portfolio_id"] == pid]

        for _, h in pf_holdings.iterrows():
            inst_id = h["instrument_id"]
            inst = instruments[instruments["instrument_id"] == inst_id]
            if inst.empty:
                continue
            inst = inst.iloc[0]

            # Only check concentration on instruments where it applies
            conc_applies = inst.get("concentration_limit_applies", "N") == "Y"
            weight = h["weight_pct"]

            is_breach = conc_applies and weight > max_single

            # Look-through for structured products
            underlying = inst.get("underlying_reference", "")
            if pd.isna(underlying):
                underlying = ""

            results.append({
                "portfolio_id": pid,
                "client_id": pf_info["client_id"],
                "instrument_id": inst_id,
                "instrument_name": h["instrument_name"],
                "asset_class": h["asset_class"],
                "weight_pct": weight,
                "max_single_position_pct": max_single,
                "concentration_limit_applies": conc_applies,
                "is_breach": is_breach,
                "underlying_reference": underlying,
                "sustainability_excluded": inst.get("sustainability_excluded", "N"),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Unrealised P&L (Tax-Aware)
# ---------------------------------------------------------------------------

def compute_unrealised_pnl(data: dict) -> pd.DataFrame:
    """
    Compute unrealised P&L per client per holding for tax-loss harvesting analysis.
    Groups by tax_domicile for cross-border tax planning.
    """
    holdings = data["holdings"]
    clients = data["clients"]
    current = holdings[holdings["snapshot_date"] == TODAY]

    results = []
    for _, h in current.iterrows():
        cid = h["client_id"]
        client = clients[clients["client_id"] == cid]
        tax_dom = client.iloc[0]["tax_domicile"] if not client.empty else "Unknown"

        results.append({
            "client_id": cid,
            "portfolio_id": h["portfolio_id"],
            "instrument_id": h["instrument_id"],
            "instrument_name": h["instrument_name"],
            "asset_class": h["asset_class"],
            "market_value_usd": h["market_value_usd"],
            "unrealised_pnl_base": h["unrealised_pnl_base"],
            "unrealised_pnl_pct": h["unrealised_pnl_pct"],
            "tax_domicile": tax_dom,
            "is_loss": h["unrealised_pnl_base"] < 0,
        })

    return pd.DataFrame(results)


def get_tax_optimization_opportunities(data: dict, client_id: str) -> dict:
    """Get tax-loss harvesting opportunities for a specific client."""
    pnl = compute_unrealised_pnl(data)
    client_pnl = pnl[pnl["client_id"] == client_id]
    if client_pnl.empty:
        return {"losses": [], "gains": [], "net_pnl": 0}

    losses = client_pnl[client_pnl["is_loss"]].sort_values("unrealised_pnl_base")
    gains = client_pnl[~client_pnl["is_loss"]].sort_values("unrealised_pnl_base", ascending=False)

    return {
        "losses": losses.to_dict("records"),
        "gains": gains.to_dict("records"),
        "total_losses": losses["unrealised_pnl_base"].sum(),
        "total_gains": gains["unrealised_pnl_base"].sum(),
        "net_pnl": client_pnl["unrealised_pnl_base"].sum(),
        "tax_domicile": client_pnl.iloc[0]["tax_domicile"] if not client_pnl.empty else "Unknown",
    }


def get_lookthrough_exposure(data: dict, client_id: str, snapshot_date: str = TODAY) -> pd.DataFrame:
    """Aggregate direct and structured-product exposure, including underlying references."""
    holdings = data["holdings"]
    instruments = data["instruments"]
    client_holdings = holdings[
        (holdings["client_id"] == client_id) &
        (holdings["snapshot_date"] == snapshot_date)
    ].copy()
    if client_holdings.empty:
        return pd.DataFrame()

    instrument_cols = ["instrument_id", "underlying_reference"]
    joined = client_holdings.merge(
        instruments[instrument_cols], on="instrument_id", how="left"
    )
    joined["underlying_reference"] = joined["underlying_reference"].fillna("")
    joined["exposure_name"] = joined.apply(
        lambda row: row["underlying_reference"] or row["instrument_name"], axis=1
    )
    return joined.groupby("exposure_name", as_index=False).agg(
        market_value_usd=("market_value_usd", "sum"),
        direct_positions=("instrument_id", "count"),
        source_instruments=("instrument_name", lambda values: ", ".join(sorted(set(values)))),
    ).sort_values("market_value_usd", ascending=False)


def simulate_continuous_monitoring(
    clients_df: pd.DataFrame,
    holdings_df: pd.DataFrame,
    credit_df: pd.DataFrame,
    commitments_df: pd.DataFrame,
    planned_cash_df: pd.DataFrame,
    mandates_df: pd.DataFrame | None = None,
) -> list[dict]:
    """Scan every client and snapshot and emit deterministic proactive alerts."""
    alerts = []
    client_names = clients_df.set_index("client_id")["client_name"].to_dict()

    for snapshot_date in SNAPSHOT_DATES:
        snapshot_holdings = holdings_df[holdings_df["snapshot_date"] == snapshot_date]

        for _, facility in credit_df.iterrows():
            ltv = facility.get(f"ltv_pct_{snapshot_date}")
            trigger = facility.get("margin_call_ltv_pct")
            if pd.isna(ltv) or pd.isna(trigger):
                continue
            headroom = trigger - ltv
            if ltv >= trigger or headroom <= 3:
                label = "[URGENT: LTV BREACH]" if ltv >= trigger else "[URGENT: LTV BREACH]"
                alerts.append({
                    "snapshot_date": snapshot_date,
                    "client_id": facility["client_id"],
                    "client_name": client_names.get(facility["client_id"], "Unknown"),
                    "alert_type": label,
                    "severity": "URGENT" if ltv >= trigger else "HIGH",
                    "message": f"{facility['facility_id']} LTV {ltv:.1f}% vs {trigger:.1f}% trigger ({headroom:+.1f}% headroom).",
                })

        for client_id in clients_df["client_id"]:
            client_holdings = snapshot_holdings[snapshot_holdings["client_id"] == client_id]
            daily_liquid = client_holdings[
                client_holdings["liquidity_tier"] == "Daily"
            ]["market_value_usd"].sum()
            near_term_needs = 0.0
            client_needs = planned_cash_df[planned_cash_df["client_id"] == client_id]
            for _, need in client_needs.iterrows():
                due_from = str(need.get("due_from", ""))
                if due_from and due_from <= "2026-09-25":
                    near_term_needs += float(need.get("amount", 0))
            uncalled = commitments_df.loc[
                commitments_df["client_id"] == client_id, "uncalled"
            ].sum()
            liquidity_gap = daily_liquid - near_term_needs - uncalled
            if liquidity_gap < 0:
                alerts.append({
                    "snapshot_date": snapshot_date,
                    "client_id": client_id,
                    "client_name": client_names.get(client_id, "Unknown"),
                    "alert_type": "[LIQUIDITY WARNING]",
                    "severity": "HIGH",
                    "message": f"Daily liquid assets are ${daily_liquid:,.0f} against ${near_term_needs + uncalled:,.0f} near-term needs and uncalled commitments; gap ${liquidity_gap:,.0f}.",
                })

        # Mandate drift is evaluated from the same snapshot rather than only today.
        for client_id in clients_df["client_id"]:
            client_holdings = snapshot_holdings[snapshot_holdings["client_id"] == client_id]
            client_portfolios = client_holdings["portfolio_id"].unique()
            for portfolio_id in client_portfolios:
                portfolio_holdings = client_holdings[client_holdings["portfolio_id"] == portfolio_id]
                if mandates_df is not None and "mandate_code" in mandates_df.columns:
                    mandate_code = portfolio_holdings["mandate_code"].iloc[0] if "mandate_code" in portfolio_holdings else None
                    bands = mandates_df[mandates_df["mandate_code"] == mandate_code]
                    actuals = portfolio_holdings.groupby("asset_class")["weight_pct"].sum()
                    drift_rows = bands[
                        (bands["asset_class"].map(actuals).fillna(0) < bands["min_pct"]) |
                        (bands["asset_class"].map(actuals).fillna(0) > bands["max_pct"])
                    ]
                else:
                    actuals = portfolio_holdings.groupby("asset_class")["weight_pct"].sum()
                    drift_rows = actuals[(actuals > 60) | (actuals < 5)]
                if len(drift_rows) > 0:
                    alerts.append({
                        "snapshot_date": snapshot_date,
                        "client_id": client_id,
                        "client_name": client_names.get(client_id, "Unknown"),
                        "alert_type": "[MANDATE DRIFT]",
                        "severity": "MEDIUM",
                        "message": f"Portfolio {portfolio_id} has {len(drift_rows)} asset-class band breach(es); review mandate alignment.",
                    })

            losses = client_holdings[client_holdings["unrealised_pnl_base"] < 0]
            loss_value = losses["unrealised_pnl_base"].sum()
            if loss_value < 0:
                alerts.append({
                    "snapshot_date": snapshot_date,
                    "client_id": client_id,
                    "client_name": client_names.get(client_id, "Unknown"),
                    "alert_type": "[TAX OPTIMIZATION]",
                    "severity": "LOW",
                    "message": f"${abs(loss_value):,.0f} of unrealised losses may warrant tax-aware review, subject to domicile and advisor confirmation.",
                })

    return alerts


def get_client_analytics_context(data: dict, client_id: str) -> dict:
    """Return the exact derived analytics used by the client deep-dive and chat."""
    summary = compute_portfolio_summary(data)
    client_summary = summary[summary["client_id"] == client_id]
    ltv = compute_ltv_analysis(data)
    liquidity = compute_liquidity_coverage(data)
    drift = compute_mandate_drift(data)
    tax = get_tax_optimization_opportunities(data, client_id)
    alerts = simulate_continuous_monitoring(
        data["clients"],
        data["holdings"],
        data["credit_facilities"],
        data["commitments"],
        data["planned_cash_needs"],
    )

    return {
        "portfolio_summary": client_summary.to_dict("records"),
        "ltv": ltv[ltv["client_id"] == client_id].to_dict("records"),
        "liquidity": liquidity[liquidity["client_id"] == client_id].to_dict("records"),
        "mandate_drift": drift[drift["client_id"] == client_id].to_dict("records"),
        "tax_optimization": tax,
        "lookthrough_exposure": get_lookthrough_exposure(data, client_id).to_dict("records"),
        "monitoring_alerts": [
            alert for alert in alerts if alert["client_id"] == client_id
        ],
    }


# ---------------------------------------------------------------------------
# Event Attribution
# ---------------------------------------------------------------------------

# Mapping from event transmission channels to instrument characteristics
TRANSMISSION_MAP = {
    "Energy": {"sector": ["Energy"], "asset_class": [], "sub_asset_class": []},
    "Gold": {"sector": ["Gold"], "asset_class": ["Commodities"], "sub_asset_class": ["Precious Metals"]},
    "precious metals": {"sector": ["Gold"], "asset_class": ["Commodities"], "sub_asset_class": ["Precious Metals"]},
    "US technology": {"sector": ["Information Technology"], "region": ["North America"]},
    "concentrated equity": {"asset_class": ["Equity"]},
    "collateralised lending": {},
    "Duration": {"asset_class": ["Fixed Income"]},
    "Long-duration fixed income": {"asset_class": ["Fixed Income"], "sub_asset_class": ["Government Bond"]},
    "rate-sensitive credit": {"asset_class": ["Fixed Income"]},
    "defence": {"sector": ["Industrials"]},
    "airlines": {"sector": ["Industrials"]},
    "shipping": {"sector": ["Industrials"]},
    "transport": {"sector": ["Industrials"]},
    "EM credit": {"asset_class": ["Fixed Income"], "sub_asset_class": ["Emerging Market Debt", "High Yield Credit"]},
    "safe havens": {"asset_class": ["Commodities", "Cash and Equivalents"]},
    "European fixed income": {"asset_class": ["Fixed Income"], "region": ["Europe"]},
    "EUR assets": {"region": ["Europe"]},
    "Private credit": {"sub_asset_class": ["Private Credit"]},
    "semi-liquid alternatives": {"asset_class": ["Alternatives"]},
    "inflation-sensitive assets": {"sub_asset_class": ["Inflation Linked"]},
    "growth equity valuations": {"asset_class": ["Equity"]},
    "insurance": {},
    "Gulf exposure": {"region": ["Middle East", "Asia"]},
    "LNG": {"sector": ["Energy"]},
    "Gulf credit": {"region": ["Middle East"]},
    "oil-linked structured products": {"asset_class": ["Structured Products"]},
}


def attribute_portfolio_changes(data: dict, client_id: str) -> list[dict]:
    """
    For a given client, attribute position-level changes between snapshots
    to events in event_log.csv via transmission channel matching.
    """
    holdings = data["holdings"]
    events = data["event_log"]
    instruments = data["instruments"]

    client_holdings = holdings[holdings["client_id"] == client_id]

    # Compare each consecutive snapshot pair
    attributions = []
    for i in range(len(SNAPSHOT_DATES) - 1):
        date_from = SNAPSHOT_DATES[i]
        date_to = SNAPSHOT_DATES[i + 1]

        h_from = client_holdings[client_holdings["snapshot_date"] == date_from]
        h_to = client_holdings[client_holdings["snapshot_date"] == date_to]

        # Calculate changes per holding
        for _, h2 in h_to.iterrows():
            inst_id = h2["instrument_id"]
            pid = h2["portfolio_id"]
            h1_match = h_from[
                (h_from["instrument_id"] == inst_id) & (h_from["portfolio_id"] == pid)
            ]

            if h1_match.empty:
                mv_change = h2["market_value_usd"]
            else:
                mv_change = h2["market_value_usd"] - h1_match.iloc[0]["market_value_usd"]

            if abs(mv_change) < 100:
                continue

            # Find matching events in this period
            inst_info = instruments[instruments["instrument_id"] == inst_id]
            if inst_info.empty:
                continue
            inst_info = inst_info.iloc[0]

            period_events = events[
                (events["event_date"] >= date_from) & (events["event_date"] <= date_to)
            ]

            matched_events = []
            for _, evt in period_events.iterrows():
                transmission = str(evt.get("primary_transmission", ""))
                channels = [ch.strip() for ch in transmission.split(",")]

                for channel in channels:
                    if _channel_matches_instrument(channel, inst_info):
                        matched_events.append({
                            "event_date": evt["event_date"],
                            "description": evt["description"],
                            "severity": evt["severity"],
                            "transmission": transmission,
                        })
                        break

            if matched_events:
                attributions.append({
                    "period": f"{date_from} -> {date_to}",
                    "instrument_name": h2["instrument_name"],
                    "instrument_id": inst_id,
                    "asset_class": h2["asset_class"],
                    "mv_change_usd": round(mv_change, 2),
                    "mv_change_pct": round(
                        (mv_change / h1_match.iloc[0]["market_value_usd"] * 100)
                        if not h1_match.empty and h1_match.iloc[0]["market_value_usd"] != 0
                        else 0, 2
                    ),
                    "events": matched_events,
                })

    return attributions


def _channel_matches_instrument(channel: str, inst: pd.Series) -> bool:
    """Check if an event transmission channel matches an instrument."""
    channel_lower = channel.lower().strip()

    # Direct keyword matching against instrument properties
    sector = str(inst.get("sector", "")).lower()
    asset_class = str(inst.get("asset_class", "")).lower()
    sub_asset_class = str(inst.get("sub_asset_class", "")).lower()
    region = str(inst.get("region", "")).lower()
    name = str(inst.get("instrument_name", "")).lower()
    underlying = str(inst.get("underlying_reference", "")).lower()

    # Check the transmission map
    for key, criteria in TRANSMISSION_MAP.items():
        if key.lower() in channel_lower:
            matches = True
            for field, values in criteria.items():
                if not values:
                    continue
                field_val = str(inst.get(field, "")).lower()
                if not any(v.lower() in field_val for v in values):
                    matches = False
                    break
            if matches and criteria:
                return True

    # Fallback: direct text matching
    if channel_lower in sector or channel_lower in asset_class:
        return True
    if channel_lower in name or channel_lower in underlying:
        return True

    return False


# ---------------------------------------------------------------------------
# Sustainability Screening
# ---------------------------------------------------------------------------

def check_sustainability_breaches(data: dict) -> pd.DataFrame:
    """
    Check for holdings flagged sustainability_excluded=Y in sustainable mandates.
    """
    holdings = data["holdings"]
    instruments = data["instruments"]
    portfolios = data["portfolios"]

    current = holdings[holdings["snapshot_date"] == TODAY]
    sus_mandates = ["SUSBAL"]

    results = []
    for pid in current["portfolio_id"].unique():
        pf = portfolios[portfolios["portfolio_id"] == pid]
        if pf.empty or pf.iloc[0]["mandate_code"] not in sus_mandates:
            continue

        pf_holdings = current[current["portfolio_id"] == pid]
        for _, h in pf_holdings.iterrows():
            inst = instruments[instruments["instrument_id"] == h["instrument_id"]]
            if inst.empty:
                continue
            if inst.iloc[0].get("sustainability_excluded", "N") == "Y":
                results.append({
                    "portfolio_id": pid,
                    "client_id": h["client_id"],
                    "instrument_name": h["instrument_name"],
                    "instrument_id": h["instrument_id"],
                    "weight_pct": h["weight_pct"],
                    "asset_class": h["asset_class"],
                    "sector": h.get("sector", ""),
                })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Client Detail Helpers
# ---------------------------------------------------------------------------

def get_client_detail(data: dict, client_id: str) -> dict:
    """Get a comprehensive detail view for a single client."""
    clients = data["clients"]
    client = clients[clients["client_id"] == client_id].iloc[0]

    # RM notes for this client
    notes = [n for n in data["rm_notes"] if n["client_id"] == client_id]

    # Portfolios
    portfolios = data["portfolios"][data["portfolios"]["client_id"] == client_id]

    # Current holdings
    current_holdings = data["holdings"][
        (data["holdings"]["client_id"] == client_id) &
        (data["holdings"]["snapshot_date"] == TODAY)
    ]

    # Credit facilities
    facilities = data["credit_facilities"][data["credit_facilities"]["client_id"] == client_id]

    # Cash needs
    cash_needs = data["planned_cash_needs"][data["planned_cash_needs"]["client_id"] == client_id]

    # Commitments
    commitments = data["commitments"][data["commitments"]["client_id"] == client_id]

    return {
        "client": client.to_dict(),
        "rm_notes": notes,
        "portfolios": portfolios,
        "current_holdings": current_holdings,
        "facilities": facilities,
        "cash_needs": cash_needs,
        "commitments": commitments,
    }


def get_asset_allocation(data: dict, client_id: str, snapshot_date: str = TODAY) -> pd.DataFrame:
    """Get asset allocation breakdown for a client at a given snapshot."""
    holdings = data["holdings"]
    client_h = holdings[
        (holdings["client_id"] == client_id) &
        (holdings["snapshot_date"] == snapshot_date)
    ]

    if client_h.empty:
        return pd.DataFrame()

    alloc = client_h.groupby("asset_class").agg(
        market_value_usd=("market_value_usd", "sum"),
        positions=("instrument_id", "count"),
    ).reset_index()

    total = alloc["market_value_usd"].sum()
    alloc["weight_pct"] = (alloc["market_value_usd"] / total * 100).round(2)

    return alloc.sort_values("weight_pct", ascending=False)


def get_holdings_detail(data: dict, client_id: str, snapshot_date: str = TODAY) -> pd.DataFrame:
    """Get detailed holdings for a client at a snapshot date."""
    holdings = data["holdings"]
    return holdings[
        (holdings["client_id"] == client_id) &
        (holdings["snapshot_date"] == snapshot_date)
    ].sort_values("market_value_usd", ascending=False)
