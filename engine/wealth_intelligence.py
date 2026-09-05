"""AI Wealth Intelligence Layer.

This module is the only backend surface the Streamlit workbench needs. It owns
loading, deterministic calculations, grounded attribution, scenario modelling,
and audit persistence. The UI consumes its outputs and does not calculate risk.
"""

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd

from src.analytics import (
    SNAPSHOT_DATES,
    TODAY,
    attribute_portfolio_changes,
    compute_concentration_risk,
    compute_unrealised_pnl,
    check_sustainability_breaches,
    compute_ltv_analysis,
    compute_mandate_drift,
    compute_portfolio_summary,
    compute_liquidity_coverage,
    get_asset_allocation,
    get_client_detail,
    get_client_portfolio_timeseries,
    get_holdings_detail,
    get_lookthrough_exposure,
    get_tax_optimization_opportunities,
    load_all_data,
    simulate_continuous_monitoring,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
AUDIT_LOG_PATH = os.path.join(DATA_DIR, "audit_log.json")


def load_intelligence_data() -> dict:
    """Load the complete synthetic book for the intelligence layer."""
    return load_all_data()


def analyse(data: dict) -> dict:
    """Run deterministic monitoring across the complete book and five snapshots."""
    mandate_drift = compute_mandate_drift(data)
    alerts = simulate_continuous_monitoring(
        data["clients"], data["holdings"], data["credit_facilities"],
        data["commitments"], data["planned_cash_needs"], data["mandates"],
    )
    # compute_mandate_drift is the authoritative band calculation. Add its
    # breaches to the stream because the monitor's public contract has no
    # portfolio-to-mandate mapping argument.
    for _, breach in mandate_drift[mandate_drift["breach"] != "None"].iterrows():
        alerts.append({
            "snapshot_date": TODAY,
            "client_id": breach["client_id"],
            "client_name": data["clients"].set_index("client_id").loc[breach["client_id"], "client_name"],
            "alert_type": "[MANDATE DRIFT]",
            "severity": "MEDIUM",
            "message": f"{breach['portfolio_name']} {breach['asset_class']} is {breach['actual_weight_pct']:.1f}% versus {breach['target_pct']:.1f}% target ({breach['breach_amount_pct']:.1f}% outside band).",
        })
    return {
        "portfolio_summary": compute_portfolio_summary(data),
        "ltv": compute_ltv_analysis(data),
        "liquidity": compute_liquidity_coverage(data),
        "mandate_drift": mandate_drift,
        "concentration": compute_concentration_risk(data),
        "monitoring_alerts": alerts,
    }


def _client_notes(data: dict, client_id: str) -> list[dict]:
    return [note for note in data["rm_notes"] if note["client_id"] == client_id]


def _client_event_attributions(data: dict, client_id: str) -> list[dict]:
    """Return only attributions backed by matched event_log rows."""
    return attribute_portfolio_changes(data, client_id)


def explain(data: dict, client_id: str) -> dict:
    """Build snapshot movements and event-grounded explanations."""
    summary = compute_portfolio_summary(data)
    client_summary = summary[summary["client_id"] == client_id]
    attributions = _client_event_attributions(data, client_id)
    return {
        "summary": client_summary.to_dict("records"),
        "timeseries": get_client_portfolio_timeseries(data, client_id),
        "attributions": attributions,
        "event_log": data["event_log"],
        "rm_notes": _client_notes(data, client_id),
    }


def recommend(data: dict, client_id: str) -> dict:
    """Build mandate, tax, life-context, and editable recommendation inputs."""
    drift = compute_mandate_drift(data)
    ltv = compute_ltv_analysis(data)
    liquidity = compute_liquidity_coverage(data)
    return {
        "mandate_drift": drift[drift["client_id"] == client_id],
        "tax_optimization": get_tax_optimization_opportunities(data, client_id),
        "ltv": ltv[ltv["client_id"] == client_id],
        "liquidity": liquidity[liquidity["client_id"] == client_id],
        "rm_notes": _client_notes(data, client_id),
    }


def stress_test(data: dict, client_id: str) -> pd.DataFrame:
    """Apply transparent illustrative sensitivities to current client holdings."""
    holdings = get_holdings_detail(data, client_id)
    if holdings.empty:
        return pd.DataFrame(columns=["scenario", "projected_impact_usd", "method"])
    exposures = holdings.groupby("asset_class")["market_value_usd"].sum()
    scenarios = [
        ("Middle East conflict de-escalates", 0.02),
        ("Prolonged oil shock", -0.04),
    ]
    rows = []
    for scenario, shock in scenarios:
        impact = 0.0
        for asset_class, value in exposures.items():
            sensitivity = {
                "Commodities": 1.5,
                "Energy": 1.5,
                "Fixed Income": -1.0,
                "Equity": 0.4,
            }.get(asset_class, 0.2)
            impact += value * shock * sensitivity
        rows.append({
            "scenario": scenario,
            "projected_impact_usd": round(impact, 2),
            "method": "Illustrative asset-class sensitivity; not a forecast.",
        })
    return pd.DataFrame(rows)


def client_snapshot(data: dict, client_id: str, analysis: dict | None = None) -> dict:
    """Return the complete machine-readable context for one RM review."""
    analysis = analysis or analyse(data)
    detail = get_client_detail(data, client_id)
    client_alerts = [a for a in analysis["monitoring_alerts"] if a["client_id"] == client_id]
    return {
        "client": detail["client"],
        "portfolios": detail["portfolios"],
        "holdings": get_holdings_detail(data, client_id),
        "lookthrough": get_lookthrough_exposure(data, client_id),
        "summary": analysis["portfolio_summary"][analysis["portfolio_summary"]["client_id"] == client_id],
        "ltv": analysis["ltv"][analysis["ltv"]["client_id"] == client_id],
        "liquidity": analysis["liquidity"][analysis["liquidity"]["client_id"] == client_id],
        "mandate_drift": analysis["mandate_drift"][analysis["mandate_drift"]["client_id"] == client_id],
        "concentration": analysis["concentration"][analysis["concentration"]["client_id"] == client_id],
        "alerts": client_alerts,
        "tax": get_tax_optimization_opportunities(data, client_id),
        "explanation": explain(data, client_id),
        "recommendations": recommend(data, client_id),
        "stress_test": stress_test(data, client_id),
        "rm_notes": detail["rm_notes"],
    }


def prioritise(data: dict, analysis: dict | None = None) -> pd.DataFrame:
    """Rank the book using the required 35/25/20/20 risk weights."""
    analysis = analysis or analyse(data)
    summary = analysis["portfolio_summary"]
    ltv = analysis["ltv"]
    liquidity = analysis["liquidity"]
    drift = analysis["mandate_drift"]
    alerts = analysis["monitoring_alerts"]
    rows = []
    for _, client in data["clients"].iterrows():
        cid = client["client_id"]
        client_ltv = ltv[ltv["client_id"] == cid]
        ltv_component = 0.0 if client_ltv.empty else min(100.0, max(0.0, 100 - client_ltv["ltv_headroom_pct"].min() * 5))
        client_liq = liquidity[liquidity["client_id"] == cid]
        liquidity_component = 0.0
        if not client_liq.empty:
            coverage = float(client_liq.iloc[0]["coverage_ratio"])
            liquidity_component = 100.0 if coverage < 1 else max(0.0, min(100.0, 100 - coverage * 20))
        client_drift = drift[(drift["client_id"] == cid) & (drift["breach"] != "None")]
        drift_component = min(100.0, len(client_drift) * 25 + (client_drift["breach_amount_pct"].max() * 2 if not client_drift.empty else 0))
        client_summary = summary[summary["client_id"] == cid]
        event_component = 0.0
        if not client_summary.empty:
            event_component = min(100.0, max(0.0, -float(client_summary.iloc[0]["ytd_change_pct"]) * 5))
        score = round(ltv_component * .35 + liquidity_component * .25 + drift_component * .20 + event_component * .20)
        client_alerts = {alert["alert_type"] for alert in alerts if alert["client_id"] == cid}
        badges = []
        if "[URGENT: LTV BREACH]" in client_alerts:
            badges.append("[MARGIN CALL WARNING]")
        if "[LIQUIDITY WARNING]" in client_alerts:
            badges.append("[CASH NEED <30 DAYS]")
        if "[MANDATE DRIFT]" in client_alerts:
            badges.append("[MANDATE DRIFT]")
        rows.append({
            "client_id": cid,
            "client_name": client["client_name"],
            "priority_score": max(1, min(100, score)),
            "badges": " ".join(badges) or "[MONITOR]",
            "action_tag": "URGENT" if score >= 70 else "REVIEW" if score >= 45 else "MONITOR" if score >= 20 else "ON TRACK",
            "risk_flags": " ".join(badges) or "No critical flags",
            "ltv_component": round(ltv_component, 1),
            "liquidity_component": round(liquidity_component, 1),
            "drift_component": round(drift_component, 1),
            "event_component": round(event_component, 1),
            "risk_profile": client["risk_profile"],
            "life_stage": client["life_stage"],
            "total_aum_usd": client["total_aum_usd"],
        })
    return pd.DataFrame(rows).sort_values(["priority_score", "client_name"], ascending=[False, True]).reset_index(drop=True)


def get_score_breakdown(data: dict, client_id: str) -> dict:
    """Compatibility view of the prioritisation output for an individual client."""
    ranked = prioritise(data)
    row = ranked[ranked["client_id"] == client_id]
    return row.iloc[0].to_dict() if not row.empty else {}


def get_client_analytics_context(data: dict, client_id: str) -> dict:
    """Compatibility context for grounded synthesis, sourced from this layer."""
    snapshot = client_snapshot(data, client_id)
    return {
        "portfolio_summary": snapshot["summary"].to_dict("records"),
        "ltv": snapshot["ltv"].to_dict("records"),
        "liquidity": snapshot["liquidity"].to_dict("records"),
        "mandate_drift": snapshot["mandate_drift"].to_dict("records"),
        "tax_optimization": snapshot["tax"],
        "lookthrough_exposure": snapshot["lookthrough"].to_dict("records"),
        "monitoring_alerts": snapshot["alerts"],
    }


def compare(data: dict, client_id: str, analysis: dict | None = None) -> pd.DataFrame:
    """Produce current, target, and proposed allocation states."""
    analysis = analysis or analyse(data)
    current = get_asset_allocation(data, client_id).rename(columns={"weight_pct": "current_pct"})
    portfolios = data["portfolios"][data["portfolios"]["client_id"] == client_id]
    mandate_codes = portfolios["mandate_code"].dropna().unique().tolist()
    bands = data["mandates"][data["mandates"]["mandate_code"].isin(mandate_codes)]
    target = bands.groupby("asset_class", as_index=False)["target_pct"].mean()
    target = target.rename(columns={"target_pct": "target_pct"})
    result = current[["asset_class", "current_pct"]].merge(target, on="asset_class", how="outer").fillna(0)
    result["proposed_pct"] = result["target_pct"]
    result["delta_to_target_pct"] = result["target_pct"] - result["current_pct"]
    return result.sort_values("current_pct", ascending=False)


def conflict_flags(data: dict, client_id: str, snapshot: dict | None = None) -> list[str]:
    """Flag simple, auditable disagreements between notes and calculated numbers."""
    snapshot = snapshot or client_snapshot(data, client_id)
    flags = []
    notes = " ".join(note["note"].lower() for note in snapshot["rm_notes"])
    summary = snapshot["summary"]
    if "positive" in notes and not summary.empty and float(summary.iloc[0]["ytd_change_pct"]) < 0:
        flags.append("RM note describes a positive outlook while YTD portfolio change is negative; confirm which exposure the note refers to.")
    if ("low liquidity" in notes or "liquidity is strong" in notes) and not snapshot["liquidity"].empty:
        if bool(snapshot["liquidity"].iloc[0]["is_stressed"]):
            flags.append("RM note indicates comfortable liquidity but calculated coverage is stressed.")
    if "no leverage" in notes and not snapshot["ltv"].empty and snapshot["ltv"]["current_ltv"].max() > 0:
        flags.append("RM note says no leverage while a credit facility has a non-zero current LTV.")
    return flags


def audit_action(client_id: str, original_draft: str, rm_draft: str, action_type: str) -> dict:
    """Append an immutable local audit record for RM approval."""
    records = []
    if os.path.exists(AUDIT_LOG_PATH):
        try:
            with open(AUDIT_LOG_PATH, encoding="utf-8") as file:
                records = json.load(file)
        except (json.JSONDecodeError, OSError):
            records = []
    record = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "client_id": client_id,
        "action_type": action_type,
        "original_ai_draft": original_draft,
        "rm_modified_draft": rm_draft,
        "status": "APPROVED_BY_RM",
    }
    records.append(record)
    with open(AUDIT_LOG_PATH, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=True)
    return record


__all__ = [
    "AUDIT_LOG_PATH", "SNAPSHOT_DATES", "TODAY", "analyse", "audit_action",
    "client_snapshot", "compare", "conflict_flags", "explain", "load_intelligence_data",
    "prioritise", "recommend", "stress_test", "get_score_breakdown",
    "get_client_analytics_context", "get_client_detail", "get_asset_allocation",
    "get_holdings_detail", "get_client_portfolio_timeseries", "get_tax_optimization_opportunities",
    "get_lookthrough_exposure", "compute_portfolio_summary", "compute_ltv_analysis",
    "compute_mandate_drift", "compute_liquidity_coverage", "compute_concentration_risk",
    "attribute_portfolio_changes", "simulate_continuous_monitoring",
]
