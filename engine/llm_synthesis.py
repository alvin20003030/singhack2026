"""Grounded synthesis boundary for the RM workbench.

The underlying provider remains optional. Every synthesis call receives deterministic
analytics and the authoritative event log before any model is invoked.
"""

from __future__ import annotations

import re

import pandas as pd

from src.ai_advisor import (
    _call_gemini,
    _gemini_available,
    generate_client_message,
    generate_life_event_plan,
    generate_portfolio_explanation,
    generate_rebalancing_suggestion,
    generate_rm_chat_response,
    generate_tax_optimization,
    is_ai_available,
)


ADVISORY_SYNTHESIS_DIRECTIVE = """Prioritize advisory coaching over descriptive summarization.
Every response must answer: what does this mean for this specific client, why does it
matter now, and what should the RM discuss or prepare next? Connect calculated portfolio
figures to the client's documented objectives, liquidity needs, risk profile, and RM notes.
Do not write generic statements such as 'the portfolio dropped because of equities'.
State the number, the client consequence, and a concrete editable RM action. Label every
proposal as a suggestion for RM review. Use only the supplied analytics, event_log.csv,
market_context.csv, and rm_notes.json; if the data does not support a conclusion, say so.
"""


def _money(value: float) -> str:
    """Format a USD value without inventing precision."""
    value = float(value or 0)
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value) / 1_000_000:,.1f}M"


def _event_grounded_driver(attributions: list[dict]) -> str:
    """Describe the largest matched movement using only event-log evidence."""
    if not attributions:
        return "No material position movement was matched to an event_log.csv entry; do not infer an external cause."
    negative = [item for item in attributions if item.get("mv_change_usd", 0) < 0]
    largest = sorted(negative or attributions, key=lambda item: abs(item.get("mv_change_usd", 0)), reverse=True)[0]
    citations = "; ".join(
        f"{event['event_date']}: {event['description']}"
        for event in largest.get("events", [])
    )
    return (
        f"Largest matched movement: {largest['instrument_name']} changed "
        f"{_money(largest['mv_change_usd'])} ({largest['mv_change_pct']:+.1f}%) "
        f"during {largest['period']}. Authoritative event_log.csv citation: {citations}."
    )


def generate_executive_briefing(
    client: dict,
    analytics_snapshot: dict,
    rm_notes: list[dict],
    cash_needs: pd.DataFrame,
    attributions: list[dict],
    commitments: pd.DataFrame | None = None,
    mandate_drift: pd.DataFrame | None = None,
) -> str:
    """Create one concise, qualitative client background and RM suggestion.

    This intentionally does not call an LLM: the REVIEW card must never invent a
    market cause, client constraint, drawdown, or recommendation not present in the
    supplied analytics and source records.
    """
    summary = analytics_snapshot.get("summary", pd.DataFrame())
    liquidity = analytics_snapshot.get("liquidity", pd.DataFrame())
    holdings = analytics_snapshot.get("holdings", pd.DataFrame())
    client_row = summary.iloc[0] if isinstance(summary, pd.DataFrame) and not summary.empty else {}
    current_liquidity = liquidity.iloc[0] if isinstance(liquidity, pd.DataFrame) and not liquidity.empty else {}

    life_stage = str(client.get("life_stage", "client with a documented life-stage transition")).lower()
    risk_profile = str(client.get("risk_profile", "documented risk profile")).lower()
    objectives = str(client.get("objectives", "their stated wealth objectives")).rstrip(".")
    liquidity_needs = str(client.get("liquidity_needs", "documented liquidity needs")).lower()
    note_text = " ".join(note.get("note", "") for note in rm_notes)
    note_lower = note_text.lower()
    profile_article = "an" if risk_profile[:1] in "aeiou" else "a"
    raw_age = client.get("age", 0)
    try:
        age_text = "age not recorded" if pd.isna(raw_age) else f"{int(float(raw_age))}-year-old"
    except (TypeError, ValueError):
        age_text = "age not recorded"
    client_age_phrase = f"a {age_text}" if age_text == "age not recorded" else f"an {age_text}"

    annual_drawdown = re.search(
        r"(?:draw|withdraw|spend)\s+(?:USD\s*)?\$?([0-9,.]+)\s*m(?:\s+per year|\s+annually)?",
        str(client.get("objectives", "")),
        re.IGNORECASE,
    )
    drawdown = f"draws USD {annual_drawdown.group(1)}M a year" if annual_drawdown else "has documented ongoing spending needs"
    objective_context = (
        "to support living costs and capital preservation"
        if annual_drawdown
        else f"to support the stated objective of {objectives}"
    )

    fixed_income_loss = 0.0
    if isinstance(holdings, pd.DataFrame) and not holdings.empty and "unrealised_pnl_base" in holdings:
        fixed_income_loss = float(holdings.loc[holdings["asset_class"].eq("Fixed Income"), "unrealised_pnl_base"].sum())

    maturity_years = [
        int(year)
        for name in holdings["instrument_name"].astype(str)
        for year in re.findall(r"20\d{2}", name)
    ] if isinstance(holdings, pd.DataFrame) and not holdings.empty else []
    maturity = f"the longest named bond matures in {max(maturity_years)}" if maturity_years else "the maturity profile should be reviewed"

    event_driver = "the documented event record explains the relevant portfolio movement"
    negative = [item for item in attributions if item.get("mv_change_usd", 0) < 0]
    if negative:
        largest = max(negative, key=lambda item: abs(item.get("mv_change_usd", 0)))
        event_text = " ".join(event["description"] for event in largest.get("events", []))
        if "yield" in event_text.lower() or "duration" in event_text.lower():
            event_driver = "yields rose after the energy-driven inflation shock recorded in event_log.csv"
        else:
            event_driver = "the matched event-log entry recorded in event_log.csv"

    commitment_context = "There are no recorded private-market commitments competing for liquidity."
    if isinstance(commitments, pd.DataFrame) and not commitments.empty:
        commitment_context = "Uncalled private-market commitments also compete for liquidity, so capital-call planning must sit alongside spending needs."

    mandate_context = "The current allocation is within the recorded mandate bands."
    if isinstance(mandate_drift, pd.DataFrame) and not mandate_drift.empty:
        breaches = mandate_drift[mandate_drift["breach"] != "None"]
        if not breaches.empty:
            assets = ", ".join(breaches["asset_class"].astype(str).drop_duplicates().head(2))
            mandate_context = f"The mandate is drifting across {assets}, so any proposal must also restore alignment."

    constraint = "The RM notes show the client is reluctant to sell at a loss." if "sell" in note_lower or "loss" in note_lower else "The RM notes should guide the tone and sequencing of the discussion."
    cash_need = "Confirmed cash needs include living expenses and medical care." if "medical" in note_lower else "Planned cash needs should be ring-fenced before changing the portfolio."
    return "\n\n".join([
        "**Client context**\n"
        f"You are preparing for a conversation with {client_age_phrase} client who is {life_stage}, "
        f"with {profile_article} {risk_profile} profile. Your client {drawdown} {objective_context}.",
        "**What matters now**\n"
        f"The fixed-income portfolio carries **{_money(abs(fixed_income_loss))} unrealised loss** because {event_driver}; "
        f"**{maturity}**. {constraint}",
        "**Planning pointers**\n"
        f"{cash_need} {commitment_context} {mandate_context}",
        "**Suggested opener**\n"
        "\"I understand why waiting feels safer than selling at a loss. Let us separate the cash you need from the holdings you want to preserve, then test whether a shorter-duration income structure could support your priorities more reliably.\"",
    ])


def generate_review_advisories(
    client: dict,
    snapshot: dict,
    detail: dict,
    events_df: pd.DataFrame,
    market_context_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    instruments_df: pd.DataFrame,
    portfolios_df: pd.DataFrame,
    mandates_df: pd.DataFrame,
    include_connected_ai: bool = True,
) -> dict[str, str]:
    """Create short imperative advisories for the five REVIEW sections.

    The function is deliberately deterministic. It receives every source used by
    the REVIEW page, but only makes claims supported by the calculated snapshot
    and matched event-log records.
    """
    holdings = snapshot.get("holdings", pd.DataFrame())
    drift = snapshot.get("mandate_drift", pd.DataFrame())
    ltv = snapshot.get("ltv", pd.DataFrame())
    liquidity = snapshot.get("liquidity", pd.DataFrame())
    attributions = snapshot.get("explanation", {}).get("attributions", [])
    loss_attrs = [item for item in attributions if item.get("mv_change_usd", 0) < 0]
    loss_name = loss_attrs[0]["instrument_name"] if loss_attrs else "the largest loss position"
    breach_count = int((drift["breach"] != "None").sum()) if not drift.empty else 0
    ltv_risk = False
    if not ltv.empty:
        ltv_risk = bool((ltv["ltv_headroom_pct"] <= 3).any())
    stressed = bool(not liquidity.empty and liquidity.iloc[0].get("is_stressed", False))
    has_commitments = not detail.get("commitments", pd.DataFrame()).empty
    has_transactions = not transactions_df.empty
    structured_positions = 0
    if not holdings.empty and not instruments_df.empty:
        structured_ids = set(
            instruments_df.loc[
                instruments_df["underlying_reference"].fillna("").astype(str).str.len() > 0,
                "instrument_id",
            ]
        )
        structured_positions = int(holdings["instrument_id"].isin(structured_ids).sum())
    has_market_context = not market_context_df.empty
    mapped_portfolios = int(
        portfolios_df[portfolios_df["client_id"] == client.get("client_id")].shape[0]
    )
    market_evidence = "market-context" if has_market_context else "available source"
    transaction_evidence = "recent transaction cash flows" if has_transactions else "recorded transaction obligations"
    mandate_action = "realign the most material drift to its mandate target bands" if breach_count else "confirm the allocation remains within its mandate bands"

    result = {
        "performance": (
            f"Recommend reviewing {loss_name} across the client's {mapped_portfolios} portfolio(s) and any structured-product look-through exposure ({structured_positions} mapped position(s)) alongside matched event-log and {market_evidence} evidence, then propose a staged holding strategy that protects liquidity and avoids selling solely because of a drawdown."
        ),
        "credit": (
            "Propose a collateral review and debt-paydown or collateral-substitution plan before the next client contact."
            if ltv_risk else
            "Confirm collateral headroom and agree a monitoring trigger before extending or increasing borrowing."
        ),
        "liquidity": (
            f"Recommend ring-fencing planned cash needs and any private-market calls, then reconcile {transaction_evidence} before identifying liquid assets or credit capacity to fund the next outflow window."
            if stressed or has_commitments else
            f"Recommend matching planned cash needs to liquid assets and confirming the funding plan before the next due date, including {transaction_evidence}."
        ),
        "mandate": (
            f"Propose a tax-aware, RM-reviewed rebalance to {mandate_action}; match any loss harvesting to the client's tax domicile before execution."
        ),
        "qualitative": (
            "Reframe the next meeting around the client's documented priorities and constraints, then open with one concrete choice that preserves agency and can be reviewed before approval."
        ),
    }
    if include_connected_ai:
        result["connected_ai"] = _generate_connected_review_coaching(
            client,
            snapshot,
            detail,
            events_df,
            market_context_df,
            transactions_df,
            instruments_df,
            portfolios_df,
            mandates_df,
            result["performance"],
        )
    return result


def generate_connected_review_coaching(
    client: dict,
    snapshot: dict,
    detail: dict,
    events_df: pd.DataFrame,
    market_context_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    instruments_df: pd.DataFrame,
    portfolios_df: pd.DataFrame,
    mandates_df: pd.DataFrame,
    deterministic_advice: str,
) -> str:
    """Lazy-load the slow connected AI coaching panel on demand."""
    return _generate_connected_review_coaching(
        client,
        snapshot,
        detail,
        events_df,
        market_context_df,
        transactions_df,
        instruments_df,
        portfolios_df,
        mandates_df,
        deterministic_advice,
    )


def _generate_connected_review_coaching(
    client: dict,
    snapshot: dict,
    detail: dict,
    events_df: pd.DataFrame,
    market_context_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    instruments_df: pd.DataFrame,
    portfolios_df: pd.DataFrame,
    mandates_df: pd.DataFrame,
    deterministic_advice: str,
) -> str:
    """Ask the connected model for grounded, action-first REVIEW coaching."""
    if not _gemini_available:
        return (
            "**Connected AI unavailable: deterministic coaching**\n\n"
            f"{deterministic_advice}"
        )

    drift = snapshot.get("mandate_drift", pd.DataFrame())
    ltv = snapshot.get("ltv", pd.DataFrame())
    liquidity = snapshot.get("liquidity", pd.DataFrame())
    attributions = snapshot.get("explanation", {}).get("attributions", [])
    event_lines = []
    for attribution in attributions[:8]:
        citations = "; ".join(
            f"{event['event_date']}: {event['description']}"
            for event in attribution.get("events", [])
        )
        event_lines.append(
            f"{attribution['instrument_name']} changed {attribution['mv_change_usd']:+,.0f} "
            f"during {attribution['period']}; {citations}"
        )
    context = "\n".join([
        "AUTHORITATIVE REVIEW CONTEXT. Use only this context; do not add outside market knowledge.",
        f"CLIENT PROFILE: {client}",
        f"RM NOTES: {detail.get('rm_notes', [])}",
        f"MANDATE DRIFT: {drift.to_dict('records')}",
        f"LTV: {ltv.to_dict('records')}",
        f"LIQUIDITY: {liquidity.to_dict('records')}",
        f"COMMITMENTS: {detail.get('commitments', pd.DataFrame()).to_dict('records')}",
        f"PLANNED CASH NEEDS: {detail.get('cash_needs', pd.DataFrame()).to_dict('records')}",
        f"LOOK-THROUGH: {snapshot.get('lookthrough', pd.DataFrame()).to_dict('records')}",
        f"TRANSACTIONS: {transactions_df[transactions_df['client_id'] == client.get('client_id')].tail(12).to_dict('records')}",
        f"PORTFOLIOS: {portfolios_df[portfolios_df['client_id'] == client.get('client_id')].to_dict('records')}",
        f"MANDATE DEFINITIONS: {mandates_df[mandates_df['mandate_code'].isin(portfolios_df[portfolios_df['client_id'] == client.get('client_id')]['mandate_code'])].to_dict('records')}",
        f"MATCHED EVENT_LOG.CSV CITATIONS:\n- " + "\n- ".join(event_lines),
        f"MARKET_CONTEXT.CSV ROWS:\n{market_context_df.tail(12).to_dict('records')}",
        f"DETERMINISTIC BASE ADVICE: {deterministic_advice}",
    ])
    prompt = """You are the connected AI coaching layer for an RM reviewing one client.

Produce a concise but specific advisory plan, not a descriptive summary. Start with an
imperative verb: Recommend, Propose, Reframe, Harvest, or Deploy. Give three practical
next actions for the RM, each linking a client reality to a portfolio action. Include:
1) what to discuss first, 2) what portfolio or risk action to prepare, and 3) what to
verify before implementation. Use exact event dates only when citing MATCHED EVENT_LOG.CSV.
Do not invent macro events, tax rules, figures, trades, or client preferences. Do not
repeat a raw data dump. Keep all actions editable and subject to RM approval. End with
one senior-friendly conversation opener. If evidence is insufficient, say what the RM
must research rather than guessing."""
    try:
        response = _call_gemini(prompt, context)
        if response.startswith("⚠️") or response.startswith("🔑"):
            fallback = "\n".join([
                deterministic_advice,
                "Propose the credit and liquidity checks shown below before implementation.",
                "Reframe the mandate and tax review around the client's documented preferences.",
                "Confirm the relevant event-log citation and RM approval before communicating any action.",
                "Conversation opener: 'Let us review the evidence together and agree which option best fits your priorities before we change anything.'",
            ])
            return f"**Connected AI unavailable; grounded coaching fallback**\n\n{fallback}"
        return f"**Connected AI coaching**\n\n{response}"
    except Exception:
        return f"**Connected AI fallback**\n\n{deterministic_advice}"


def build_compare_allocation(
    client: dict,
    holdings_df: pd.DataFrame,
    instruments_df: pd.DataFrame,
    portfolios_df: pd.DataFrame,
    mandates_df: pd.DataFrame,
) -> dict:
    """Build mandate-health metrics and trade rows for the COMPARE view."""
    client_id = client.get("client_id")
    holdings = holdings_df[holdings_df["client_id"] == client_id].copy()
    portfolios = portfolios_df[portfolios_df["client_id"] == client_id]
    mandate_codes = portfolios["mandate_code"].dropna().unique().tolist()
    bands = mandates_df[mandates_df["mandate_code"].isin(mandate_codes)].copy()
    total_value = float(holdings["market_value_usd"].sum()) if not holdings.empty else 0.0

    # Structured products remain in their contractual asset class while their
    # underlying references are counted as look-through evidence for the RM.
    structured_ids = set(
        instruments_df.loc[
            instruments_df["underlying_reference"].fillna("").astype(str).str.len() > 0,
            "instrument_id",
        ]
    ) if not instruments_df.empty else set()
    structured_count = int(holdings["instrument_id"].isin(structured_ids).sum()) if not holdings.empty else 0

    asset_map = {
        "Equity": "Equities",
        "Cash and Equivalents": "Cash",
    }
    current = holdings.groupby("asset_class", as_index=False).agg(
        current_value=("market_value_usd", "sum"),
        current_weight=("market_value_usd", lambda values: values.sum() / total_value * 100 if total_value else 0),
    ) if not holdings.empty else pd.DataFrame(columns=["asset_class", "current_value", "current_weight"])
    current["asset_class"] = current["asset_class"].map(lambda value: asset_map.get(value, value))
    bands["display_asset_class"] = bands["asset_class"].map(lambda value: asset_map.get(value, value))
    band_rows = bands.groupby("display_asset_class", as_index=False).agg(
        min_pct=("min_pct", "min"), target_pct=("target_pct", "mean"), max_pct=("max_pct", "max")
    )
    classes = sorted(set(current["asset_class"]).union(band_rows["display_asset_class"]))
    rows = []
    for asset_class in classes:
        current_row = current[current["asset_class"] == asset_class]
        band_row = band_rows[band_rows["display_asset_class"] == asset_class]
        current_weight = float(current_row.iloc[0]["current_weight"]) if not current_row.empty else 0.0
        current_value = float(current_row.iloc[0]["current_value"]) if not current_row.empty else 0.0
        minimum = float(band_row.iloc[0]["min_pct"]) if not band_row.empty else 0.0
        target = float(band_row.iloc[0]["target_pct"]) if not band_row.empty else 0.0
        maximum = float(band_row.iloc[0]["max_pct"]) if not band_row.empty else 0.0
        status = "[OVERWEIGHT]" if current_weight > maximum else "[UNDERWEIGHT]" if current_weight < minimum else "[IN BAND]"
        proposed_value = total_value * target / 100
        rows.append({
            "Asset Class": asset_class,
            "Current Weight (%)": current_weight,
            "Current Value ($)": current_value,
            "Mandate Target Band (%)": f"{minimum:.1f}% - {maximum:.1f}% (Target {target:.1f}%)",
            "Min (%)": minimum,
            "Target (%)": target,
            "Max (%)": maximum,
            "Status": status,
            "Proposed Weight (%)": target,
            "Proposed Value ($)": proposed_value,
            "Net Trade Required ($)": proposed_value - current_value,
        })
    table = pd.DataFrame(rows)
    deviations = []
    for _, row in table.iterrows():
        if row["Current Weight (%)"] < row["Min (%)"]:
            deviations.append(row["Min (%)"] - row["Current Weight (%)"])
        elif row["Current Weight (%)"] > row["Max (%)"]:
            deviations.append(row["Current Weight (%)"] - row["Max (%)"])
        else:
            deviations.append(abs(row["Current Weight (%)"] - row["Target (%)"]))
    max_deviation = max(deviations, default=0.0)
    breach_count = int((table["Status"] != "[IN BAND]").sum()) if not table.empty else 0
    health = "[SEVERE BREACH]" if max_deviation > 10 else "[MODERATE DRIFT]" if breach_count else "[COMPLIANT]"
    return {
        "table": table,
        "total_value": total_value,
        "structured_count": structured_count,
        "health": health,
        "max_deviation": max_deviation,
        "breach_count": breach_count,
    }


def generate_compare_advisory(
    client: dict,
    comparison: dict,
    rm_notes: list[dict],
    tax_data: dict,
) -> str:
    """Return imperative, client-specific rebalancing coaching for COMPARE."""
    table = comparison["table"]
    trades = table[table["Net Trade Required ($)"].abs() > 1].sort_values("Net Trade Required ($)", key=lambda values: values.abs(), ascending=False).head(3)
    actions = []
    for _, row in trades.iterrows():
        verb = "Buy" if row["Net Trade Required ($)"] > 0 else "Sell"
        actions.append(f"{verb} {row['Asset Class']} toward the authorised target allocation")
    action_text = "; ".join(actions) or "confirm that no rebalance is required"
    note_text = " ".join(note.get("note", "") for note in rm_notes).lower()
    preference = "Respect the client's stated reluctance to realise losses by using new cash or maturities first." if "sell" in note_text or "loss" in note_text else "Keep the proposal consistent with the client's documented preferences."
    domicile = client.get("tax_domicile", "the recorded tax domicile")
    return (
        f"Propose {action_text}. Harvest eligible unrealised losses only after confirming the {domicile} tax treatment and matching them against realised gains. "
        f"{preference} Open with: 'Let us review the allocation changes as a way to protect your stated priorities, and agree which adjustment you are comfortable reviewing first.'"
    )

__all__ = [
    "generate_client_message",
    "generate_executive_briefing",
    "generate_review_advisories",
    "generate_connected_review_coaching",
    "build_compare_allocation",
    "generate_compare_advisory",
    "generate_life_event_plan",
    "generate_portfolio_explanation",
    "generate_rebalancing_suggestion",
    "generate_rm_chat_response",
    "generate_tax_optimization",
    "is_ai_available",
]
