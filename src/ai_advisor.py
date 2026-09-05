"""
Gemini AI Advisory Engine for the RM Intelligence Workbench.

Generates grounded portfolio explanations, rebalancing suggestions,
tax-aware optimizations, life-event plans, and client messages.
All prompts enforce that 2026 events MUST cite event_log.csv entries only.
"""

import json
import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Try to import and configure Gemini
_gemini_available = False
_client = None

try:
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key and api_key != "your-api-key-here":
        _client = genai.Client(api_key=api_key)
        _gemini_available = True
except ImportError:
    pass

MODEL = "gemini-3.6-flash"

# System prompt that enforces grounding to event_log.csv
SYSTEM_PROMPT = """You are an AI advisory assistant for a private bank Relationship Manager (RM) named Priscilla Ong.
You help her understand portfolios, explain market impacts, and draft recommendations.

CRITICAL RULES:
1. For any 2026 market or geopolitical events, you MUST ONLY reference events from the provided event_log data. Do NOT use your own knowledge of 2026 events.
2. Every market explanation must cite specific events with their dates from the event_log.
3. All recommendations must respect the client's mandate, risk profile, and stated objectives.
4. Present analysis as suggestions for RM review — never as direct investment advice.
5. Flag uncertainty honestly. Say "we are not sure" when the data doesn't support a conclusion.
6. Use professional but accessible language suitable for a client meeting.
7. The RM retains final authority over all recommendations.
"""


def _call_gemini(prompt: str, context: str = "") -> str:
    """Call the Gemini API with the given prompt. Falls back to template if unavailable."""
    if not _gemini_available or _client is None:
        return _fallback_response(prompt)

    try:
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = _client.models.generate_content(
            model=MODEL,
            contents=full_prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.3,
                "max_output_tokens": 8192,
            },
        )
        return response.text
    except Exception as e:
        return f"⚠️ AI service unavailable: {str(e)}\n\nPlease configure your GEMINI_API_KEY in the .env file to enable AI-powered insights."


def _fallback_response(prompt: str) -> str:
    """Provide a helpful fallback when Gemini API is not configured."""
    return (
        "🔑 **AI Advisory Engine — API Key Required**\n\n"
        "To enable AI-powered insights, please add your Gemini API key to the `.env` file:\n\n"
        "```\nGEMINI_API_KEY=your-actual-api-key\n```\n\n"
        "Get a free API key at: https://aistudio.google.com/apikey\n\n"
        "Once configured, this panel will generate:\n"
        "- Portfolio explanations grounded in event data\n"
        "- Personalised rebalancing suggestions\n"
        "- Tax-aware optimization opportunities\n"
        "- Life-event wealth planning strategies\n"
        "- Draft client messages for RM review"
    )


def _format_event_log(events_df: pd.DataFrame) -> str:
    """Format event_log.csv for inclusion in prompts."""
    lines = ["EVENT LOG (authoritative source for 2026 events):"]
    for _, e in events_df.iterrows():
        lines.append(
            f"- {e['event_date']} [{e['severity']}] ({e['event_type']}, {e['region']}): "
            f"{e['description']} | Transmission: {e['primary_transmission']}"
        )
    return "\n".join(lines)


def _format_client_context(client_detail: dict) -> str:
    """Build a comprehensive client context string for prompts."""
    c = client_detail["client"]
    lines = [
        f"CLIENT: {c.get('client_name', 'Unknown')} ({c.get('client_id', '')})",
        f"Age: {c.get('age', 'N/A')}, Gender: {c.get('gender', 'N/A')}, Nationality: {c.get('nationality', 'N/A')}",
        f"Country of Residence: {c.get('country_of_residence', 'N/A')}, Tax Domicile: {c.get('tax_domicile', 'N/A')}",
        f"Risk Profile: {c.get('risk_profile', 'N/A')} (Score: {c.get('risk_tolerance_score', 'N/A')}/10)",
        f"Investment Horizon: {c.get('investment_horizon_years', 'N/A')} years, Liquidity Needs: {c.get('liquidity_needs', 'N/A')}",
        f"Life Stage: {c.get('life_stage', 'N/A')}",
        f"Source of Wealth: {c.get('source_of_wealth', 'N/A')}",
        f"Total AUM (USD): {c.get('total_aum_usd', 0):,.0f}",
        f"Objectives: {c.get('objectives', 'N/A')}",
    ]

    # RM Notes
    notes = client_detail.get("rm_notes", [])
    if notes:
        lines.append("\nRM NOTES:")
        for n in notes:
            lines.append(f"- {n['note_date']} ({n['channel']}): {n['note']}")

    # Portfolios
    pfs = client_detail.get("portfolios", pd.DataFrame())
    if not pfs.empty if isinstance(pfs, pd.DataFrame) else False:
        lines.append("\nPORTFOLIOS:")
        for _, pf in pfs.iterrows():
            lines.append(
                f"- {pf['portfolio_id']}: {pf['portfolio_name']} "
                f"({pf['service_model']}, {pf['mandate_name']}, {pf['base_currency']}) "
                f"AUM: {pf.get('aum_usd_current', 0):,.0f} USD"
            )

    # Cash Needs
    cn = client_detail.get("cash_needs", pd.DataFrame())
    if not cn.empty if isinstance(cn, pd.DataFrame) else False:
        lines.append("\nPLANNED CASH NEEDS:")
        for _, need in cn.iterrows():
            lines.append(
                f"- {need['description']}: {need['currency']} {need['amount']:,.0f} "
                f"(Due: {need['due_from']} to {need['due_to']}, {need['certainty']})"
            )

    # Commitments
    comms = client_detail.get("commitments", pd.DataFrame())
    if not comms.empty if isinstance(comms, pd.DataFrame) else False:
        lines.append("\nUNCALLED COMMITMENTS:")
        for _, comm in comms.iterrows():
            lines.append(
                f"- {comm['fund_name']}: USD {comm['uncalled']:,.0f} uncalled "
                f"(Window: {comm['expected_call_window']})"
            )

    # Credit Facilities
    facs = client_detail.get("facilities", pd.DataFrame())
    if not facs.empty if isinstance(facs, pd.DataFrame) else False:
        lines.append("\nCREDIT FACILITIES:")
        for _, fac in facs.iterrows():
            lines.append(
                f"- {fac['facility_id']}: {fac['facility_type']} {fac['facility_ccy']} "
                f"{fac['credit_limit']:,.0f}, Current LTV: {fac.get('ltv_pct_2026-08-26', 'N/A')}%, "
                f"Margin Call Trigger: {fac['margin_call_ltv_pct']}%"
            )

    return "\n".join(lines)


def _format_holdings(holdings_df: pd.DataFrame) -> str:
    """Format holdings data for prompts."""
    if holdings_df.empty:
        return "No holdings data available."

    lines = ["CURRENT HOLDINGS (2026-08-26):"]
    for _, h in holdings_df.head(20).iterrows():
        lines.append(
            f"- {h['instrument_name']} ({h['asset_class']}): "
            f"USD {h['market_value_usd']:,.0f} ({h['weight_pct']:.1f}%), "
            f"Unrealised P&L: {h['unrealised_pnl_pct']:.1f}%, "
            f"Liquidity: {h.get('liquidity_tier', 'N/A')}"
        )
    return "\n".join(lines)


def _format_attributions(attributions: list) -> str:
    """Format event attributions for prompts."""
    if not attributions:
        return "No event attributions available."

    lines = ["PORTFOLIO CHANGE ATTRIBUTIONS:"]
    for attr in attributions[:15]:
        evt_descs = "; ".join(
            [f"{e['event_date']}: {e['description'][:100]}" for e in attr.get("events", [])]
        )
        lines.append(
            f"- {attr['instrument_name']}: "
            f"USD {attr['mv_change_usd']:+,.0f} ({attr['mv_change_pct']:+.1f}%) "
            f"[{attr['period']}] — Events: {evt_descs}"
        )
    return "\n".join(lines)


def _fallback_portfolio_explanation(client_detail: dict, holdings_df: pd.DataFrame,
                                    attributions: list) -> str:
    """Create a useful grounded explanation when Gemini is unavailable."""
    client = client_detail.get("client", {})
    portfolios = client_detail.get("portfolios", pd.DataFrame())
    client_name = client.get("client_name", "the client")
    current_aum = float(client.get("total_aum_usd", 0) or 0)
    lines = [
        "### Portfolio Overview",
        f"{client_name}'s current reported AUM is approximately ${current_aum:,.0f}.",
    ]

    if isinstance(portfolios, pd.DataFrame) and not portfolios.empty:
        portfolio_names = ", ".join(str(name) for name in portfolios["portfolio_name"].tolist())
        lines.append(f"The client has {len(portfolios)} portfolio(s): {portfolio_names}.")

    lines.append("\n### Key Drivers")
    if attributions:
        for attribution in attributions[:5]:
            events = attribution.get("events", [])
            event_text = "; ".join(
                f"{event['event_date']}: {event['description']}" for event in events[:2]
            ) or "No matching event-log entry was identified."
            lines.append(
                f"- **{attribution['instrument_name']}** changed by "
                f"USD {attribution['mv_change_usd']:+,.0f} "
                f"({attribution['mv_change_pct']:+.1f}%) during {attribution['period']}. "
                f"Relevant event context: {event_text}"
            )
    else:
        lines.append("No significant position changes were identified in the available snapshots.")

    lines.extend([
        "\n### Risk Observations",
        f"The current review covers {len(holdings_df)} holding row(s). "
        "Mandate, concentration, liquidity, and credit metrics should be reviewed alongside this summary.",
        "\n### Review Note",
        "This explanation was generated from the loaded portfolio and event-log data because the AI service was unavailable. "
        "It is for RM review and should not be treated as direct investment advice.",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public Advisory Functions
# ---------------------------------------------------------------------------

def generate_portfolio_explanation(
    client_detail: dict,
    holdings_df: pd.DataFrame,
    attributions: list,
    events_df: pd.DataFrame,
) -> str:
    """
    Generate a narrative explanation of what happened to a client's portfolio
    and why, grounded in event_log.csv.
    """
    context = "\n\n".join([
        _format_event_log(events_df),
        _format_client_context(client_detail),
        _format_holdings(holdings_df),
        _format_attributions(attributions),
    ])

    prompt = """Based on the provided data, generate a clear and concise portfolio explanation for the RM.

Structure your response as:
1. **Portfolio Overview** — What happened to this client's portfolio year-to-date (total AUM change)
2. **Key Drivers** — The 3-5 most significant position changes, each attributed to specific events from the EVENT LOG with dates
3. **Risk Observations** — Any concerning patterns (concentration, mandate drift, liquidity, LTV)
4. **What the RM Notes Tell Us** — Insights from Priscilla's notes that add context the numbers don't show

IMPORTANT: Only cite events from the EVENT LOG provided. Do not reference any events not in the data."""

    if not is_ai_available():
        return _fallback_portfolio_explanation(client_detail, holdings_df, attributions)

    response = _call_gemini(prompt, context)
    if not response or response.startswith("⚠️ AI service unavailable"):
        return _fallback_portfolio_explanation(client_detail, holdings_df, attributions)
    return response


def generate_rebalancing_suggestion(
    client_detail: dict,
    holdings_df: pd.DataFrame,
    mandate_drift: pd.DataFrame,
    events_df: pd.DataFrame,
) -> str:
    """
    Generate actionable rebalancing proposals with underlying rationale.
    """
    context = "\n\n".join([
        _format_event_log(events_df),
        _format_client_context(client_detail),
        _format_holdings(holdings_df),
    ])

    drift_text = "MANDATE DRIFT ANALYSIS:\n"
    if not mandate_drift.empty:
        for _, d in mandate_drift.iterrows():
            drift_text += (
                f"- {d['asset_class']}: {d['actual_weight_pct']:.1f}% actual vs "
                f"{d['min_pct']}-{d['max_pct']}% band (target {d['target_pct']}%) — {d['breach']}\n"
            )
    else:
        drift_text += "No breaches detected.\n"

    context += f"\n\n{drift_text}"

    prompt = """Based on the portfolio data, mandate bands, and market context, generate specific rebalancing suggestions.

Structure your response as:
1. **Mandate Compliance** — Which bands are breached and by how much
2. **Proposed Rebalancing Actions** — Specific buy/sell suggestions with estimated amounts, explaining WHY each action is appropriate given the client's mandate, risk profile, and current market conditions
3. **Implementation Considerations** — Liquidity, timing, tax, and cost factors
4. **Risks of Inaction** — What could happen if the current allocation persists

Each suggestion should reference the client's stated objectives and risk tolerance.
Mark each as [SUGGESTION FOR RM REVIEW] — the RM decides what to implement."""

    return _call_gemini(prompt, context)


def generate_tax_optimization(
    client_detail: dict,
    tax_data: dict,
    events_df: pd.DataFrame,
) -> str:
    """
    Generate tax-aware optimization opportunities based on unrealised P&L
    and the client's tax domicile.
    """
    context = "\n\n".join([
        _format_event_log(events_df),
        _format_client_context(client_detail),
    ])

    tax_text = (
        f"\nTAX ANALYSIS:\n"
        f"Tax Domicile: {tax_data.get('tax_domicile', 'Unknown')}\n"
        f"Total Unrealised Gains: {tax_data.get('total_gains', 0):,.0f}\n"
        f"Total Unrealised Losses: {tax_data.get('total_losses', 0):,.0f}\n"
        f"Net Unrealised P&L: {tax_data.get('net_pnl', 0):,.0f}\n"
    )

    losses = tax_data.get("losses", [])
    if losses:
        tax_text += "\nPositions with Unrealised Losses:\n"
        for loss in losses[:10]:
            tax_text += (
                f"- {loss['instrument_name']}: {loss['unrealised_pnl_base']:,.0f} "
                f"({loss['unrealised_pnl_pct']:.1f}%)\n"
            )

    gains = tax_data.get("gains", [])
    if gains:
        tax_text += "\nPositions with Unrealised Gains:\n"
        for gain in gains[:10]:
            tax_text += (
                f"- {gain['instrument_name']}: {gain['unrealised_pnl_base']:+,.0f} "
                f"({gain['unrealised_pnl_pct']:.1f}%)\n"
            )

    context += tax_text

    prompt = """Based on the client's tax domicile, unrealised P&L positions, and planned cash needs, suggest tax-efficient strategies.

Structure your response as:
1. **Tax Domicile Considerations** — Key tax rules for this jurisdiction relevant to portfolio decisions
2. **Tax-Loss Harvesting Opportunities** — Specific positions that could be sold at a loss to offset gains
3. **Gain Deferral Strategies** — Ways to manage timing of realising gains
4. **Interaction with Cash Needs** — How tax planning should coordinate with upcoming liquidity requirements
5. **Important Caveats** — What needs to be confirmed with a tax advisor before implementing

Mark all suggestions as [REQUIRES TAX ADVISOR REVIEW]."""

    return _call_gemini(prompt, context)


def generate_life_event_plan(
    client_detail: dict,
    events_df: pd.DataFrame,
) -> str:
    """
    Generate structured transition strategies for life events:
    retirement, business sale, succession, philanthropy, education.
    """
    context = "\n\n".join([
        _format_event_log(events_df),
        _format_client_context(client_detail),
    ])

    prompt = """Based on the client's life stage, objectives, planned cash needs, and current portfolio, create a life-event wealth planning strategy.

Structure your response as:
1. **Life Situation Summary** — Current stage, key upcoming transitions
2. **Wealth Transition Plan** — Specific phased recommendations tied to timelines from planned cash needs
3. **Portfolio Positioning** — How the portfolio should evolve to support these transitions
4. **Risk Factors** — What could go wrong and how to mitigate
5. **Recommended Next Steps** — Concrete actions for the RM to discuss with the client in the upcoming meeting

Each recommendation should be practical and respect the client's stated preferences from the RM notes.
Mark as [RM REVIEW REQUIRED] — Priscilla decides what to present to the client."""

    return _call_gemini(prompt, context)


def generate_client_message(
    client_detail: dict,
    action_summary: str,
    events_df: pd.DataFrame,
) -> str:
    """
    Draft an RM-to-client message for a proposed action.
    """
    context = "\n\n".join([
        _format_event_log(events_df),
        _format_client_context(client_detail),
    ])

    prompt = f"""Draft a professional but warm client communication for the RM to send.

ACTION BEING PROPOSED:
{action_summary}

Requirements:
- Address the client by name
- Reference their stated objectives and concerns from RM notes
- Explain the rationale in plain language (no jargon)
- Mention relevant market events from the EVENT LOG with dates
- Include a clear call to action (e.g., schedule a meeting)
- Keep it under 200 words
- Sign off as Priscilla Ong, Relationship Manager

Format as a ready-to-send email or message. Mark as [DRAFT — RM TO REVIEW AND EDIT BEFORE SENDING]."""

    return _call_gemini(prompt, context)


def suggest_data_repairs(findings: list, data: dict) -> list:
    """Return strictly structured repair proposals for the data-quality loop.

    The validator remains authoritative: proposals are limited to flagged cells
    and are checked against deterministic expected values before application.
    """
    if not findings or not _gemini_available or _client is None:
        return []

    compact_findings = []
    for finding in findings:
        compact_findings.append({
            "finding_id": finding.get("finding_id"),
            "dataset": finding.get("dataset"),
            "row_index": finding.get("row_index"),
            "field": finding.get("field"),
            "message": finding.get("message"),
            "current_value": finding.get("current_value"),
            "expected_value": finding.get("expected_value"),
        })

    prompt = """You are reviewing small data-quality flaws in a private-bank dataset.
Return JSON only: an object with a `repairs` array. Each item must contain:
`finding_id`, `dataset`, `row_index`, `field`, `value`, and `rationale`.
Only propose the exact expected_value already supplied for a finding. Do not
invent values, rewrite rows, alter source files, or repair business decisions.
If no repair is justified, return {"repairs": []}.

FINDINGS:
""" + json.dumps(compact_findings, default=str)
    response = _call_gemini(prompt)
    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        payload = json.loads(text)
        repairs = payload.get("repairs", [])
        return repairs if isinstance(repairs, list) else []
    except (AttributeError, ValueError, TypeError, json.JSONDecodeError):
        return []


def is_ai_available() -> bool:
    """Check if the Gemini AI service is configured and available."""
    return _gemini_available

