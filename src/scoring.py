"""
Client Prioritization Scoring Engine for the RM Intelligence Workbench.

Computes a 1-100 urgency score for each of Priscilla Ong's 20 clients,
combining weighted factors: LTV proximity, cash needs, mandate drift,
portfolio losses, concentration risk, life events, KYC, and sustainability.
"""

from datetime import datetime

import pandas as pd

from src.analytics import (
    TODAY,
    compute_concentration_risk,
    compute_liquidity_coverage,
    compute_ltv_analysis,
    compute_mandate_drift,
    compute_portfolio_summary,
    check_sustainability_breaches,
)


def compute_priority_scores(data: dict) -> pd.DataFrame:
    """
    Score all clients 1-100 by urgency, combining multiple risk factors.

    Weights:
      - LTV breach proximity:     20
      - Cash needs within 30 days: 20
      - Mandate allocation drift:  15
      - Portfolio loss severity:   15
      - Concentration risk:        10
      - Life event urgency:        10
      - KYC review overdue:         5
      - Sustainability breach:      5
    """
    clients = data["clients"]
    portfolio_summary = compute_portfolio_summary(data)
    ltv_analysis = compute_ltv_analysis(data)
    mandate_drift = compute_mandate_drift(data)
    liquidity = compute_liquidity_coverage(data)
    concentration = compute_concentration_risk(data)
    sus_breaches = check_sustainability_breaches(data)

    scores = []

    for _, client in clients.iterrows():
        cid = client["client_id"]

        # --- Factor 1: LTV Breach Proximity (0-20) ---
        ltv_score = 0
        client_ltv = ltv_analysis[ltv_analysis["client_id"] == cid]
        if not client_ltv.empty:
            for _, fac in client_ltv.iterrows():
                headroom = fac.get("ltv_headroom_pct", 100)
                if headroom <= 0:
                    ltv_score = 20  # Already breached
                elif headroom < 2:
                    ltv_score = max(ltv_score, 18)
                elif headroom < 5:
                    ltv_score = max(ltv_score, 14)
                elif headroom < 10:
                    ltv_score = max(ltv_score, 10)
                elif headroom < 15:
                    ltv_score = max(ltv_score, 6)

        # --- Factor 2: Cash Needs within 30 days (0-20) ---
        cash_score = 0
        client_liq = liquidity[liquidity["client_id"] == cid]
        if not client_liq.empty:
            liq = client_liq.iloc[0]
            if liq["is_stressed"]:
                cash_score = 20
            elif liq["coverage_ratio"] < 1.5:
                cash_score = 15
            elif liq["coverage_ratio"] < 3:
                cash_score = 10
            elif liq["near_term_needs_usd"] > 0:
                cash_score = 5

        # --- Factor 3: Mandate Allocation Drift (0-15) ---
        drift_score = 0
        if not mandate_drift.empty:
            client_drift = mandate_drift[
                (mandate_drift["client_id"] == cid) & (mandate_drift["breach"] != "None")
            ]
            if not client_drift.empty:
                num_breaches = len(client_drift)
                max_breach = client_drift["breach_amount_pct"].max()
                if max_breach > 10:
                    drift_score = 15
                elif max_breach > 5:
                    drift_score = 12
                elif num_breaches >= 3:
                    drift_score = 10
                elif num_breaches >= 2:
                    drift_score = 8
                elif num_breaches >= 1:
                    drift_score = 5

        # --- Factor 4: Portfolio Loss Severity (0-15) ---
        loss_score = 0
        client_summary = portfolio_summary[portfolio_summary["client_id"] == cid]
        if not client_summary.empty:
            ytd_change = client_summary.iloc[0].get("ytd_change_pct", 0)
            if ytd_change < -15:
                loss_score = 15
            elif ytd_change < -10:
                loss_score = 12
            elif ytd_change < -5:
                loss_score = 9
            elif ytd_change < -2:
                loss_score = 6
            elif ytd_change < 0:
                loss_score = 3

        # --- Factor 5: Concentration Risk (0-10) ---
        conc_score = 0
        if not concentration.empty:
            client_conc = concentration[
                (concentration["client_id"] == cid) & (concentration["is_breach"])
            ]
            if not client_conc.empty:
                max_weight = client_conc["weight_pct"].max()
                if max_weight > 50:
                    conc_score = 10
                elif max_weight > 30:
                    conc_score = 8
                elif max_weight > 20:
                    conc_score = 6
                else:
                    conc_score = 4

        # --- Factor 6: Life Event Urgency (0-10) ---
        life_score = _score_life_event(client)

        # --- Factor 7: KYC Review Overdue (0-5) ---
        kyc_score = 0
        kyc_due = client.get("kyc_review_due", "")
        if isinstance(kyc_due, str) and kyc_due:
            try:
                kyc_dt = datetime.strptime(kyc_due, "%Y-%m-%d")
                today_dt = datetime.strptime(TODAY, "%Y-%m-%d")
                if kyc_dt < today_dt:
                    days_overdue = (today_dt - kyc_dt).days
                    kyc_score = min(5, 1 + days_overdue // 30)
                elif (kyc_dt - today_dt).days < 30:
                    kyc_score = 2
            except (ValueError, TypeError):
                pass

        # --- Factor 8: Sustainability Breach (0-5) ---
        sus_score = 0
        if not sus_breaches.empty:
            client_sus = sus_breaches[sus_breaches["client_id"] == cid]
            if not client_sus.empty:
                sus_score = min(5, len(client_sus) * 2 + 1)

        # --- Combine ---
        total = ltv_score + cash_score + drift_score + loss_score + conc_score + life_score + kyc_score + sus_score
        total = min(100, max(1, total))

        # Determine action tag
        if total >= 70:
            action_tag = "URGENT"
        elif total >= 45:
            action_tag = "REVIEW"
        elif total >= 20:
            action_tag = "MONITOR"
        else:
            action_tag = "ON TRACK"

        # Collect risk flags
        risk_flags = []
        if ltv_score >= 14:
            risk_flags.append("LTV Near Trigger")
        if ltv_score == 20:
            risk_flags.append("LTV BREACHED")
        if cash_score >= 15:
            risk_flags.append("Liquidity Stress")
        if drift_score >= 10:
            risk_flags.append("Mandate Breach")
        if loss_score >= 9:
            risk_flags.append("Significant Losses")
        if conc_score >= 6:
            risk_flags.append("Concentration Risk")
        if life_score >= 7:
            risk_flags.append("Life Event Imminent")
        if kyc_score >= 3:
            risk_flags.append("KYC Overdue")
        if sus_score >= 3:
            risk_flags.append("Sustainability Breach")

        scores.append({
            "client_id": cid,
            "client_name": client["client_name"],
            "priority_score": total,
            "action_tag": action_tag,
            "total_aum_usd": client["total_aum_usd"],
            "wealth_band": client["wealth_band"],
            "risk_profile": client["risk_profile"],
            "life_stage": client["life_stage"],
            "risk_flags": ", ".join(risk_flags) if risk_flags else "No critical flags",
            "ltv_score": ltv_score,
            "cash_score": cash_score,
            "drift_score": drift_score,
            "loss_score": loss_score,
            "conc_score": conc_score,
            "life_score": life_score,
            "kyc_score": kyc_score,
            "sus_score": sus_score,
        })

    df = pd.DataFrame(scores).sort_values("priority_score", ascending=False).reset_index(drop=True)
    return df


def _score_life_event(client: pd.Series) -> int:
    """Score life event urgency based on life_stage and objectives."""
    life_stage = str(client.get("life_stage", "")).lower()
    objectives = str(client.get("objectives", "")).lower()
    age = client.get("age", 0)

    score = 0

    # Succession planning with advanced age
    if "succession" in life_stage and age and age > 75:
        score = max(score, 10)
    elif "succession" in life_stage:
        score = max(score, 7)

    # Pre-retirement
    if "pre-retirement" in life_stage or "retire" in life_stage:
        score = max(score, 7)

    # Recently inherited — transition needed
    if "inherited" in life_stage and "transition" in life_stage:
        score = max(score, 8)

    # Pre-liquidity event
    if "pre-liquidity" in life_stage:
        score = max(score, 6)

    # Retired and elderly
    if "retired" in life_stage:
        if age and age > 70:
            score = max(score, 6)
        else:
            score = max(score, 4)

    # Foundation / philanthropy goals
    if "foundation" in objectives or "philanthropy" in objectives:
        score = max(score, 3)

    # Education funding
    if "education" in objectives or "university" in objectives:
        score = max(score, 3)

    return score


def get_top_priorities(data: dict, n: int = 5) -> pd.DataFrame:
    """Get the top N priority clients for immediate outreach."""
    scores = compute_priority_scores(data)
    return scores.head(n)


def get_score_breakdown(data: dict, client_id: str) -> dict:
    """Get the detailed score breakdown for a specific client."""
    scores = compute_priority_scores(data)
    client_score = scores[scores["client_id"] == client_id]
    if client_score.empty:
        return {}
    return client_score.iloc[0].to_dict()

