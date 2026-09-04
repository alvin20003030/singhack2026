"""
RM Intelligence Workbench — Main Streamlit Application

Three-screen interactive workbench for Relationship Manager Priscilla Ong:
  1. Book Prioritizer — Ranked client list with priority scores and action tags
  2. Client Deep-Dive — Time-series analysis with event attribution
  3. Advisory Action Builder — AI-powered recommendations with RM edit/override
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.analytics import (
    load_all_data,
    compute_portfolio_summary,
    compute_ltv_analysis,
    compute_mandate_drift,
    compute_liquidity_coverage,
    compute_concentration_risk,
    compute_unrealised_pnl,
    attribute_portfolio_changes,
    check_sustainability_breaches,
    get_client_detail,
    get_asset_allocation,
    get_holdings_detail,
    get_client_portfolio_timeseries,
    get_tax_optimization_opportunities,
    SNAPSHOT_DATES,
    TODAY,
)
from src.scoring import compute_priority_scores, get_score_breakdown
from src.ai_advisor import (
    generate_portfolio_explanation,
    generate_rebalancing_suggestion,
    generate_tax_optimization,
    generate_life_event_plan,
    generate_client_message,
    is_ai_available,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RM Intelligence Workbench",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_data():
    return load_all_data()

@st.cache_data(ttl=300)
def get_scores(_data):
    return compute_priority_scores(_data)

data = load_data()

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=64)
st.sidebar.title("RM Intelligence Workbench")
st.sidebar.caption("Priscilla Ong · Asia Desk · 26 Aug 2026")
st.sidebar.divider()

screen = st.sidebar.radio(
    "Navigate",
    ["📊 Book Prioritizer", "🔍 Client Deep-Dive", "💡 Advisory Action Builder"],
    index=0,
)

# AI status indicator
if is_ai_available():
    st.sidebar.success("✅ Gemini AI Connected")
else:
    st.sidebar.warning("⚠️ AI Offline — Add API key to .env")

st.sidebar.divider()
st.sidebar.caption("SingHacks 2026 · Julius Baer Wealth Intelligence")
st.sidebar.caption(f"Data as of: {TODAY}")


# ═══════════════════════════════════════════════════════════════════════════
# SCREEN 1: BOOK PRIORITIZER
# ═══════════════════════════════════════════════════════════════════════════

def render_book_prioritizer():
    st.title("📊 Book Prioritizer")
    st.markdown("Ranked view of Priscilla's 20-client book — **who to call first today.**")

    scores = get_scores(data)
    portfolio_summary = compute_portfolio_summary(data)

    # --- Top-line metrics ---
    total_aum = data["clients"]["total_aum_usd"].sum()
    urgent_count = len(scores[scores["action_tag"] == "URGENT"])
    review_count = len(scores[scores["action_tag"] == "REVIEW"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Book AUM", f"${total_aum / 1e6:,.0f}M")
    col2.metric("Clients", len(scores))
    col3.metric("🔴 Urgent", urgent_count)
    col4.metric("🟡 Review", review_count)

    st.divider()

    # --- Top 3 Priority Actions ---
    st.subheader("🚨 Today's Top Priority Actions")
    top3 = scores.head(3)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top3.iterrows()):
        with cols[idx]:
            color = _action_color(row["action_tag"])
            st.markdown(
                f"<div style='padding:16px; border-radius:10px; border-left: 5px solid {color}; "
                f"background-color: {color}15;'>"
                f"<h4 style='margin:0;'>{row['client_name']}</h4>"
                f"<p style='margin:4px 0; font-size:28px; font-weight:bold; color:{color};'>"
                f"{row['priority_score']}/100</p>"
                f"<p style='margin:2px 0;'>AUM: ${row['total_aum_usd']/1e6:,.1f}M</p>"
                f"<p style='margin:2px 0; font-size:13px;'>⚡ {row['risk_flags']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # --- Full Ranked Table ---
    st.subheader("📋 Full Client Ranking")

    # Merge with YTD performance
    display_df = scores.merge(
        portfolio_summary[["client_id", "ytd_change_pct", "ytd_change_abs"]],
        on="client_id",
        how="left",
    )

    # Format for display
    display_df["AUM ($M)"] = display_df["total_aum_usd"].apply(lambda x: f"${x/1e6:,.1f}M")
    display_df["YTD Change"] = display_df["ytd_change_pct"].apply(
        lambda x: f"{'↑' if x >= 0 else '↓'} {abs(x):.1f}%"
    )
    display_df["Score"] = display_df["priority_score"]

    # Show table
    for _, row in display_df.iterrows():
        color = _action_color(row["action_tag"])
        tag_emoji = {"URGENT": "🔴", "REVIEW": "🟡", "MONITOR": "🔵", "ON TRACK": "🟢"}.get(
            row["action_tag"], "⚪"
        )

        with st.container():
            c1, c2, c3, c4, c5, c6 = st.columns([0.5, 2.5, 1, 1, 1, 3])
            c1.markdown(f"**{row['Score']}**")
            c2.markdown(f"**{row['client_name']}** · {row['client_id']}")
            c3.markdown(f"{row['AUM ($M)']}")
            c4.markdown(f"{row['YTD Change']}")
            c5.markdown(f"{tag_emoji} {row['action_tag']}")
            c6.markdown(f"_{row['risk_flags']}_")

    st.divider()

    # --- Score Distribution ---
    st.subheader("📈 Score Distribution")
    fig = px.bar(
        scores,
        x="client_name",
        y="priority_score",
        color="action_tag",
        color_discrete_map={
            "URGENT": "#dc3545",
            "REVIEW": "#ffc107",
            "MONITOR": "#0d6efd",
            "ON TRACK": "#198754",
        },
        title="Client Priority Scores",
        labels={"priority_score": "Priority Score", "client_name": "Client"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
# SCREEN 2: CLIENT DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════════════

def render_client_deep_dive():
    st.title("🔍 Client Deep-Dive")

    # Client selector
    clients = data["clients"]
    client_options = {
        f"{row['client_name']} ({row['client_id']}) — ${row['total_aum_usd']/1e6:,.1f}M": row["client_id"]
        for _, row in clients.sort_values("total_aum_usd", ascending=False).iterrows()
    }

    selected_label = st.selectbox("Select Client", list(client_options.keys()))
    client_id = client_options[selected_label]

    detail = get_client_detail(data, client_id)
    client = detail["client"]
    score_detail = get_score_breakdown(data, client_id)

    # --- Client Profile Header ---
    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader(f"{client['client_name']}")
        st.markdown(
            f"**{client.get('life_stage', '')}** · {client.get('nationality', '')} · "
            f"Age {client.get('age', 'N/A')} · Tax Domicile: {client.get('tax_domicile', '')}"
        )
        st.markdown(f"**Risk Profile:** {client.get('risk_profile', '')} ({client.get('risk_tolerance_score', '')}/10)")
        st.markdown(f"**Objectives:** {client.get('objectives', '')}")
        st.markdown(f"**Source of Wealth:** {client.get('source_of_wealth', '')}")

    with col2:
        score = score_detail.get("priority_score", 0)
        tag = score_detail.get("action_tag", "N/A")
        color = _action_color(tag)
        st.markdown(
            f"<div style='text-align:center; padding:20px; border-radius:10px; "
            f"background-color:{color}15; border: 2px solid {color};'>"
            f"<p style='margin:0; font-size:14px;'>Priority Score</p>"
            f"<p style='margin:0; font-size:48px; font-weight:bold; color:{color};'>{score}</p>"
            f"<p style='margin:0; font-size:16px; color:{color};'>{tag}</p></div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.metric("Total AUM", f"${client.get('total_aum_usd', 0)/1e6:,.1f}M")
        st.metric("Booking Centre", client.get("booking_centre", ""))
        st.metric("KYC Due", client.get("kyc_review_due", "N/A"))

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Portfolio Timeline",
        "⚖️ Asset Allocation & Mandate",
        "📋 Holdings",
        "🏦 Credit & Liquidity",
        "📝 RM Notes",
    ])

    # --- Tab 1: Portfolio Timeline ---
    with tab1:
        st.subheader("Portfolio AUM Over Time")
        ts = get_client_portfolio_timeseries(data, client_id)
        if not ts.empty:
            # Total AUM per snapshot
            total_ts = ts.groupby("snapshot_date")["aum"].sum().reset_index()
            total_ts = total_ts.sort_values("snapshot_date")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=total_ts["snapshot_date"],
                y=total_ts["aum"],
                mode="lines+markers",
                name="Total AUM",
                line=dict(width=3, color="#0d6efd"),
                marker=dict(size=10),
            ))

            # Per-portfolio lines
            for pf_name in ts["portfolio_name"].unique():
                pf_ts = ts[ts["portfolio_name"] == pf_name].sort_values("snapshot_date")
                fig.add_trace(go.Scatter(
                    x=pf_ts["snapshot_date"],
                    y=pf_ts["aum"],
                    mode="lines+markers",
                    name=pf_name,
                    line=dict(width=1, dash="dot"),
                    marker=dict(size=6),
                ))

            # Event annotations
            events = data["event_log"]
            for _, evt in events[events["severity"].isin(["Severe", "High"])].iterrows():
                evt_date = evt["event_date"]
                closest_snap = min(SNAPSHOT_DATES, key=lambda d: abs(
                    pd.Timestamp(d) - pd.Timestamp(evt_date)
                ))
                snap_aum = total_ts[total_ts["snapshot_date"] == closest_snap]
                if not snap_aum.empty:
                    fig.add_annotation(
                        x=closest_snap,
                        y=snap_aum.iloc[0]["aum"],
                        text=evt["description"][:50] + "...",
                        showarrow=True,
                        arrowhead=2,
                        font=dict(size=9),
                        bgcolor="rgba(255,255,255,0.8)",
                    )

            fig.update_layout(
                height=450,
                title="AUM Across 5 Snapshots with Key Events",
                xaxis_title="Snapshot Date",
                yaxis_title=f"AUM ({ts.iloc[0]['base_currency'] if not ts.empty else 'USD'})",
                hovermode="x unified",
            )
            st.plotly_chart(fig, width="stretch")

        # Event Attribution
        st.subheader("🔗 Event Attribution — What Drove Changes")
        attributions = attribute_portfolio_changes(data, client_id)
        if attributions:
            for attr in attributions[:10]:
                change_emoji = "📈" if attr["mv_change_usd"] > 0 else "📉"
                with st.expander(
                    f"{change_emoji} {attr['instrument_name']} — "
                    f"USD {attr['mv_change_usd']:+,.0f} ({attr['mv_change_pct']:+.1f}%) "
                    f"[{attr['period']}]"
                ):
                    for evt in attr["events"]:
                        sev_color = {"Severe": "🔴", "High": "🟠", "Medium": "🟡"}.get(
                            evt["severity"], "⚪"
                        )
                        st.markdown(
                            f"{sev_color} **{evt['event_date']}** [{evt['severity']}]: "
                            f"{evt['description']}"
                        )
                        st.caption(f"Transmission: {evt['transmission']}")
        else:
            st.info("No significant attributable changes detected.")

    # --- Tab 2: Asset Allocation & Mandate ---
    with tab2:
        st.subheader("Asset Allocation vs Mandate Bands")

        alloc = get_asset_allocation(data, client_id)
        drift = compute_mandate_drift(data)
        client_drift = drift[drift["client_id"] == client_id] if not drift.empty else pd.DataFrame()

        if not alloc.empty:
            col_a, col_b = st.columns([1, 1])

            with col_a:
                fig = px.pie(
                    alloc,
                    values="market_value_usd",
                    names="asset_class",
                    title="Current Asset Allocation",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, width="stretch")

            with col_b:
                if not client_drift.empty:
                    fig = go.Figure()
                    for _, d in client_drift.iterrows():
                        color = "#dc3545" if d["breach"] != "None" else "#198754"
                        fig.add_trace(go.Bar(
                            name=d["asset_class"],
                            x=[d["asset_class"]],
                            y=[d["actual_weight_pct"]],
                            marker_color=color,
                            text=[f"{d['actual_weight_pct']:.1f}%"],
                            textposition="outside",
                        ))
                        # Mandate band as error bars area
                        fig.add_shape(
                            type="rect",
                            x0=d["asset_class"],
                            x1=d["asset_class"],
                            y0=d["min_pct"],
                            y1=d["max_pct"],
                            line=dict(color="rgba(0,0,0,0.3)", width=2, dash="dash"),
                            fillcolor="rgba(0,100,200,0.05)",
                            xref=f"x",
                            yref="y",
                        )

                    fig.update_layout(
                        title="Actual vs Mandate Bands",
                        height=350,
                        showlegend=False,
                        yaxis_title="Weight (%)",
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No mandate data (custody account or no active mandate).")

        # Mandate breaches table
        if not client_drift.empty:
            breaches = client_drift[client_drift["breach"] != "None"]
            if not breaches.empty:
                st.warning(f"⚠️ {len(breaches)} mandate breach(es) detected")
                st.dataframe(
                    breaches[["portfolio_name", "asset_class", "actual_weight_pct",
                              "min_pct", "max_pct", "breach", "breach_amount_pct"]],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.success("✅ All asset classes within mandate bands")

    # --- Tab 3: Holdings ---
    with tab3:
        st.subheader("Current Holdings")
        holdings = get_holdings_detail(data, client_id)
        if not holdings.empty:
            display_h = holdings[[
                "portfolio_id", "instrument_name", "asset_class", "weight_pct",
                "market_value_usd", "unrealised_pnl_base", "unrealised_pnl_pct",
                "liquidity_tier",
            ]].copy()
            display_h.columns = [
                "Portfolio", "Instrument", "Asset Class", "Weight %",
                "MV (USD)", "Unrealised P&L", "P&L %", "Liquidity",
            ]
            display_h["MV (USD)"] = display_h["MV (USD)"].apply(lambda x: f"${x:,.0f}")
            display_h["Unrealised P&L"] = display_h["Unrealised P&L"].apply(
                lambda x: f"{'+'if x>0 else ''}{x:,.0f}"
            )
            display_h["P&L %"] = display_h["P&L %"].apply(lambda x: f"{x:+.1f}%")
            display_h["Weight %"] = display_h["Weight %"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_h, width="stretch", hide_index=True, height=500)

        # Concentration warnings
        conc = compute_concentration_risk(data)
        if not conc.empty:
            client_conc = conc[(conc["client_id"] == client_id) & (conc["is_breach"])]
            if not client_conc.empty:
                st.warning("⚠️ Concentration Limit Breaches")
                for _, c in client_conc.iterrows():
                    underlying = c.get("underlying_reference", "")
                    look_through = f" | **Look-through:** {underlying}" if underlying else ""
                    st.markdown(
                        f"- **{c['instrument_name']}**: {c['weight_pct']:.1f}% "
                        f"(limit: {c['max_single_position_pct']:.0f}%){look_through}"
                    )

    # --- Tab 4: Credit & Liquidity ---
    with tab4:
        st.subheader("Credit Facilities")
        facilities = detail["facilities"]
        if not facilities.empty:
            for _, fac in facilities.iterrows():
                ltv_col = f"ltv_pct_{TODAY}"
                current_ltv = fac.get(ltv_col, 0)
                trigger = fac["margin_call_ltv_pct"]
                headroom = trigger - current_ltv

                st.markdown(
                    f"**{fac['facility_id']}** — {fac['facility_type']} "
                    f"({fac['facility_ccy']} {fac['credit_limit']:,.0f})"
                )

                # LTV Gauge
                gauge_color = "#198754" if headroom > 10 else ("#ffc107" if headroom > 3 else "#dc3545")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=current_ltv,
                    delta={"reference": trigger, "decreasing": {"color": "#198754"}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, trigger * 0.7], "color": "#d4edda"},
                            {"range": [trigger * 0.7, trigger * 0.9], "color": "#fff3cd"},
                            {"range": [trigger * 0.9, 100], "color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": trigger,
                        },
                    },
                    title={"text": f"Current LTV vs Trigger ({trigger}%)"},
                    number={"suffix": "%"},
                ))
                fig.update_layout(height=280)
                st.plotly_chart(fig, width="stretch")

                # LTV History
                ltv_history = []
                for date in SNAPSHOT_DATES:
                    ltv_val = fac.get(f"ltv_pct_{date}", None)
                    if ltv_val is not None:
                        ltv_history.append({"Date": date, "LTV %": ltv_val})

                if ltv_history:
                    ltv_df = pd.DataFrame(ltv_history)
                    fig2 = px.line(
                        ltv_df, x="Date", y="LTV %",
                        title="LTV History Across Snapshots",
                        markers=True,
                    )
                    fig2.add_hline(
                        y=trigger, line_dash="dash", line_color="red",
                        annotation_text=f"Margin Call Trigger ({trigger}%)",
                    )
                    fig2.update_layout(height=300)
                    st.plotly_chart(fig2, width="stretch")

                st.divider()
        else:
            st.info("No credit facilities for this client.")

        # Liquidity Coverage
        st.subheader("Liquidity Coverage Analysis")
        liquidity = compute_liquidity_coverage(data)
        client_liq = liquidity[liquidity["client_id"] == client_id]
        if not client_liq.empty:
            liq = client_liq.iloc[0]
            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("Daily Liquid", f"${liq['daily_liquid_usd']/1e6:,.1f}M")
            lc2.metric("Near-Term Needs", f"${liq['near_term_needs_usd']/1e6:,.1f}M")
            lc3.metric("Uncalled Commitments", f"${liq['uncalled_commitments_usd']/1e6:,.1f}M")
            lc4.metric(
                "Coverage Ratio",
                f"{liq['coverage_ratio']:.1f}x" if liq['coverage_ratio'] < 100 else "✅ No obligations",
                delta="Stressed" if liq['is_stressed'] else "OK",
                delta_color="inverse" if liq['is_stressed'] else "normal",
            )

            # Cash needs detail
            cn = detail["cash_needs"]
            if not cn.empty:
                st.markdown("**Planned Cash Needs:**")
                st.dataframe(cn[["description", "currency", "amount", "due_from", "due_to", "certainty"]],
                             use_container_width=True, hide_index=True)

    # --- Tab 5: RM Notes ---
    with tab5:
        st.subheader("Priscilla's Notes")
        notes = detail["rm_notes"]
        if notes:
            for note in sorted(notes, key=lambda n: n["note_date"], reverse=True):
                with st.expander(f"📝 {note['note_date']} — {note['channel']}"):
                    st.markdown(note["note"])
        else:
            st.info("No RM notes for this client.")


# ═══════════════════════════════════════════════════════════════════════════
# SCREEN 3: ADVISORY ACTION BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def render_advisory_builder():
    st.title("💡 Advisory Action Builder")
    st.markdown(
        "AI-powered recommendations grounded in event data — "
        "**every suggestion is for RM review and override.**"
    )

    # Client selector
    clients = data["clients"]
    scores = get_scores(data)
    scored_options = {
        f"[{row['action_tag']}] {row['client_name']} ({row['client_id']}) — Score: {row['priority_score']}": row["client_id"]
        for _, row in scores.iterrows()
    }

    selected_label = st.selectbox("Select Client", list(scored_options.keys()))
    client_id = scored_options[selected_label]

    detail = get_client_detail(data, client_id)
    holdings = get_holdings_detail(data, client_id)
    events_df = data["event_log"]

    st.divider()

    # Advisory tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Portfolio Explanation",
        "⚖️ Rebalancing Suggestions",
        "💰 Tax-Aware Optimization",
        "🎯 Life-Event Planning",
    ])

    # --- Tab 1: Portfolio Explanation ---
    with tab1:
        st.subheader("📊 AI Portfolio Explanation")
        st.caption("Grounded in event_log.csv — only cites auditable market events")

        if st.button("🔄 Generate Explanation", key="gen_explain"):
            with st.spinner("Analyzing portfolio changes and attributing to events..."):
                attributions = attribute_portfolio_changes(data, client_id)
                explanation = generate_portfolio_explanation(
                    detail, holdings, attributions, events_df
                )
            st.session_state[f"explanation_{client_id}"] = explanation

        if f"explanation_{client_id}" in st.session_state:
            st.markdown(st.session_state[f"explanation_{client_id}"])

            st.divider()
            st.markdown("**✏️ RM Edit / Override:**")
            edited = st.text_area(
                "Edit the explanation before sharing with client:",
                value=st.session_state[f"explanation_{client_id}"],
                height=200,
                key=f"edit_explain_{client_id}",
            )
            if st.button("✅ Approve Edited Version", key=f"approve_explain_{client_id}"):
                st.success("✅ Explanation approved and saved for client communication.")

    # --- Tab 2: Rebalancing Suggestions ---
    with tab2:
        st.subheader("⚖️ AI Rebalancing Suggestions")
        st.caption("Respects mandate bands, risk profile, and client objectives")

        if st.button("🔄 Generate Rebalancing Plan", key="gen_rebalance"):
            with st.spinner("Analyzing mandate drift and generating rebalancing proposals..."):
                drift = compute_mandate_drift(data)
                client_drift = drift[drift["client_id"] == client_id] if not drift.empty else pd.DataFrame()
                suggestion = generate_rebalancing_suggestion(
                    detail, holdings, client_drift, events_df
                )
            st.session_state[f"rebalance_{client_id}"] = suggestion

        if f"rebalance_{client_id}" in st.session_state:
            st.markdown(st.session_state[f"rebalance_{client_id}"])

            st.divider()
            st.markdown("**✏️ RM Edit / Override:**")
            edited = st.text_area(
                "Modify the rebalancing suggestions:",
                value=st.session_state[f"rebalance_{client_id}"],
                height=200,
                key=f"edit_rebalance_{client_id}",
            )
            if st.button("✅ Approve Rebalancing Plan", key=f"approve_rebal_{client_id}"):
                st.success("✅ Rebalancing plan approved for implementation.")

    # --- Tab 3: Tax-Aware Optimization ---
    with tab3:
        st.subheader("💰 Tax-Aware Optimization")
        st.caption("Based on unrealised P&L and tax domicile")

        if st.button("🔄 Generate Tax Strategy", key="gen_tax"):
            with st.spinner("Analyzing unrealised P&L and tax domicile..."):
                tax_data = get_tax_optimization_opportunities(data, client_id)
                tax_advice = generate_tax_optimization(detail, tax_data, events_df)
            st.session_state[f"tax_{client_id}"] = tax_advice
            st.session_state[f"tax_data_{client_id}"] = tax_data

        if f"tax_{client_id}" in st.session_state:
            tax_data = st.session_state.get(f"tax_data_{client_id}", {})

            # Summary metrics
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Total Unrealised Gains", f"${tax_data.get('total_gains', 0):,.0f}")
            tc2.metric("Total Unrealised Losses", f"${tax_data.get('total_losses', 0):,.0f}")
            tc3.metric("Net Unrealised P&L", f"${tax_data.get('net_pnl', 0):,.0f}")

            st.markdown(st.session_state[f"tax_{client_id}"])

            st.divider()
            st.markdown("**✏️ RM Edit / Override:**")
            edited = st.text_area(
                "Modify the tax strategy:",
                value=st.session_state[f"tax_{client_id}"],
                height=200,
                key=f"edit_tax_{client_id}",
            )

    # --- Tab 4: Life-Event Planning ---
    with tab4:
        st.subheader("🎯 Life-Event Wealth Planning")
        st.caption("Structured strategies for retirement, succession, philanthropy, education")

        if st.button("🔄 Generate Life-Event Plan", key="gen_life"):
            with st.spinner("Building life-event transition strategy..."):
                plan = generate_life_event_plan(detail, events_df)
            st.session_state[f"life_{client_id}"] = plan

        if f"life_{client_id}" in st.session_state:
            st.markdown(st.session_state[f"life_{client_id}"])

            st.divider()
            st.markdown("**✏️ RM Edit / Override:**")
            edited = st.text_area(
                "Modify the life-event plan:",
                value=st.session_state[f"life_{client_id}"],
                height=200,
                key=f"edit_life_{client_id}",
            )

    # --- Client Message Generator ---
    st.divider()
    st.subheader("✉️ Draft Client Message")
    st.caption("AI drafts a message for Priscilla to review, edit, and send")

    action_summary = st.text_area(
        "Describe the action or insight to communicate to the client:",
        placeholder="e.g., Schedule a meeting to discuss portfolio rebalancing and upcoming cash needs...",
        key=f"msg_action_{client_id}",
    )

    if st.button("📧 Generate Draft Message", key=f"gen_msg_{client_id}"):
        if action_summary:
            with st.spinner("Drafting client message..."):
                msg = generate_client_message(detail, action_summary, events_df)
            st.session_state[f"msg_{client_id}"] = msg
        else:
            st.warning("Please describe the action to communicate.")

    if f"msg_{client_id}" in st.session_state:
        st.markdown("---")
        st.markdown("**📧 Draft Message:**")
        edited_msg = st.text_area(
            "Review and edit before sending:",
            value=st.session_state[f"msg_{client_id}"],
            height=250,
            key=f"edit_msg_{client_id}",
        )
        if st.button("✅ Approve & Mark Ready to Send", key=f"approve_msg_{client_id}"):
            st.success("✅ Message approved and ready for dispatch.")
            st.balloons()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_color(tag: str) -> str:
    return {
        "URGENT": "#dc3545",
        "REVIEW": "#ffc107",
        "MONITOR": "#0d6efd",
        "ON TRACK": "#198754",
    }.get(tag, "#6c757d")


# ---------------------------------------------------------------------------
# Main Router
# ---------------------------------------------------------------------------

if screen == "📊 Book Prioritizer":
    render_book_prioritizer()
elif screen == "🔍 Client Deep-Dive":
    render_client_deep_dive()
elif screen == "💡 Advisory Action Builder":
    render_advisory_builder()

