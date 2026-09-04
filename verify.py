"""Quick verification script — tests all analytics + scoring modules."""
import sys
sys.path.insert(0, ".")

from src.analytics import (
    load_all_data, compute_portfolio_summary, compute_ltv_analysis,
    compute_mandate_drift, compute_liquidity_coverage, compute_concentration_risk,
    attribute_portfolio_changes, check_sustainability_breaches,
)
from src.scoring import compute_priority_scores

print("Loading data...")
data = load_all_data()
print(f"  Clients: {len(data['clients'])}")
print(f"  Holdings: {len(data['holdings'])}")
print(f"  Events: {len(data['event_log'])}")

print("\n=== PORTFOLIO SUMMARY ===")
ps = compute_portfolio_summary(data)
for _, r in ps.sort_values("aum_usd_current", ascending=False).iterrows():
    print(f"  {r['client_id']}  {r['client_name']:<30s}  AUM: ${r['aum_usd_current']/1e6:>8.1f}M  YTD: {r['ytd_change_pct']:>+6.1f}%")

print("\n=== LTV ANALYSIS ===")
ltv = compute_ltv_analysis(data)
for _, r in ltv.iterrows():
    breach_flag = " *** BREACHED ***" if r["ever_breached"] else ""
    print(f"  {r['client_id']}  {r['client_name']:<30s}  LTV: {r['current_ltv']:>5.1f}%  Trigger: {r['margin_call_ltv_pct']:.0f}%  Headroom: {r['ltv_headroom_pct']:>+5.1f}%{breach_flag}")

print("\n=== MANDATE BREACHES ===")
drift = compute_mandate_drift(data)
breaches = drift[drift["breach"] != "None"] if not drift.empty else drift
if not breaches.empty:
    for _, r in breaches.iterrows():
        print(f"  {r['client_id']}  {r['portfolio_name']:<30s}  {r['asset_class']:<25s}  {r['actual_weight_pct']:>5.1f}% vs [{r['min_pct']}-{r['max_pct']}%]  {r['breach']} by {r['breach_amount_pct']:.1f}%")
else:
    print("  No breaches detected.")

print("\n=== LIQUIDITY COVERAGE ===")
liq = compute_liquidity_coverage(data)
for _, r in liq.sort_values("coverage_ratio").iterrows():
    flag = " *** STRESSED ***" if r["is_stressed"] else ""
    cov = f"{r['coverage_ratio']:.1f}x" if r['coverage_ratio'] < 100 else "N/A"
    print(f"  {r['client_id']}  {r['client_name']:<30s}  Daily Liquid: ${r['daily_liquid_usd']/1e6:>6.1f}M  Obligations: ${r['total_obligations_usd']/1e6:>6.1f}M  Coverage: {cov}{flag}")

print("\n=== SUSTAINABILITY BREACHES ===")
sus = check_sustainability_breaches(data)
if not sus.empty:
    for _, r in sus.iterrows():
        print(f"  {r['client_id']}  {r['instrument_name']:<40s}  Weight: {r['weight_pct']:.1f}%")
else:
    print("  No sustainability breaches.")

print("\n=== PRIORITY SCORES ===")
scores = compute_priority_scores(data)
for _, r in scores.iterrows():
    print(f"  {r['priority_score']:>3d}  [{r['action_tag']:<8s}]  {r['client_name']:<30s}  ${r['total_aum_usd']/1e6:>8.1f}M  {r['risk_flags']}")

print("\n=== EVENT ATTRIBUTION (CL-0012 sample) ===")
attr = attribute_portfolio_changes(data, "CL-0012")
for a in attr[:5]:
    print(f"  {a['instrument_name']:<40s}  USD {a['mv_change_usd']:>+12,.0f}  ({a['mv_change_pct']:>+5.1f}%)  [{a['period']}]")
    for e in a['events']:
        print(f"    -> {e['event_date']}: {e['description'][:80]}...")

print("\n=== ALL TESTS PASSED ===")

