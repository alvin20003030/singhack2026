"""Deterministic data-quality checks and bounded in-memory repair orchestration."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.analytics import SNAPSHOT_DATES


@dataclass
class DataFinding:
    finding_id: str
    severity: str
    code: str
    dataset: str
    row_index: Optional[int]
    field: Optional[str]
    message: str
    current_value: object = None
    expected_value: object = None
    repairable: bool = False


@dataclass
class RepairRecord:
    finding_id: str
    dataset: str
    row_index: int
    field: str
    before: object
    after: object
    rationale: str


def _finding(code, dataset, row_index, field, message, current=None,
             expected=None, repairable=False, severity="warning"):
    suffix = f"-{row_index}" if row_index is not None else ""
    return DataFinding(
        finding_id=f"{code}{suffix}", severity=severity, code=code,
        dataset=dataset, row_index=row_index, field=field,
        message=message, current_value=current, expected_value=expected,
        repairable=repairable,
    )


def validate_datasets(data: Dict[str, object]) -> List[DataFinding]:
    """Return logical inconsistencies without mutating the supplied datasets."""
    findings = []
    clients = data["clients"]
    portfolios = data["portfolios"]
    holdings = data["holdings"]
    instruments = data["instruments"]
    mandates = data["mandates"]
    facilities = data["credit_facilities"]

    client_ids = set(clients["client_id"].dropna())
    portfolio_ids = set(portfolios["portfolio_id"].dropna())
    instrument_ids = set(instruments["instrument_id"].dropna())
    mandate_codes = set(mandates["mandate_code"].dropna())

    for name, frame, column, allowed, code in [
        ("portfolios", portfolios, "client_id", client_ids, "UNKNOWN_CLIENT"),
        ("holdings", holdings, "client_id", client_ids, "UNKNOWN_CLIENT"),
        ("transactions", data["transactions"], "client_id", client_ids, "UNKNOWN_CLIENT"),
        ("commitments", data["commitments"], "client_id", client_ids, "UNKNOWN_CLIENT"),
        ("planned_cash_needs", data["planned_cash_needs"], "client_id", client_ids, "UNKNOWN_CLIENT"),
        ("credit_facilities", facilities, "client_id", client_ids, "UNKNOWN_CLIENT"),
    ]:
        for index, value in frame[column].items():
            if value not in allowed:
                findings.append(_finding(code, name, index, column,
                    f"{column} {value!r} does not reference a known client.", value))

    for index, row in holdings.iterrows():
        if row["portfolio_id"] not in portfolio_ids:
            findings.append(_finding("UNKNOWN_PORTFOLIO", "holdings", index, "portfolio_id",
                f"Holding references unknown portfolio {row['portfolio_id']!r}.", row["portfolio_id"]))
        if row["instrument_id"] not in instrument_ids:
            findings.append(_finding("UNKNOWN_INSTRUMENT", "holdings", index, "instrument_id",
                f"Holding references unknown instrument {row['instrument_id']!r}.", row["instrument_id"]))

    for index, row in portfolios.iterrows():
        if row["mandate_code"] not in mandate_codes:
            findings.append(_finding("UNKNOWN_MANDATE", "portfolios", index, "mandate_code",
                f"Portfolio references unknown mandate {row['mandate_code']!r}.", row["mandate_code"]))

    portfolio_clients = portfolios.set_index("portfolio_id")["client_id"].to_dict()
    for index, row in holdings.iterrows():
        owner = portfolio_clients.get(row["portfolio_id"])
        if owner is not None and row["client_id"] != owner:
            findings.append(_finding("HOLDING_OWNER_MISMATCH", "holdings", index, "client_id",
                "Holding client does not match the owning portfolio.", row["client_id"], owner,
                repairable=True, severity="error"))

    for index, row in facilities.iterrows():
        portfolio_id = row["collateral_portfolio_id"]
        if portfolio_id not in portfolio_ids:
            findings.append(_finding("UNKNOWN_COLLATERAL_PORTFOLIO", "credit_facilities", index,
                "collateral_portfolio_id", f"Collateral portfolio {portfolio_id!r} is unknown.", portfolio_id))

    for index, row in holdings.iterrows():
        date = str(row["snapshot_date"])
        if date not in SNAPSHOT_DATES:
            findings.append(_finding("INVALID_SNAPSHOT_DATE", "holdings", index, "snapshot_date",
                f"Snapshot date {date!r} is outside the documented snapshots.", date))

    for index, row in mandates.iterrows():
        values = [row["min_pct"], row["target_pct"], row["max_pct"]]
        if not (values[0] <= values[1] <= values[2]):
            findings.append(_finding("INVALID_MANDATE_BANDS", "mandates", index, None,
                "Mandate bands must satisfy min <= target <= max.", values))

    for index, row in holdings.iterrows():
        expected = float(row["quantity"]) * float(row["price_local"])
        actual = float(row["market_value_local"])
        if abs(actual - expected) > max(0.01, abs(expected) * 0.0001):
            findings.append(_finding("MARKET_VALUE_MISMATCH", "holdings", index, "market_value_local",
                "Market value does not reconcile to quantity multiplied by local price.", actual,
                expected, repairable=True, severity="error"))

    for (portfolio_id, snapshot), group in holdings.groupby(["portfolio_id", "snapshot_date"]):
        total = float(group["weight_pct"].sum())
        if abs(total - 100.0) > 0.01:
            index = group.index[0]
            findings.append(_finding("WEIGHT_TOTAL_MISMATCH", "holdings", index, "weight_pct",
                f"Weights for {portfolio_id} at {snapshot} sum to {total:.4f}% rather than 100%.",
                total, 100.0))

    for index, row in facilities.iterrows():
        for snapshot in SNAPSHOT_DATES:
            drawn = float(row[f"drawn_{snapshot}"])
            lending = float(row[f"lending_value_{snapshot}"])
            ltv = float(row[f"ltv_pct_{snapshot}"])
            headroom = float(row[f"headroom_{snapshot}"])
            expected_ltv = (drawn / lending * 100.0) if lending else 0.0
            expected_headroom = lending - drawn
            if abs(ltv - expected_ltv) > 0.01:
                findings.append(_finding("LTV_MISMATCH", "credit_facilities", index,
                    f"ltv_pct_{snapshot}", f"LTV does not reconcile for {snapshot}.", ltv,
                    expected_ltv, repairable=True, severity="error"))
            if abs(headroom - expected_headroom) > 0.01:
                findings.append(_finding("HEADROOM_MISMATCH", "credit_facilities", index,
                    f"headroom_{snapshot}", f"Headroom does not reconcile for {snapshot}.", headroom,
                    expected_headroom, repairable=True, severity="error"))

    for index, row in facilities.iterrows():
        if row["collateral_portfolio_id"] in portfolio_ids:
            owner = portfolio_clients[row["collateral_portfolio_id"]]
            if row["client_id"] != owner:
                findings.append(_finding("FACILITY_OWNER_MISMATCH", "credit_facilities", index,
                    "client_id", "Facility client does not own its collateral portfolio.", row["client_id"], owner))

    notes = data.get("rm_notes", [])
    for index, note in enumerate(notes):
        if not isinstance(note, dict) or not {"note_id", "client_id", "note_date", "note"}.issubset(note):
            findings.append(_finding("INVALID_RM_NOTE", "rm_notes", index, None,
                "RM note is missing one or more documented fields."))
        elif note["client_id"] not in client_ids:
            findings.append(_finding("UNKNOWN_NOTE_CLIENT", "rm_notes", index, "client_id",
                f"RM note references unknown client {note['client_id']!r}.", note["client_id"]))

    return findings


def _default_repair_suggester(findings, _data):
    """Use deterministic expected values when no AI adapter is configured."""
    return [
        {
            "finding_id": finding["finding_id"],
            "dataset": finding["dataset"],
            "row_index": finding["row_index"],
            "field": finding["field"],
            "value": finding["expected_value"],
            "rationale": "Deterministic reconciliation from the documented data contract.",
        }
        for finding in findings
        if finding.get("repairable") and finding.get("row_index") is not None and finding.get("field")
    ]


def _apply_repairs(data, findings, proposals):
    by_id = {finding.finding_id: finding for finding in findings}
    repairs = []
    for proposal in proposals or []:
        finding = by_id.get(proposal.get("finding_id"))
        if finding is None or not finding.repairable:
            continue
        if proposal.get("dataset") != finding.dataset or proposal.get("field") != finding.field:
            continue
        if proposal.get("row_index") != finding.row_index:
            continue
        after = proposal.get("value", finding.expected_value)
        if finding.expected_value is not None:
            try:
                if abs(float(after) - float(finding.expected_value)) > max(0.01, abs(float(finding.expected_value)) * 0.0001):
                    continue
            except (TypeError, ValueError):
                if after != finding.expected_value:
                    continue
        frame = data.get(finding.dataset)
        if not isinstance(frame, pd.DataFrame) or finding.field not in frame.columns:
            continue
        before = frame.at[finding.row_index, finding.field]
        if before == after:
            continue
        frame.at[finding.row_index, finding.field] = after
        repairs.append(RepairRecord(
            finding_id=finding.finding_id, dataset=finding.dataset,
            row_index=finding.row_index, field=finding.field, before=before,
            after=after, rationale=proposal.get("rationale", finding.message),
        ))
    return repairs


def validate_and_repair(data: Dict[str, object], repair_suggester: Optional[Callable] = None,
                        max_iterations: int = 3) -> Dict[str, object]:
    """Validate copied data and apply only verified, bounded in-memory repairs."""
    repaired_data = {key: value.copy(deep=True) if isinstance(value, pd.DataFrame) else deepcopy(value)
                     for key, value in data.items()}
    all_repairs = []
    iterations = []
    suggest = repair_suggester or _default_repair_suggester

    for iteration in range(1, max_iterations + 1):
        findings = validate_datasets(repaired_data)
        repairable = [finding for finding in findings if finding.repairable]
        if not repairable:
            iterations.append({"iteration": iteration, "finding_count": len(findings), "repair_count": 0})
            break
        proposals = suggest([asdict(finding) for finding in repairable], repaired_data)
        repairs = _apply_repairs(repaired_data, repairable, proposals)
        all_repairs.extend(repairs)
        iterations.append({"iteration": iteration, "finding_count": len(findings), "repair_count": len(repairs)})
        if not repairs:
            break

    final_findings = validate_datasets(repaired_data)
    return {
        "data": repaired_data,
        "findings": final_findings,
        "repairs": [asdict(repair) for repair in all_repairs],
        "iterations": iterations,
        "clean": not final_findings,
        "source_unchanged": True,
    }