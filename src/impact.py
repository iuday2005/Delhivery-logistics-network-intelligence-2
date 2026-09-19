"""Business impact data and filtering helpers for the dashboard."""

from pathlib import Path

import pandas as pd


BUSINESS_IMPACT = {
    "revenue_at_risk": 2116800,
    "potential_recovery": 395985,
    "monthly_recovery_estimate": 395985,
    "annualized_recovery_estimate": 4751820,
    "best_hub": "IND421302AAG",
    "best_recovery": 454385,
    "best_breaches_avoided": 3029,
}

FALLBACK_HUB_IMPACT = pd.DataFrame(
    [
        ["IND421302AAG", 30, 1281, 29, 10.15, 3029, 454385, 913.07],
        ["IND000000ACB", 30, 1848, 49, 10.37, 2897, 434575, 873.40],
        ["IND562132AAA", 30, 1366, 36, 9.04, 1771, 265670, 534.48],
        ["IND400072AAJ", 30, 53, 1, 32.40, 1240, 186030, 379.47],
        ["IND501359AAE", 30, 660, 27, 9.37, 981, 147097, 296.80],
    ],
    columns=[
        "hub",
        "intervention_pct",
        "trip_volume",
        "affected_corridors",
        "eta_improvement_min",
        "sla_breaches_avoided",
        "revenue_recovered_inr",
        "roi_score",
    ],
)

FALLBACK_CORRIDOR_RISK = pd.DataFrame(
    [
        ["IND743270AAA", "IND712311AAA", "FTL", 15.35, 1.00, 5, 51.11, "Critical"],
        ["IND741201AAC", "IND712311AAA", "FTL", 15.27, 1.00, 4, 50.91, "Critical"],
        ["IND844505AAB", "IND842001AAA", "FTL", 11.36, 0.94, 14, 40.22, "Critical"],
        ["IND751002AAB", "IND754103AAA", "FTL", 10.25, 1.00, 7, 39.58, "Critical"],
        ["IND722151AAA", "IND723130AAA", "FTL", 14.21, 0.70, 8, 38.55, "Critical"],
    ],
    columns=[
        "source_center",
        "destination_center",
        "route_type",
        "median_delay_ratio",
        "sla_severe_rate",
        "trip_count",
        "risk_score",
        "risk_category",
    ],
)

EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "reports" / "evidence"

HUB_SIMULATOR_ASSUMPTIONS = pd.DataFrame(
    [
        ["sample_trips", "14804", "observed", "Cleaned trip count used in modeling and simulation."],
        ["revenue_at_risk_inr", "2116800", "derived", "Phase 5 annualized severe-delay exposure."],
        ["potential_recovery_inr", "395985", "derived", "Phase 5 top-hub recoverable value estimate."],
        ["avg_shipment_value_inr", "2000", "assumed", "Phase 5 conservative shipment value assumption."],
        ["sla_penalty_rate", "0.05", "assumed", "Phase 5 penalty rate per SLA breach."],
        ["canonical_intervention_pct", "0.30", "assumed", "Primary dashboard scenario: 30% hub delay reduction."],
    ],
    columns=["assumption", "value", "evidence_type", "notes"],
)

IMPACT_CLAIM_RECONCILIATION = pd.DataFrame(
    [
        [
            "Estimated revenue at risk from severe delays",
            "INR 2116800",
            "phase5_checkpoint",
            "canonical",
            "Derived from severe delayed trips, shipment value, penalty rate, and annual multiplier assumptions.",
        ],
        [
            "Potential recovery from targeted interventions",
            "INR 395985",
            "phase5_checkpoint",
            "canonical",
            "Use as conservative portfolio headline; depends on top-hub contribution and recovery rate.",
        ],
    ],
    columns=["claim", "value", "source", "status", "defensibility_note"],
)


def format_inr(value: float) -> str:
    """Format Indian Rupee values in a compact readable form."""
    if value >= 100000:
        return f"INR {value / 100000:.2f} lakh"
    return f"INR {value:,.0f}"


def get_hub_impact_table(artifacts: dict) -> pd.DataFrame:
    """Use simulation artifacts when available, otherwise use static fallback."""
    hub_results = artifacts.get("Hub simulation results")
    if isinstance(hub_results, dict) and isinstance(hub_results.get("best_interventions"), pd.DataFrame):
        table = hub_results["best_interventions"].copy()
        table["intervention_pct"] = (table["intervention_pct"] * 100).round(0).astype(int)
        return table
    return FALLBACK_HUB_IMPACT.copy()


def get_hub_scenario_table(artifacts: dict) -> pd.DataFrame:
    """Return all intervention scenarios when available."""
    hub_results = artifacts.get("Hub simulation results")
    if isinstance(hub_results, dict) and isinstance(hub_results.get("sim_df"), pd.DataFrame):
        table = hub_results["sim_df"].copy()
        table["intervention_pct"] = (table["intervention_pct"] * 100).round(0).astype(int)
        return table
    return FALLBACK_HUB_IMPACT.copy()


def get_corridor_risk_table(artifacts: dict, top_n: int = 20) -> pd.DataFrame:
    """Use ranked corridor artifacts when available, otherwise use static fallback."""
    risk_results = artifacts.get("Corridor risk results")
    if isinstance(risk_results, dict) and isinstance(risk_results.get("top20"), pd.DataFrame):
        return risk_results["top20"].head(top_n).copy()
    return FALLBACK_CORRIDOR_RISK.copy()


def get_full_corridor_risk_table(artifacts: dict) -> pd.DataFrame:
    """Return the full corridor risk table when available."""
    risk_results = artifacts.get("Corridor risk results")
    if isinstance(risk_results, dict) and isinstance(risk_results.get("corridor_stats_risk"), pd.DataFrame):
        return risk_results["corridor_stats_risk"].copy()
    return FALLBACK_CORRIDOR_RISK.copy()


def read_evidence_csv(file_name: str, fallback: pd.DataFrame) -> pd.DataFrame:
    """Read a lightweight evidence CSV when present."""
    path = EVIDENCE_DIR / file_name
    if path.exists():
        return pd.read_csv(path)
    return fallback.copy()


def get_hub_assumptions() -> pd.DataFrame:
    """Return hub simulator assumptions and evidence type labels."""
    return read_evidence_csv("hub_simulator_assumptions.csv", HUB_SIMULATOR_ASSUMPTIONS)


def get_impact_reconciliation() -> pd.DataFrame:
    """Return reconciled business impact claims."""
    return read_evidence_csv("impact_claim_reconciliation.csv", IMPACT_CLAIM_RECONCILIATION)


def apply_hub_filters(table: pd.DataFrame, selected_hubs: list[str], selected_pct: list[int]) -> pd.DataFrame:
    """Filter hub intervention rows by selected hub and intervention scenario."""
    filtered = table.copy()
    if not selected_hubs or not selected_pct:
        return filtered.iloc[0:0]
    filtered = filtered[filtered["hub"].isin(selected_hubs)]
    filtered = filtered[filtered["intervention_pct"].isin(selected_pct)]
    return filtered


def apply_corridor_filters(
    table: pd.DataFrame,
    route_types: list[str],
    risk_categories: list[str],
    min_trips: int,
) -> pd.DataFrame:
    """Filter corridor rankings by route, risk category, and minimum trip volume."""
    filtered = table.copy()
    if not route_types or not risk_categories:
        return filtered.iloc[0:0]
    if "route_type" in filtered.columns:
        filtered = filtered[filtered["route_type"].isin(route_types)]
    if "risk_category" in filtered.columns:
        filtered = filtered[filtered["risk_category"].isin(risk_categories)]
    if "trip_count" in filtered.columns:
        filtered = filtered[filtered["trip_count"] >= min_trips]
    return filtered
