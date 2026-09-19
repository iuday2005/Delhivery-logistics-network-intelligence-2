"""Build lightweight evidence tables for the Delhivery portfolio project.

The raw CSV and large pickle checkpoints are intentionally kept out of GitHub.
This script lets the local analysis artifacts generate small CSV files that
make the dashboard and README claims auditable.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
EVIDENCE_DIR = BASE_DIR / "reports" / "evidence"
RANDOM_STATE = 42


def load_pickle(name: str):
    with (ARTIFACTS_DIR / name).open("rb") as file:
        return pickle.load(file)


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, notes: str) -> dict:
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    return {
        "model": name,
        "mae_min": mean_absolute_error(y_true, y_pred),
        "rmse_min": float(np.sqrt(np.mean(errors**2))),
        "r2": r2_score(y_true, y_pred),
        "within_15_pct": float(np.mean(abs_errors / np.maximum(np.abs(y_true), 1e-8) < 0.15) * 100),
        "bias_min": float(np.mean(y_pred - y_true)),
        "cv_mae_mean_min": np.nan,
        "cv_mae_std_min": np.nan,
        "notes": notes,
    }


def graph_feature_matrix(trip_agg: pd.DataFrame, p4: dict) -> tuple[np.ndarray, list[str]]:
    baseline_features = p4["feat_names_e"][:22]
    graph_features = p4["graph_features"]
    embeddings = p4["embeddings"]
    emb_dim = len(next(iter(embeddings.values())))

    src_embs = np.vstack(
        [
            embeddings.get(str(center), np.zeros(emb_dim))
            for center in trip_agg["source_center"].astype(str)
        ]
    )
    dst_embs = np.vstack(
        [
            embeddings.get(str(center), np.zeros(emb_dim))
            for center in trip_agg["destination_center"].astype(str)
        ]
    )
    x_base = trip_agg[baseline_features].fillna(0).values
    x_graph = trip_agg[graph_features].fillna(0).values
    feature_names = (
        baseline_features
        + graph_features
        + [f"src_emb_{idx}" for idx in range(emb_dim)]
        + [f"dst_emb_{idx}" for idx in range(emb_dim)]
    )
    return np.concatenate([x_base, x_graph, src_embs, dst_embs], axis=1), feature_names


def build_canonical_model_validation(p3: dict, p4: dict) -> pd.DataFrame:
    results = p4["results_df"].copy()
    canonical = []
    canonical.append(
        {
            "model": "OSRM baseline",
            "mae_min": 161.50,
            "rmse_min": np.nan,
            "r2": np.nan,
            "within_15_pct": 5.50,
            "bias_min": np.nan,
            "cv_mae_mean_min": np.nan,
            "cv_mae_std_min": np.nan,
            "notes": "Routing estimate baseline from phase 3/5 notebook outputs.",
        }
    )
    for _, row in results.iterrows():
        canonical.append(
            {
                "model": row["name"].replace(" — reproduce Phase 3", ""),
                "mae_min": row["mae"],
                "rmse_min": row["rmse"],
                "r2": row["r2"],
                "within_15_pct": row["within_15"] * 100,
                "bias_min": row["bias"],
                "cv_mae_mean_min": np.nan,
                "cv_mae_std_min": np.nan,
                "notes": "Same validation split; route_type-stratified trip-level holdout.",
            }
        )

    canonical.append(
        {
            "model": "5-fold CV on best graph model",
            "mae_min": np.nan,
            "rmse_min": np.nan,
            "r2": np.nan,
            "within_15_pct": np.nan,
            "bias_min": np.nan,
            "cv_mae_mean_min": 28.78,
            "cv_mae_std_min": 0.51,
            "notes": "Canonical notebook output from phase4.ipynb; exported here to avoid rerunning expensive CV during dashboard evidence generation.",
        }
    )
    return pd.DataFrame(canonical).round(4)


def build_graph_validation_tests(p3: dict, p4: dict) -> pd.DataFrame:
    trip_agg = p4["trip_agg_p4"]
    x_graph, graph_feature_names = graph_feature_matrix(trip_agg, p4)
    y = trip_agg["total_actual_time"].values
    route_type = trip_agg["route_type"].values
    x_base = trip_agg[p3["baseline_features"]].fillna(0).values
    (
        x_base_train,
        x_base_test,
        x_graph_train,
        x_graph_test,
        y_train,
        y_test,
        _,
        _,
    ) = train_test_split(
        x_base,
        x_graph,
        y,
        route_type,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=route_type,
    )

    baseline_model = p3["best_rf"]
    graph_model = p4["best_rf"]
    x_base_test_df = pd.DataFrame(x_base_test, columns=p3["baseline_features"])
    x_graph_test_df = pd.DataFrame(x_graph_test, columns=graph_feature_names)
    baseline_errors = np.abs(y_test - baseline_model.predict(x_base_test_df))
    graph_errors = np.abs(y_test - graph_model.predict(x_graph_test_df))
    stat, p_value = stats.wilcoxon(baseline_errors, graph_errors)

    return pd.DataFrame(
        [
            {
                "test_name": "Wilcoxon signed-rank test",
                "comparison": "RF baseline vs graph-enhanced RF",
                "sample_size": len(y_test),
                "statistic": stat,
                "p_value": p_value,
                "alpha": 0.05,
                "conclusion": (
                    "Graph model has statistically different paired absolute errors on the same validation trips."
                    if p_value < 0.05
                    else "No statistically significant paired error difference at alpha=0.05."
                ),
                "evidence_scope": "paired absolute errors on route_type-stratified trip-level validation split",
            }
        ]
    ).round(6)


def bucketize_hour(hour: int) -> str:
    if 0 <= hour <= 5:
        return "night_00_05"
    if 6 <= hour <= 11:
        return "morning_06_11"
    if 12 <= hour <= 17:
        return "afternoon_12_17"
    return "evening_18_23"


def summarize_errors(frame: pd.DataFrame, group: str, column: str) -> list[dict]:
    rows = []
    for value, part in frame.groupby(column, dropna=False):
        if len(part) < 5:
            continue
        rows.append(
            {
                "segment_group": group,
                "segment_value": str(value),
                "n_trips": int(len(part)),
                "mae_min": part["abs_error_min"].mean(),
                "median_abs_error_min": part["abs_error_min"].median(),
                "p90_abs_error_min": part["abs_error_min"].quantile(0.90),
                "bias_min": part["error_min"].mean(),
                "actionable_readout": make_error_readout(group, str(value), part),
            }
        )
    return rows


def make_error_readout(group: str, value: str, part: pd.DataFrame) -> str:
    mae = part["abs_error_min"].mean()
    bias = part["error_min"].mean()
    direction = "under-predicting" if bias > 0 else "over-predicting"
    return f"{group}={value}: MAE {mae:.1f} min; model is {direction} by {abs(bias):.1f} min on average."


def build_segment_error_analysis(p3: dict, p4: dict, risk: dict) -> pd.DataFrame:
    trip_agg = p4["trip_agg_p4"].copy()
    x_graph, graph_feature_names = graph_feature_matrix(trip_agg, p4)
    y = trip_agg["total_actual_time"].values
    route_type = trip_agg["route_type"].values
    _, x_test, _, y_test, _, rt_test = train_test_split(
        x_graph,
        y,
        route_type,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=route_type,
    )
    _, trip_test = train_test_split(
        trip_agg,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=route_type,
    )

    predictions = p4["best_rf"].predict(pd.DataFrame(x_test, columns=graph_feature_names))
    eval_df = trip_test.copy().reset_index(drop=True)
    eval_df["actual_min"] = y_test
    eval_df["predicted_min"] = predictions
    eval_df["error_min"] = eval_df["actual_min"] - eval_df["predicted_min"]
    eval_df["abs_error_min"] = eval_df["error_min"].abs()
    eval_df["time_band"] = eval_df["creation_hour"].astype(int).map(bucketize_hour)
    eval_df["trip_complexity"] = pd.qcut(
        eval_df["total_segments"],
        q=3,
        labels=["low_segment_count", "medium_segment_count", "high_segment_count"],
        duplicates="drop",
    )
    eval_df["delay_bucket"] = np.select(
        [
            eval_df["is_trip_extreme"].astype(bool),
            eval_df["is_trip_severe"].astype(bool),
        ],
        ["extreme_delay", "severe_delay"],
        default="normal_or_moderate_delay",
    )
    eval_df["lane_key"] = (
        eval_df["source_center"].astype(str)
        + " -> "
        + eval_df["destination_center"].astype(str)
        + " | "
        + eval_df["route_type"].astype(str)
    )

    risk_df = risk["corridor_stats_risk"].copy()
    risk_df["lane_key"] = (
        risk_df["source_center"].astype(str)
        + " -> "
        + risk_df["destination_center"].astype(str)
        + " | "
        + risk_df["route_type"].astype(str)
    )
    eval_df = eval_df.merge(
        risk_df[["lane_key", "risk_category"]],
        on="lane_key",
        how="left",
    )
    eval_df["risk_category"] = eval_df["risk_category"].astype("object").fillna("Unranked")

    rows: list[dict] = []
    for group, column in [
        ("route_type", "route_type"),
        ("corridor_risk_bucket", "risk_category"),
        ("trip_complexity", "trip_complexity"),
        ("time_band", "time_band"),
        ("delay_bucket", "delay_bucket"),
        ("source_hub", "source_center"),
    ]:
        group_rows = summarize_errors(eval_df, group, column)
        if group == "source_hub":
            group_rows = sorted(group_rows, key=lambda row: row["mae_min"], reverse=True)[:10]
        rows.extend(group_rows)

    out = pd.DataFrame(rows)
    return out.sort_values(["segment_group", "mae_min"], ascending=[True, False]).round(3)


def build_leakage_safe_graph_validation(p3: dict) -> pd.DataFrame:
    trip_agg = p3["trip_agg"].copy()
    baseline_features = p3["baseline_features"]
    y = trip_agg["total_actual_time"].values
    route_type = trip_agg["route_type"].values
    train_df, test_df = train_test_split(
        trip_agg,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=route_type,
    )

    train_corridors = (
        train_df.groupby(["source_center", "destination_center", "route_type"])
        .agg(
            trip_count=("trip_uuid", "count"),
            median_delay_ratio=("trip_delay_ratio", "median"),
            severe_rate=("is_trip_severe", "mean"),
        )
        .reset_index()
    )
    graph = nx.DiGraph()
    for _, row in train_corridors.iterrows():
        graph.add_edge(
            row["source_center"],
            row["destination_center"],
            weight=float(row["trip_count"]),
        )
    if graph.number_of_nodes() == 0:
        raise RuntimeError("Training graph has no nodes; cannot build leakage-safe validation.")

    betweenness = nx.betweenness_centrality(graph, k=min(100, graph.number_of_nodes()), seed=RANDOM_STATE)
    pagerank = nx.pagerank(graph, weight="weight")
    out_degree = dict(graph.out_degree(weight="weight"))
    in_degree = dict(graph.in_degree(weight="weight"))

    hub_stats = (
        train_corridors.groupby("source_center")
        .agg(
            train_src_severe_rate=("severe_rate", "mean"),
            train_src_trip_volume=("trip_count", "sum"),
        )
        .rename_axis("facility")
        .reset_index()
    )
    hub_stats["train_src_betweenness"] = hub_stats["facility"].map(betweenness).fillna(0)
    hub_stats["train_src_pagerank"] = hub_stats["facility"].map(pagerank).fillna(0)
    hub_stats["train_src_out_degree"] = hub_stats["facility"].map(out_degree).fillna(0)
    hub_stats = hub_stats.set_index("facility")

    def add_train_graph_features(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for prefix, center_col in [("src", "source_center"), ("dst", "destination_center")]:
            centers = out[center_col].astype(str)
            out[f"train_{prefix}_betweenness"] = centers.map(betweenness).fillna(0)
            out[f"train_{prefix}_pagerank"] = centers.map(pagerank).fillna(0)
            out[f"train_{prefix}_out_degree"] = centers.map(out_degree).fillna(0)
            out[f"train_{prefix}_in_degree"] = centers.map(in_degree).fillna(0)
            out[f"train_{prefix}_severe_rate"] = centers.map(hub_stats["train_src_severe_rate"]).fillna(
                hub_stats["train_src_severe_rate"].median()
            )
            out[f"train_{prefix}_trip_volume"] = centers.map(hub_stats["train_src_trip_volume"]).fillna(0)
        out["train_combined_bottleneck"] = (
            out["train_src_betweenness"]
            + out["train_dst_betweenness"]
            + out["train_src_pagerank"]
            + out["train_dst_pagerank"]
        )
        out["train_bottleneck_x_segments"] = out["train_combined_bottleneck"] * out["total_segments"]
        return out

    train_scored = add_train_graph_features(train_df)
    test_scored = add_train_graph_features(test_df)
    leakage_features = [
        col
        for col in train_scored.columns
        if col.startswith("train_") and col not in {"train_src_trip_volume", "train_dst_trip_volume"}
    ] + ["train_src_trip_volume", "train_dst_trip_volume"]

    baseline_model = RandomForestRegressor(
        n_estimators=80,
        max_depth=None,
        min_samples_leaf=3,
        min_samples_split=5,
        max_features=0.6,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    graph_model = RandomForestRegressor(
        n_estimators=80,
        max_depth=None,
        min_samples_leaf=3,
        min_samples_split=5,
        max_features=0.6,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    x_base_train = train_scored[baseline_features].fillna(0)
    x_base_test = test_scored[baseline_features].fillna(0)
    x_graph_train = train_scored[baseline_features + leakage_features].fillna(0)
    x_graph_test = test_scored[baseline_features + leakage_features].fillna(0)
    y_train = train_scored["total_actual_time"].values
    y_test = test_scored["total_actual_time"].values

    baseline_model.fit(x_base_train, y_train)
    graph_model.fit(x_graph_train, y_train)
    base_pred = baseline_model.predict(x_base_test)
    graph_pred = graph_model.predict(x_graph_test)

    rows = [
        evaluate(
            "RF baseline retrained for leakage-safe comparison",
            y_test,
            base_pred,
            "Trip split first; no graph features.",
        ),
        evaluate(
            "Leakage-safe graph centrality RF",
            y_test,
            graph_pred,
            "Graph rebuilt from training trips only; validation hubs mapped to train-window graph features.",
        ),
    ]
    stat, p_value = stats.wilcoxon(np.abs(y_test - base_pred), np.abs(y_test - graph_pred))
    rows[1]["wilcoxon_statistic_vs_retrained_baseline"] = stat
    rows[1]["wilcoxon_p_value_vs_retrained_baseline"] = p_value
    rows[1]["mae_lift_min_vs_retrained_baseline"] = rows[0]["mae_min"] - rows[1]["mae_min"]
    out = pd.DataFrame(rows)
    return out.round(6)


def build_hub_assumptions(p5: dict, hub: dict) -> pd.DataFrame:
    sim_df = hub["sim_df"]
    best30 = sim_df[sim_df["intervention_pct"] == 0.30]
    return pd.DataFrame(
        [
            ["sample_trips", "14804", "observed", "Cleaned trip count used in modeling and simulation."],
            ["revenue_at_risk_inr", f"{p5['revenue_at_risk']:.0f}", "derived", "Phase 5 annualized severe-delay exposure."],
            ["potential_recovery_inr", f"{p5['potential_recovery']:.0f}", "derived", "Phase 5 top-hub recoverable value estimate."],
            ["avg_shipment_value_inr", "2000", "assumed", "Phase 5 conservative shipment value assumption."],
            ["sla_penalty_rate", "0.05", "assumed", "Phase 5 penalty rate per SLA breach."],
            ["phase5_annual_multiplier", "12", "assumed", "Phase 5 sample-to-annual multiplier."],
            ["hub_sim_annual_multiplier", "52", "assumed", "Hub simulator weekly-sample annualization assumption."],
            ["hub_sim_revenue_per_breach_inr", "150", "assumed", "Equivalent penalty used in hub simulator recovery calculation."],
            ["canonical_intervention_pct", "0.30", "assumed", "Primary dashboard scenario: 30% hub delay reduction."],
            ["best_hub_by_recovery", str(best30.sort_values("revenue_recovered_inr", ascending=False).iloc[0]["hub"]), "derived", "Best 30% intervention scenario in hub simulator."],
            ["best_hub_recovery_inr", f"{best30['revenue_recovered_inr'].max():.0f}", "derived", "Recovery from the highest-value single hub scenario."],
            ["top5_hub_recovery_inr", f"{best30['revenue_recovered_inr'].sum():.0f}", "derived", "Total recovery from all simulated top hubs at 30% delay reduction."],
            ["intervention_independence", "true", "assumed", "Hub simulator treats interventions independently; no interaction effects modeled."],
        ],
        columns=["assumption", "value", "evidence_type", "notes"],
    )


def build_impact_reconciliation(p5: dict, hub: dict) -> pd.DataFrame:
    sim30 = hub["sim_df"][hub["sim_df"]["intervention_pct"] == 0.30]
    best30 = sim30.sort_values("revenue_recovered_inr", ascending=False).iloc[0]
    return pd.DataFrame(
        [
            {
                "claim": "Estimated revenue at risk from severe delays",
                "value": f"INR {p5['revenue_at_risk']:.0f}",
                "source": "phase5_checkpoint",
                "status": "canonical",
                "defensibility_note": "Derived from severe delayed trips, shipment value, penalty rate, and annual multiplier assumptions.",
            },
            {
                "claim": "Potential recovery from targeted interventions",
                "value": f"INR {p5['potential_recovery']:.0f}",
                "source": "phase5_checkpoint",
                "status": "canonical",
                "defensibility_note": "Use as conservative portfolio headline; depends on top-hub contribution and recovery rate.",
            },
            {
                "claim": "Best single hub intervention recovery",
                "value": f"INR {best30['revenue_recovered_inr']:.0f}",
                "source": "hub_simulation_results",
                "status": "scenario",
                "defensibility_note": "30% delay-reduction scenario for one hub; keep separate from phase 5 conservative recovery.",
            },
            {
                "claim": "Top simulated hubs recovery at 30% delay reduction",
                "value": f"INR {sim30['revenue_recovered_inr'].sum():.0f}",
                "source": "hub_simulation_results",
                "status": "scenario",
                "defensibility_note": "Scenario sum across simulated hubs; do not present as guaranteed or realized recovery.",
            },
            {
                "claim": "SQL top-20 lane recovery if fixed",
                "value": "INR 697515",
                "source": "sql/product_analytics_layer/exports/top20_lane_recovery_summary.tsv",
                "status": "supporting",
                "defensibility_note": "SQL lane-backlog estimate uses INR 150 per breach and 70% fix effectiveness.",
            },
        ]
    )


def main() -> None:
    warnings.filterwarnings("ignore", message="X has feature names.*")
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    p3 = load_pickle("phase3_checkpoint.pkl")
    p4 = load_pickle("phase4_results.pkl")
    risk = load_pickle("corridor_risk_results.pkl")
    hub = load_pickle("hub_simulation_results.pkl")
    p5 = load_pickle("phase5_checkpoint.pkl")

    outputs = {
        "canonical_model_validation.csv": build_canonical_model_validation(p3, p4),
        "graph_validation_tests.csv": build_graph_validation_tests(p3, p4),
        "segment_error_analysis.csv": build_segment_error_analysis(p3, p4, risk),
        "leakage_safe_graph_validation.csv": build_leakage_safe_graph_validation(p3),
        "hub_simulator_assumptions.csv": build_hub_assumptions(p5, hub),
        "impact_claim_reconciliation.csv": build_impact_reconciliation(p5, hub),
    }
    for name, frame in outputs.items():
        frame.to_csv(EVIDENCE_DIR / name, index=False)
        print(f"wrote {EVIDENCE_DIR / name} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
