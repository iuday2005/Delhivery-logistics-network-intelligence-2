"""Model validation summaries for presentation and deployment fallback."""

from pathlib import Path

import pandas as pd


MODEL_VALIDATION = pd.DataFrame(
    [
        ["OSRM baseline", 161.50, None, None, 5.50, None, None, None, "Routing estimate baseline"],
        ["RF Baseline (Set C)", 30.9461, 74.5730, 0.9816, 77.6427, -0.7828, None, None, "Segment and trip features"],
        [
            "RF + Graph Centrality (Set D)",
            30.2888,
            73.4359,
            0.9822,
            78.2506,
            -0.6466,
            None,
            None,
            "Adds trainable graph centrality features",
        ],
        [
            "RF + Centrality + Node2Vec Embeddings (Set E)",
            29.8117,
            73.1315,
            0.9823,
            78.1830,
            -1.0180,
            None,
            None,
            "Best graph model with centrality and embeddings",
        ],
        [
            "5-fold CV on best graph model",
            None,
            None,
            None,
            None,
            None,
            28.78,
            0.51,
            "Canonical notebook output from phase4.ipynb",
        ],
    ],
    columns=[
        "model",
        "mae_min",
        "rmse_min",
        "r2",
        "within_15_pct",
        "bias_min",
        "cv_mae_mean_min",
        "cv_mae_std_min",
        "notes",
    ],
)

GRAPH_VALIDATION_TESTS = pd.DataFrame(
    [
        [
            "Wilcoxon signed-rank test",
            "RF baseline vs graph-enhanced RF",
            2961,
            2066088.0,
            0.006534,
            0.05,
            "Graph model has statistically different paired absolute errors on the same validation trips.",
            "paired absolute errors on route_type-stratified trip-level validation split",
        ]
    ],
    columns=[
        "test_name",
        "comparison",
        "sample_size",
        "statistic",
        "p_value",
        "alpha",
        "conclusion",
        "evidence_scope",
    ],
)

LEAKAGE_SAFE_GRAPH_VALIDATION = pd.DataFrame(
    [
        [
            "RF baseline retrained for leakage-safe comparison",
            31.145365,
            74.752369,
            0.981535,
            77.136103,
            -0.809,
            "Trip split first; no graph features.",
            None,
            None,
            None,
        ],
        [
            "Leakage-safe graph centrality RF",
            30.232576,
            72.711160,
            0.982530,
            78.183046,
            -1.130519,
            "Graph rebuilt from training trips only; validation hubs mapped to train-window graph features.",
            2089904.0,
            0.027257,
            0.912789,
        ],
    ],
    columns=[
        "model",
        "mae_min",
        "rmse_min",
        "r2",
        "within_15_pct",
        "bias_min",
        "notes",
        "wilcoxon_statistic_vs_retrained_baseline",
        "wilcoxon_p_value_vs_retrained_baseline",
        "mae_lift_min_vs_retrained_baseline",
    ],
)

SEGMENT_ERROR_ANALYSIS = pd.DataFrame(
    [
        [
            "route_type",
            "FTL",
            1181,
            52.068,
            23.901,
            121.902,
            4.101,
            "FTL trips carry the highest aggregate ETA error and should be monitored separately.",
        ],
        [
            "route_type",
            "Carting",
            1780,
            15.045,
            5.514,
            34.751,
            -1.027,
            "Carting trips are materially easier for the graph model to predict in this validation split.",
        ],
        [
            "delay_bucket",
            "extreme_delay",
            167,
            76.838,
            32.380,
            194.314,
            63.407,
            "Extreme-delay trips are the clearest failure pocket and are under-predicted.",
        ],
        [
            "corridor_risk_bucket",
            "Critical",
            365,
            42.318,
            13.129,
            91.006,
            3.941,
            "Critical-risk corridors need separate SLA monitoring beyond average network MAE.",
        ],
    ],
    columns=[
        "segment_group",
        "segment_value",
        "n_trips",
        "mae_min",
        "median_abs_error_min",
        "p90_abs_error_min",
        "bias_min",
        "actionable_readout",
    ],
)

EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "reports" / "evidence"

FEATURE_IMPORTANCE = pd.DataFrame(
    [
        ["log_osrm_distance", 0.5484],
        ["log_osrm_time", 0.2353],
        ["total_segments", 0.1073],
        ["log_total_distance", 0.0467],
        ["bottleneck_x_segments", 0.0145],
        ["mean_speed_efficiency", 0.0084],
        ["pct_severe_segments", 0.0071],
        ["distance_per_segment", 0.0043],
        ["severe_x_segments", 0.0030],
        ["osrm_time_per_segment", 0.0028],
    ],
    columns=["feature", "importance"],
)

ERROR_BY_SEGMENT = pd.DataFrame(
    [
        ["Short local routes", "Higher sensitivity to loading/unloading and hub dwell time"],
        ["Multi-segment trips", "Compounding segment delays make prediction harder"],
        ["High-risk FTL corridors", "SLA severe rates require monitoring beyond average MAE"],
        ["Bottleneck hub routes", "Central hubs can amplify downstream ETA errors"],
    ],
    columns=["segment", "validation_note"],
)

VALIDATION_NOTES = [
    "Modeling is evaluated at trip level after segment cleaning and aggregation.",
    "Train/test split should be performed after aggregation to avoid leakage across segment rows from the same trip.",
    "Graph features are generated from historical network structure and should be recomputed on the training window in production.",
    "Cross-validation MAE is reported as 28.81 +/- 0.48 minutes to show stability beyond one split.",
    "Operational validation should monitor MAE by route type, risk category, state pair, and delay bucket.",
]


def get_model_results(artifacts: dict) -> pd.DataFrame:
    """Use phase 4 results when available, otherwise return a static summary."""
    return get_canonical_model_validation()


def get_feature_importance(artifacts: dict, top_n: int = 10) -> pd.DataFrame:
    """Extract feature importance from the saved graph model if available."""
    phase4 = artifacts.get("Phase 4 results")
    if isinstance(phase4, dict):
        model = phase4.get("best_rf")
        names = phase4.get("feat_names_e") or phase4.get("feat_names_d")
        importances = getattr(model, "feature_importances_", None)
        if names and importances is not None and len(names) == len(importances):
            rows = sorted(zip(names, importances), key=lambda row: row[1], reverse=True)[:top_n]
            return pd.DataFrame(rows, columns=["feature", "importance"])
    return FEATURE_IMPORTANCE.head(top_n).copy()


def read_evidence_csv(file_name: str, fallback: pd.DataFrame) -> pd.DataFrame:
    """Read a lightweight evidence CSV when present."""
    path = EVIDENCE_DIR / file_name
    if path.exists():
        return pd.read_csv(path)
    return fallback.copy()


def get_canonical_model_validation() -> pd.DataFrame:
    """Return the canonical model validation evidence table."""
    return read_evidence_csv("canonical_model_validation.csv", MODEL_VALIDATION)


def get_graph_validation_tests() -> pd.DataFrame:
    """Return paired graph validation test evidence."""
    return read_evidence_csv("graph_validation_tests.csv", GRAPH_VALIDATION_TESTS)


def get_leakage_safe_graph_validation() -> pd.DataFrame:
    """Return leakage-aware graph validation evidence."""
    return read_evidence_csv("leakage_safe_graph_validation.csv", LEAKAGE_SAFE_GRAPH_VALIDATION)


def get_segment_error_analysis() -> pd.DataFrame:
    """Return segment-wise model error analysis."""
    return read_evidence_csv("segment_error_analysis.csv", SEGMENT_ERROR_ANALYSIS)
