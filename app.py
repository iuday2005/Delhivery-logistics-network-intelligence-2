from pathlib import Path

import pandas as pd
import streamlit as st

from src.artifacts import load_artifacts
from src.impact import (
    BUSINESS_IMPACT,
    apply_corridor_filters,
    apply_hub_filters,
    format_inr,
    get_corridor_risk_table,
    get_full_corridor_risk_table,
    get_hub_assumptions,
    get_hub_scenario_table,
    get_impact_reconciliation,
)
from src.model_validation import (
    ERROR_BY_SEGMENT,
    VALIDATION_NOTES,
    get_feature_importance,
    get_graph_validation_tests,
    get_leakage_safe_graph_validation,
    get_model_results,
    get_segment_error_analysis,
)


# ---------------------------------------------------------------------------
# Basic paths
# ---------------------------------------------------------------------------
# The dashboard uses lightweight result files and PNG plots only. It does not
# read the large raw CSV, which keeps deployment simple.
BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "assets" / "plots"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


st.set_page_config(
    page_title="Delhivery Logistics Network Intelligence",
    page_icon="DL",
    layout="wide",
)


def apply_custom_styles():
    """Apply dashboard styling."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #18212f;
            --muted: #5d6678;
            --line: #d9dee8;
            --panel: #ffffff;
            --soft: #f5f7fb;
            --accent: #d9232e;
            --accent-dark: #9e1c25;
            --blue: #1f6feb;
            --green: #178a62;
        }

        .main .block-container {
            padding-top: 1.5rem;
            max-width: 1240px;
        }

        .stApp {
            background: #f7f8fb;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        [data-testid="stSidebar"] {
            background: #111827;
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-left: 5px solid var(--accent);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            min-height: 118px;
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.86rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 1.8rem;
        }

        .hero {
            background: #111827;
            color: white;
            padding: 1.7rem 1.8rem;
            border-radius: 8px;
            margin-bottom: 1.25rem;
            border-left: 6px solid var(--accent);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
        }

        .hero h1 {
            color: white;
            margin-bottom: 0.35rem;
            font-size: 2.15rem;
        }

        .hero p {
            color: #e5e7eb;
            max-width: 860px;
            margin-bottom: 0;
            font-size: 1rem;
        }

        .section-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.6rem 0 1rem 0;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.04);
        }

        .callout {
            border-left: 5px solid var(--blue);
            background: #f3f7ff;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            color: var(--ink);
            margin: 1rem 0;
        }

        .impact-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 1rem 0 1.15rem 0;
        }

        .impact-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-top: 4px solid var(--green);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        }

        .impact-card.red {
            border-top-color: var(--accent);
        }

        .impact-card.blue {
            border-top-color: var(--blue);
        }

        .impact-card .label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .impact-card .value {
            color: var(--ink);
            font-size: 1.65rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }

        .impact-card .note {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }

        .risk-callout {
            border-left-color: var(--accent);
            background: #fff5f5;
        }

        .small-label {
            color: var(--muted);
            font-size: 0.86rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.04rem;
        }

        @media (max-width: 760px) {
            .impact-strip {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Static summary metrics used even when pickle files are missing or unreadable.
# ---------------------------------------------------------------------------
KPI_METRICS = [
    ("Raw segments", "144,867"),
    ("Cleaned segments", "141,661"),
    ("Trips", "14,804"),
    ("Facilities", "1,657"),
    ("Corridors", "2,781"),
    ("OSRM MAE", "161.5 min"),
    ("RF MAE", "30.95 min"),
    ("Graph RF MAE", "29.85 min"),
]

ARTIFACT_FILES = {
    "Phase 1 checkpoint": "phase1_checkpoint.pkl",
    "Phase 2 checkpoint": "phase2_checkpoint.pkl",
    "Phase 3 checkpoint": "phase3_checkpoint.pkl",
    "Phase 4 results": "phase4_results.pkl",
    "Phase 5 checkpoint": "phase5_checkpoint.pkl",
    "Delay propagation results": "propagation_results.pkl",
    "Corridor risk results": "corridor_risk_results.pkl",
    "Hub simulation results": "hub_simulation_results.pkl",
}

PLOT_FILES = {
    "delay_analysis": "delay_analysis.png",
    "network_bottleneck": "network_bottleneck.png",
    "phase4_graph_advantage": "phase4_graph_advantage.png",
    "phase5_strategy": "phase5_strategy.png",
    "temporal_explainability": "temporal_explainability.png",
    "hub_intervention_simulator": "hub_intervention_simulator.png",
    "delay_propagation": "delay_propagation.png",
    "corridor_risk_ranking": "corridor_risk_ranking.png",
    "baseline_residuals": "baseline_residuals.png",
}


def get_artifacts() -> dict:
    """Load all known artifacts into a dictionary."""
    return load_artifacts(ARTIFACTS_DIR, ARTIFACT_FILES)


def plot_path(plot_key: str) -> Path:
    """Return the full path for a named plot."""
    return PLOTS_DIR / PLOT_FILES[plot_key]


def show_plot(plot_key: str, caption: str):
    """Display a PNG plot if it exists, otherwise show a friendly warning."""
    image_path = plot_path(plot_key)
    if image_path.exists():
        st.image(str(image_path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing plot: {image_path}")


def show_kpi_cards():
    """Render KPI cards in two rows."""
    first_row = st.columns(4)
    second_row = st.columns(4)
    for column, (label, value) in zip(first_row + second_row, KPI_METRICS):
        column.metric(label, value)


def summarize_artifact(value):
    """Create a small readable preview for common artifact types."""
    if value is None:
        return None

    if isinstance(value, pd.DataFrame):
        return value.head(10)

    if isinstance(value, dict):
        rows = []
        for key, item in list(value.items())[:12]:
            rows.append({"key": str(key), "value": str(item)[:200]})
        return pd.DataFrame(rows)

    if isinstance(value, (list, tuple, set)):
        rows = [{"value": str(item)[:200]} for item in list(value)[:12]]
        return pd.DataFrame(rows)

    return str(value)[:500]


def artifact_status(artifacts: dict):
    """Show which pickle files loaded successfully."""
    status_rows = []
    for label, file_name in ARTIFACT_FILES.items():
        file_path = ARTIFACTS_DIR / file_name
        loaded = artifacts.get(label) is not None
        status_rows.append(
            {
                "Artifact": label,
                "File": file_name,
                "Found": "Yes" if file_path.exists() else "No",
                "Loaded": "Yes" if loaded else "No",
            }
        )

    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)


def page_header(title: str, description: str):
    st.markdown(
        f"""
        <div class="hero">
            <div class="small-label">Delhivery Logistics Network Intelligence</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_decision_panel(title: str, bullets: list[str], style: str = "callout"):
    """Render a compact business interpretation panel."""
    items = "".join(f"<li>{item}</li>" for item in bullets)
    st.markdown(
        f"""
        <div class="{style}">
            <strong>{title}</strong>
            <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_impact_cards(revenue_at_risk: float, recovery: float, breach_avoided: float):
    """Render business impact summary cards."""
    st.markdown(
        f"""
        <div class="impact-strip">
            <div class="impact-card red">
                <div class="label">Revenue at risk</div>
                <div class="value">{format_inr(revenue_at_risk)}</div>
                <div class="note">Estimated exposure from severe delivery delays.</div>
            </div>
            <div class="impact-card">
                <div class="label">Recoverable value</div>
                <div class="value">{format_inr(recovery)}</div>
                <div class="note">Potential recovery from targeted hub interventions.</div>
            </div>
            <div class="impact-card blue">
                <div class="label">SLA breaches avoided</div>
                <div class="value">{breach_avoided:,.0f}</div>
                <div class="note">Estimated from top recommended hub interventions.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_scale_note(recovery: float):
    """Explain how sample-window impact can scale in production."""
    annualized = recovery * 12
    st.markdown(
        f"""
        <div class="section-card">
            <strong>Scale interpretation</strong>
            <p>
                These values are calculated from the available project dataset, so they are best read as a
                sample-window impact estimate. In a production Delhivery setting, the same framework can be
                applied across more facilities, more corridors, longer time windows, and live shipment volume.
            </p>
            <p>
                For example, if the observed recoverable value of <strong>{format_inr(recovery)}</strong>
                represents a monthly operating window, the annualized opportunity is approximately
                <strong>{format_inr(annualized)}</strong>. The actual number should be recalculated using
                Delhivery's current shipment volume, SLA penalty model, and revenue per delayed shipment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def executive_summary(artifacts: dict):
    page_header(
        "Executive Summary",
        "A compact operating view of shipment volume, network scale, and ETA model performance.",
    )

    show_kpi_cards()

    show_decision_panel(
        "Executive takeaways",
        [
            "Graph-enhanced ETA modeling reduced MAE from 161.5 minutes to 29.85 minutes.",
            "The analyzed network spans 1,657 facilities and 2,781 corridors.",
            "The project moves beyond prediction into hub, corridor, intervention, and impact prioritization.",
        ],
    )

    col1, col2 = st.columns(2)
    with col1:
        show_plot(
            "phase5_strategy",
            "Business strategy summary for prioritizing interventions.",
        )
    with col2:
        show_plot(
            "delay_analysis",
            "Delay distribution and operational delay patterns.",
        )

    with st.expander("Artifact Load Status"):
        artifact_status(artifacts)


def business_impact(artifacts: dict):
    page_header(
        "Business Impact",
        "A decision view that translates model outputs into revenue recovery, SLA improvement, and operational priorities.",
    )

    phase5 = artifacts.get("Phase 5 checkpoint")
    revenue_at_risk = BUSINESS_IMPACT["revenue_at_risk"]
    recovery = BUSINESS_IMPACT["potential_recovery"]
    if isinstance(phase5, dict):
        revenue_at_risk = float(phase5.get("revenue_at_risk", revenue_at_risk))
        recovery = float(phase5.get("potential_recovery", recovery))

    hub_table = get_hub_scenario_table(artifacts)
    hub_options = sorted(hub_table["hub"].dropna().astype(str).unique().tolist())
    pct_options = sorted(hub_table["intervention_pct"].dropna().astype(int).unique().tolist())

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_hubs = st.multiselect("Hub filter", hub_options, default=hub_options[:5])
    with filter_col2:
        selected_pct = st.multiselect("Intervention scenario", pct_options, default=[max(pct_options)])

    hub_table = apply_hub_filters(hub_table, selected_hubs, selected_pct)
    if hub_table.empty:
        st.warning("No hub interventions match the selected filters.")
        return

    scenario_recovery = hub_table["revenue_recovered_inr"].sum()
    breaches_avoided = hub_table["sla_breaches_avoided"].sum()
    show_impact_cards(revenue_at_risk, scenario_recovery, breaches_avoided)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Filtered scenarios", f"{len(hub_table):,}")
    col_b.metric("Selected hubs", f"{hub_table['hub'].nunique():,}")
    col_c.metric("Avg ETA gain", f"{hub_table['eta_improvement_min'].mean():.2f} min")
    show_scale_note(scenario_recovery)

    st.subheader("Impact Claim Boundaries")
    st.markdown(
        """
        The recovery values are scenario estimates, not realized Delhivery P&L.
        The canonical headline uses the conservative phase 5 sample-window
        recovery estimate. Hub-level values below are intervention scenarios
        based on explicit SLA penalty, annualization, and delay-reduction
        assumptions.
        """
    )
    assumptions = get_hub_assumptions()
    reconciliation = get_impact_reconciliation()
    tab_assumptions, tab_claims = st.tabs(["Simulator assumptions", "Claim reconciliation"])
    with tab_assumptions:
        st.dataframe(assumptions, use_container_width=True, hide_index=True)
    with tab_claims:
        st.dataframe(reconciliation, use_container_width=True, hide_index=True)

    show_decision_panel(
        "Why this matters",
        [
            "The model identifies where delays happen, but the impact layer shows where action has financial value.",
            "The current estimates come from the project dataset and can be amplified across a larger live network.",
            "Top hub interventions can reduce delay ratios, avoid SLA breaches, and recover revenue without changing every corridor.",
            "Corridor risk ranking creates a practical operations backlog for weekly review and escalation.",
        ],
    )

    st.subheader("Recommended Hub Interventions")
    display_hubs = hub_table[
        [
            "hub",
            "intervention_pct",
            "trip_volume",
            "affected_corridors",
            "eta_improvement_min",
            "sla_breaches_avoided",
            "revenue_recovered_inr",
            "roi_score",
        ]
    ].copy()
    display_hubs["eta_improvement_min"] = display_hubs["eta_improvement_min"].round(2)
    display_hubs["sla_breaches_avoided"] = display_hubs["sla_breaches_avoided"].round(0).astype(int)
    display_hubs["revenue_recovered_inr"] = display_hubs["revenue_recovered_inr"].round(0).astype(int)
    display_hubs["roi_score"] = display_hubs["roi_score"].round(2)

    st.dataframe(
        display_hubs,
        use_container_width=True,
        hide_index=True,
        column_config={
            "hub": "Hub",
            "intervention_pct": "Delay reduction %",
            "trip_volume": "Trips",
            "affected_corridors": "Affected corridors",
            "eta_improvement_min": "ETA improvement (min)",
            "sla_breaches_avoided": "SLA breaches avoided",
            "revenue_recovered_inr": st.column_config.NumberColumn("Revenue recovered", format="INR %d"),
            "roi_score": "ROI score",
        },
    )

    chart_data = display_hubs.set_index("hub")[["revenue_recovered_inr", "sla_breaches_avoided"]]
    st.bar_chart(chart_data, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Risk Corridors")
        risk_table = get_corridor_risk_table(artifacts, top_n=10)
        st.dataframe(risk_table, use_container_width=True, hide_index=True)
    with col2:
        show_plot(
            "hub_intervention_simulator",
            "Simulated benefit from 10%, 20%, and 30% hub delay reduction scenarios.",
        )

    show_decision_panel(
        "Recommended operating plan",
        [
            "Run a 30% delay reduction pilot on the highest ROI hubs first.",
            "Review critical FTL corridors with severe SLA rates near 100%.",
            "Track recovered SLA breaches and revenue weekly to validate intervention ROI.",
        ],
        style="callout risk-callout",
    )


def eta_model_performance(artifacts: dict):
    page_header(
        "ETA Model Performance",
        "Comparison of baseline routing estimates, machine learning ETA predictions, and graph-enhanced features.",
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("OSRM Baseline MAE", "161.5 min")
    col2.metric("Random Forest MAE", "30.95 min")
    col3.metric("Graph RF MAE", "29.81 min", delta="-1.13 min vs RF")

    show_decision_panel(
        "Model story",
        [
            "The baseline routing estimate is useful operational context but weak as an ETA predictor.",
            "Random Forest captures most explainable structure in segment-level ETA.",
            "Graph features add incremental lift and make the model more operationally interpretable.",
        ],
    )

    col1, col2 = st.columns(2)
    with col1:
        show_plot(
            "phase4_graph_advantage",
            "Graph features improved ETA prediction accuracy.",
        )
    with col2:
        show_plot(
            "baseline_residuals",
            "Residual behavior of the baseline ETA model.",
        )

    show_plot(
        "temporal_explainability",
        "Temporal features explain time-window and scheduling effects in ETA predictions.",
    )

    st.subheader("Validation Summary")
    validation_results = get_model_results(artifacts)
    st.dataframe(validation_results, use_container_width=True, hide_index=True)

    st.subheader("Validation Evidence")
    graph_tests = get_graph_validation_tests()
    leakage_safe = get_leakage_safe_graph_validation()
    segment_errors = get_segment_error_analysis()
    tab_tests, tab_leakage, tab_errors = st.tabs(
        ["Wilcoxon test", "Leakage-safe graph check", "Segment-wise errors"]
    )
    with tab_tests:
        st.dataframe(graph_tests, use_container_width=True, hide_index=True)
    with tab_leakage:
        st.dataframe(leakage_safe, use_container_width=True, hide_index=True)
        st.caption(
            "Leakage-safe graph validation rebuilds graph features from training trips only, then maps them to validation trips."
        )
    with tab_errors:
        st.dataframe(
            segment_errors.sort_values("mae_min", ascending=False).head(20),
            use_container_width=True,
            hide_index=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Feature Importance")
        feature_importance = get_feature_importance(artifacts)
        st.bar_chart(feature_importance.set_index("feature"), use_container_width=True)
    with col2:
        st.subheader("Operational Error Notes")
        st.dataframe(ERROR_BY_SEGMENT, use_container_width=True, hide_index=True)

    show_decision_panel("Validation controls", VALIDATION_NOTES)


def network_bottlenecks():
    page_header(
        "Network Bottlenecks",
        "Facility-level network analysis highlights hubs that are central, congested, or delay-prone.",
    )

    show_plot(
        "network_bottleneck",
        "Network bottleneck ranking based on graph and operational signals.",
    )

    st.info(
        "Use these bottleneck hubs as candidates for staffing, process, dispatch, or capacity interventions."
    )
    show_decision_panel(
        "How to use this page",
        [
            "Start with high-centrality hubs where small delays can affect many downstream corridors.",
            "Prioritize hubs that combine bottleneck score, delay rate, and business volume.",
            "Use intervention simulations before proposing costly capacity changes.",
        ],
    )


def corridor_risk_ranking(artifacts: dict):
    page_header(
        "Corridor Risk Ranking",
        "Corridor-level risk scoring combines delay behavior, SLA breach tendency, volume, and graph context.",
    )

    risk_table = get_full_corridor_risk_table(artifacts)
    route_options = sorted(risk_table["route_type"].dropna().astype(str).unique().tolist())
    category_options = sorted(risk_table["risk_category"].dropna().astype(str).unique().tolist())
    state_cols = [col for col in ["src_state", "dst_state"] if col in risk_table.columns]
    state_options = sorted(
        pd.concat([risk_table[col].dropna().astype(str) for col in state_cols]).unique().tolist()
    ) if state_cols else []

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_routes = st.multiselect("Route type", route_options, default=route_options)
    with col2:
        default_categories = [cat for cat in ["Critical", "High"] if cat in category_options] or category_options
        selected_categories = st.multiselect("Risk category", category_options, default=default_categories)
    with col3:
        min_trips = st.number_input("Minimum trips", min_value=1, max_value=100, value=4, step=1)

    selected_states = []
    if state_options:
        selected_states = st.multiselect("State filter", state_options, default=[])

    filtered = apply_corridor_filters(risk_table, selected_routes, selected_categories, int(min_trips))
    if selected_states and state_cols:
        state_mask = pd.Series(False, index=filtered.index)
        for col in state_cols:
            state_mask = state_mask | filtered[col].astype(str).isin(selected_states)
        filtered = filtered[state_mask]

    show_plot(
        "corridor_risk_ranking",
        "Highest-risk corridors after filtering unreliable low-volume outliers.",
    )

    show_decision_panel(
        "Product analytics angle",
        [
            "Corridor ranking translates model output into a backlog of operational fixes.",
            "Volume filtering keeps the ranking from overreacting to one-off failures.",
            "This can support SLA monitoring, lane-level alerting, and partner performance reviews.",
        ],
        style="callout risk-callout",
    )

    st.subheader("Filtered Corridor Risk Table")
    if filtered.empty:
        st.warning("No corridors match the selected filters.")
    else:
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Filtered corridors", f"{len(filtered):,}")
        metric_col2.metric("Avg risk score", f"{filtered['risk_score'].mean():.2f}")
        metric_col3.metric("Avg severe SLA rate", f"{filtered['sla_severe_rate'].mean() * 100:.1f}%")
        sort_col = "risk_score" if "risk_score" in filtered.columns else filtered.columns[0]
        st.dataframe(
            filtered.sort_values(sort_col, ascending=False).head(25),
            use_container_width=True,
            hide_index=True,
        )


def delay_propagation(artifacts: dict):
    page_header(
        "Delay Propagation",
        "Propagation analysis separates hubs that create, amplify, or receive downstream delay.",
    )

    show_plot(
        "delay_propagation",
        "Delay propagation roles across the logistics network.",
    )

    show_decision_panel(
        "Network insight",
        [
            "Delay sources are good candidates for root-cause analysis.",
            "Delay amplifiers may need process controls because they spread disruption.",
            "Delay receivers help explain downstream customer impact.",
        ],
    )

    preview = summarize_artifact(artifacts.get("Delay propagation results"))
    if isinstance(preview, pd.DataFrame):
        st.subheader("Loaded Artifact Preview")
        st.dataframe(preview, use_container_width=True, hide_index=True)


def hub_intervention_simulator(artifacts: dict):
    page_header(
        "Hub Intervention Simulator",
        "Scenario view for estimating the impact of reducing delays at priority hubs.",
    )

    reduction = st.slider(
        "Delay reduction scenario",
        min_value=10,
        max_value=30,
        value=20,
        step=10,
        help="Matches the 10%, 20%, and 30% intervention scenarios from the analysis.",
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Scenario", f"{reduction}%")
    col2.metric("Target", "Top bottleneck hubs")
    col3.metric("Expected effect", "Lower ETA error and SLA breaches")

    scenario_table = get_hub_scenario_table(artifacts)
    hub_options = sorted(scenario_table["hub"].dropna().astype(str).unique().tolist())
    selected_hubs = st.multiselect("Compare hubs", hub_options, default=hub_options[:5])
    scenario_table = apply_hub_filters(scenario_table, selected_hubs, [reduction])
    if scenario_table.empty:
        st.warning("No hub scenarios match the selected filters.")
        return

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Hubs selected", f"{scenario_table['hub'].nunique():,}")
    col_b.metric("Total recovery", format_inr(scenario_table["revenue_recovered_inr"].sum()))
    col_c.metric("Breaches avoided", f"{scenario_table['sla_breaches_avoided'].sum():,.0f}")

    show_decision_panel(
        "Simulator value",
        [
            "Frames analytics as a decision tool, not just reporting.",
            "Helps compare intervention intensity before committing operations resources.",
            "Connects model outcomes to SLA breach reduction and revenue recovery.",
        ],
    )

    show_plot(
        "hub_intervention_simulator",
        "Estimated value of hub-level delay reduction scenarios.",
    )

    st.subheader("Selected Scenario Details")
    display_cols = [
        "hub",
        "trip_volume",
        "affected_corridors",
        "eta_improvement_min",
        "sla_breaches_avoided",
        "revenue_recovered_inr",
        "roi_score",
    ]
    available_cols = [col for col in display_cols if col in scenario_table.columns]
    st.dataframe(
        scenario_table[available_cols].sort_values("roi_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    chart_cols = ["revenue_recovered_inr", "sla_breaches_avoided"]
    if all(col in scenario_table.columns for col in chart_cols):
        st.bar_chart(scenario_table.set_index("hub")[chart_cols], use_container_width=True)


def methodology_and_limitations(artifacts: dict):
    page_header(
        "Methodology & Limitations",
        "A brief project map from raw shipment segments to decision-ready logistics intelligence.",
    )

    st.subheader("Methodology")
    st.markdown(
        """
        1. Cleaned shipment segment records and removed inconsistent data.
        2. Built baseline ETA and Random Forest ETA models.
        3. Constructed a facility-corridor graph using NetworkX.
        4. Added graph features such as centrality, bottleneck scores, and embeddings.
        5. Ranked corridors and hubs for operational intervention.
        6. Simulated delay reduction scenarios for business impact.
        """
    )

    st.subheader("Model Validation")
    st.markdown(
        """
        - Models are evaluated at trip level after segment-level cleaning and aggregation.
        - Train/test splitting should happen after trip aggregation to reduce leakage from multiple segments in the same trip.
        - Graph features are useful, but in production they should be recomputed only from historical training-window data.
        - Cross-validation MAE is reported as 28.81 +/- 0.48 minutes to show stability across folds.
        - Operational monitoring should track MAE by route type, risk category, state pair, and delay bucket.
        """
    )

    st.subheader("Business Impact Assumptions")
    st.markdown(
        """
        - Revenue impact is a sample-window estimate, not a full Delhivery P&L claim.
        - Annualized impact assumes the observed recoverable value represents one comparable operating month.
        - Final production ROI should include intervention cost, current shipment volume, SLA penalty rules, and revenue per delayed shipment.
        - Hub recommendations are prioritization signals for pilots, not automatic capacity investment decisions.
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - The dashboard is designed for presentation and does not retrain models.
        - It avoids loading the large raw CSV to keep the app deployable.
        - Results depend on the quality and completeness of the original shipment data.
        - Large pickle artifacts are intentionally excluded from GitHub.
        - Pickle artifacts are optional; static plots and fallback summary tables keep the app available if they fail to load.
        """
    )

    with st.expander("Artifact Load Status"):
        artifact_status(artifacts)


def main():
    artifacts = get_artifacts()

    st.sidebar.title("Delhivery Intelligence")
    st.sidebar.caption("Logistics network analytics dashboard")
    apply_custom_styles()

    page = st.sidebar.radio(
        "Navigation",
        [
            "Executive Summary",
            "Business Impact",
            "ETA Model Performance",
            "Network Bottlenecks",
            "Corridor Risk Ranking",
            "Delay Propagation",
            "Hub Intervention Simulator",
            "Methodology & Limitations",
        ],
    )

    st.sidebar.divider()
    st.sidebar.write("Data source")
    st.sidebar.caption("Uses PNG plots and optional pickle artifacts.")

    if page == "Executive Summary":
        executive_summary(artifacts)
    elif page == "Business Impact":
        business_impact(artifacts)
    elif page == "ETA Model Performance":
        eta_model_performance(artifacts)
    elif page == "Network Bottlenecks":
        network_bottlenecks()
    elif page == "Corridor Risk Ranking":
        corridor_risk_ranking(artifacts)
    elif page == "Delay Propagation":
        delay_propagation(artifacts)
    elif page == "Hub Intervention Simulator":
        hub_intervention_simulator(artifacts)
    else:
        methodology_and_limitations(artifacts)


if __name__ == "__main__":
    main()
