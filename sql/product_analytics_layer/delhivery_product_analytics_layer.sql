-- Delhivery Logistics Network Intelligence
-- Product Analytics SQL Layer
-- Dialect: MySQL 8.0+
--
-- Business story:
-- Problem      -> Which corridors and hubs are creating SLA risk?
-- Diagnosis    -> Are delays concentrated by lane, cohort, hub, or time period?
-- Intervention -> Which lanes/hubs should operations fix first?
-- Impact       -> What INR recovery and North Star movement should leaders expect?
--
-- Assumption:
-- The raw CSV at data/delivery_data.csv is loaded into a MySQL table named
-- delivery_segments with column names matching the CSV header.
--
-- Optional one-time table shape for local MySQL users. Keep it commented if
-- you already loaded the CSV into MySQL.
/*
CREATE TABLE IF NOT EXISTS delivery_segments (
    `data` VARCHAR(32),
    trip_creation_time DATETIME(6),
    route_schedule_uuid VARCHAR(128),
    route_type VARCHAR(32),
    trip_uuid VARCHAR(128),
    source_center VARCHAR(32),
    source_name VARCHAR(255),
    destination_center VARCHAR(32),
    destination_name VARCHAR(255),
    od_start_time DATETIME(6),
    od_end_time DATETIME(6),
    start_scan_to_end_scan DOUBLE,
    is_cutoff VARCHAR(8),
    cutoff_factor INT,
    cutoff_timestamp DATETIME,
    actual_distance_to_destination DOUBLE,
    actual_time DOUBLE,
    osrm_time DOUBLE,
    osrm_distance DOUBLE,
    factor DOUBLE,
    segment_actual_time DOUBLE,
    segment_osrm_time DOUBLE,
    segment_osrm_distance DOUBLE,
    segment_factor DOUBLE
);

LOAD DATA LOCAL INFILE 'D:/iit g/projects/delhivery/data/delivery_data.csv'
INTO TABLE delivery_segments
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
*/

-- ---------------------------------------------------------------------------
-- 0. Shared clean feature layer
-- ---------------------------------------------------------------------------
-- WHY this exists:
-- Product analytics should not rewrite metric definitions in every query.
-- This view standardizes corridor, delay, SLA, week, and month fields so the
-- dashboard, cohorts, diagnostics, and executive summary speak the same metric
-- language as the Python/ML project.

CREATE OR REPLACE VIEW v_delivery_segment_product_features AS
WITH base_segments AS (
    SELECT
        trip_uuid,
        route_schedule_uuid,
        route_type,
        source_center,
        source_name,
        destination_center,
        destination_name,
        CONCAT(source_center, ' -> ', destination_center, ' | ', route_type) AS lane_id,
        trip_creation_time,
        DATE(trip_creation_time) AS trip_date,
        DATE_SUB(DATE(trip_creation_time), INTERVAL WEEKDAY(trip_creation_time) DAY) AS week_start,
        CAST(DATE_FORMAT(trip_creation_time, '%Y-%m-01') AS DATE) AS month_start,
        actual_distance_to_destination,
        actual_time,
        osrm_time,
        segment_actual_time,
        segment_osrm_time,
        segment_osrm_distance
    FROM delivery_segments
    WHERE trip_uuid IS NOT NULL
      AND source_center IS NOT NULL
      AND destination_center IS NOT NULL
      AND route_type IS NOT NULL
      AND trip_creation_time IS NOT NULL
      AND segment_actual_time IS NOT NULL
      AND segment_osrm_time IS NOT NULL
      AND segment_actual_time >= 0
      AND segment_osrm_time > 0
),
scored_segments AS (
    SELECT
        *,
        GREATEST(segment_actual_time - segment_osrm_time, 0) AS delay_minutes,
        segment_actual_time / NULLIF(segment_osrm_time, 0) AS delay_ratio,
        CASE WHEN segment_actual_time > segment_osrm_time THEN 1 ELSE 0 END AS is_delayed,
        CASE WHEN segment_actual_time <= segment_osrm_time THEN 1 ELSE 0 END AS is_on_time,
        CASE WHEN segment_actual_time > 2 * segment_osrm_time THEN 1 ELSE 0 END AS is_sla_breach,
        CASE
            WHEN segment_actual_time / NULLIF(segment_osrm_time, 0) >= 3 THEN 1
            ELSE 0
        END AS is_extreme_breach
    FROM base_segments
)
SELECT *
FROM scored_segments;

-- ---------------------------------------------------------------------------
-- 1A. Corridor Performance Dashboard: delay-frequency ranking
-- ---------------------------------------------------------------------------
-- WHY this exists:
-- PMs and operations teams need a weekly backlog, not just model scores.
-- DENSE_RANK keeps corridors with equal delayed-shipment counts in the same
-- priority band, which is easier to explain in a review meeting than arbitrary
-- row numbering.

CREATE OR REPLACE VIEW v_corridor_delay_frequency_rank AS
WITH corridor_metrics AS (
    SELECT
        lane_id,
        source_center,
        destination_center,
        route_type,
        COUNT(*) AS segment_count,
        COUNT(DISTINCT trip_uuid) AS trip_count,
        SUM(is_delayed) AS delayed_segment_count,
        SUM(is_sla_breach) AS sla_breach_count,
        ROUND(100 * AVG(is_delayed), 2) AS delay_frequency_pct,
        ROUND(100 * AVG(is_sla_breach), 2) AS sla_breach_rate_pct,
        ROUND(AVG(delay_minutes), 2) AS avg_delay_minutes,
        ROUND(AVG(delay_ratio), 2) AS avg_delay_ratio
    FROM v_delivery_segment_product_features
    GROUP BY lane_id, source_center, destination_center, route_type
),
ranked_corridors AS (
    SELECT
        *,
        DENSE_RANK() OVER (
            ORDER BY delayed_segment_count DESC, sla_breach_rate_pct DESC, avg_delay_minutes DESC
        ) AS delay_frequency_rank
    FROM corridor_metrics
)
SELECT *
FROM ranked_corridors;

-- ---------------------------------------------------------------------------
-- 1B. Corridor Performance Dashboard: week-over-week trend
-- ---------------------------------------------------------------------------
-- WHY this exists:
-- Risk is only actionable if leaders can see whether a corridor is getting
-- better or worse after dispatch changes, staffing changes, or hub interventions.

CREATE OR REPLACE VIEW v_corridor_week_over_week_delay_trend AS
WITH weekly_corridor_metrics AS (
    SELECT
        lane_id,
        source_center,
        destination_center,
        route_type,
        week_start,
        COUNT(*) AS segment_count,
        COUNT(DISTINCT trip_uuid) AS trip_count,
        ROUND(AVG(delay_minutes), 2) AS avg_delay_minutes,
        ROUND(100 * AVG(is_sla_breach), 2) AS sla_breach_rate_pct
    FROM v_delivery_segment_product_features
    GROUP BY lane_id, source_center, destination_center, route_type, week_start
),
weekly_with_lag AS (
    SELECT
        *,
        LAG(avg_delay_minutes) OVER (PARTITION BY lane_id ORDER BY week_start) AS prior_week_avg_delay_minutes,
        LAG(sla_breach_rate_pct) OVER (PARTITION BY lane_id ORDER BY week_start) AS prior_week_sla_breach_rate_pct
    FROM weekly_corridor_metrics
)
SELECT
    lane_id,
    source_center,
    destination_center,
    route_type,
    week_start,
    segment_count,
    trip_count,
    avg_delay_minutes,
    prior_week_avg_delay_minutes,
    ROUND(avg_delay_minutes - prior_week_avg_delay_minutes, 2) AS wow_avg_delay_change_minutes,
    sla_breach_rate_pct,
    prior_week_sla_breach_rate_pct,
    ROUND(sla_breach_rate_pct - prior_week_sla_breach_rate_pct, 2) AS wow_sla_breach_rate_change_pct
FROM weekly_with_lag;

-- ---------------------------------------------------------------------------
-- 1C. Corridor Performance Dashboard: top 10 highest risk lanes
-- ---------------------------------------------------------------------------
-- WHY this exists:
-- A PM needs a small number of lanes to escalate this week. The minimum volume
-- guardrail prevents one-off noisy lanes from dominating the operating agenda.

CREATE OR REPLACE VIEW v_top10_highest_risk_lanes_by_sla_breach AS
WITH lane_metrics AS (
    SELECT
        lane_id,
        source_center,
        destination_center,
        route_type,
        COUNT(*) AS segment_count,
        COUNT(DISTINCT trip_uuid) AS trip_count,
        SUM(is_sla_breach) AS sla_breach_count,
        ROUND(100 * AVG(is_sla_breach), 2) AS sla_breach_rate_pct,
        ROUND(AVG(delay_minutes), 2) AS avg_delay_minutes,
        ROUND(AVG(delay_ratio), 2) AS avg_delay_ratio
    FROM v_delivery_segment_product_features
    GROUP BY lane_id, source_center, destination_center, route_type
),
eligible_lanes AS (
    SELECT *
    FROM lane_metrics
    WHERE trip_count >= 5
),
ranked_lanes AS (
    SELECT
        *,
        DENSE_RANK() OVER (
            ORDER BY sla_breach_rate_pct DESC, sla_breach_count DESC, avg_delay_minutes DESC
        ) AS risk_rank,
        ROW_NUMBER() OVER (
            ORDER BY sla_breach_rate_pct DESC, sla_breach_count DESC, avg_delay_minutes DESC, lane_id
        ) AS risk_row_number
    FROM eligible_lanes
)
SELECT *
FROM ranked_lanes
WHERE risk_row_number <= 10;

-- ---------------------------------------------------------------------------
-- 2A. SLA Breach Cohort Analysis
-- ---------------------------------------------------------------------------
-- WHY this exists:
-- Cohorts turn thousands of corridors into management segments. This lets a PM
-- describe the shape of the problem: how much business impact sits in low,
-- medium, high, and critical lanes.

CREATE OR REPLACE VIEW v_sla_breach_cohort_analysis AS
WITH params AS (
    SELECT 150.00 AS assumed_sla_penalty_inr
),
lane_metrics AS (
    SELECT
        lane_id,
        source_center,
        destination_center,
        route_type,
        COUNT(*) AS segment_count,
        COUNT(DISTINCT trip_uuid) AS trip_count,
        SUM(is_sla_breach) AS sla_breach_count,
        ROUND(AVG(delay_minutes), 2) AS avg_delay_minutes,
        ROUND(AVG(delay_ratio), 2) AS avg_delay_ratio,
        ROUND(100 * AVG(is_sla_breach), 2) AS sla_breach_rate_pct
    FROM v_delivery_segment_product_features
    GROUP BY lane_id, source_center, destination_center, route_type
),
lane_cohorts AS (
    SELECT
        lane_metrics.*,
        CASE
            WHEN sla_breach_rate_pct >= 50 OR avg_delay_ratio >= 3.00 THEN 'critical'
            WHEN sla_breach_rate_pct >= 25 OR avg_delay_ratio >= 2.00 THEN 'high'
            WHEN sla_breach_rate_pct >= 10 OR avg_delay_ratio >= 1.25 THEN 'medium'
            ELSE 'low'
        END AS delay_severity_cohort
    FROM lane_metrics
),
cohort_rollup AS (
    SELECT
        delay_severity_cohort,
        COUNT(*) AS corridor_count,
        SUM(segment_count) AS segment_count,
        SUM(trip_count) AS trip_count,
        SUM(sla_breach_count) AS sla_breach_count,
        ROUND(100 * SUM(sla_breach_count) / NULLIF(SUM(segment_count), 0), 2) AS breach_rate_pct,
        ROUND(AVG(avg_delay_minutes), 2) AS avg_delay_minutes,
        ROUND(SUM(sla_breach_count) * MAX(params.assumed_sla_penalty_inr), 0) AS estimated_recovery_cost_inr
    FROM lane_cohorts
    CROSS JOIN params
    GROUP BY delay_severity_cohort
)
SELECT
    delay_severity_cohort,
    corridor_count,
    segment_count,
    trip_count,
    sla_breach_count,
    breach_rate_pct,
    avg_delay_minutes,
    estimated_recovery_cost_inr
FROM cohort_rollup
ORDER BY
    CASE delay_severity_cohort
        WHEN 'critical' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        ELSE 4
    END;

-- ---------------------------------------------------------------------------
-- 2B. Month-on-month retention of on-time corridors
-- ---------------------------------------------------------------------------
-- WHY this exists:
-- North Star metrics should be durable. A lane that is on time once but churns
-- back into delay next month is not an operational win.

CREATE OR REPLACE VIEW v_monthly_on_time_corridor_retention AS
WITH monthly_lane_metrics AS (
    SELECT
        lane_id,
        month_start,
        COUNT(*) AS segment_count,
        COUNT(DISTINCT trip_uuid) AS trip_count,
        ROUND(AVG(delay_minutes), 2) AS avg_delay_minutes,
        ROUND(AVG(delay_ratio), 2) AS avg_delay_ratio,
        ROUND(100 * AVG(is_sla_breach), 2) AS sla_breach_rate_pct
    FROM v_delivery_segment_product_features
    GROUP BY lane_id, month_start
),
monthly_lane_cohorts AS (
    SELECT
        *,
        CASE
            WHEN sla_breach_rate_pct >= 50 OR avg_delay_ratio >= 3.00 THEN 'critical'
            WHEN sla_breach_rate_pct >= 25 OR avg_delay_ratio >= 2.00 THEN 'high'
            WHEN sla_breach_rate_pct >= 10 OR avg_delay_ratio >= 1.25 THEN 'medium'
            ELSE 'low'
        END AS delay_severity_cohort,
        CASE
            WHEN sla_breach_rate_pct <= 5 AND avg_delay_minutes <= 15 THEN 1
            ELSE 0
        END AS is_on_time_corridor
    FROM monthly_lane_metrics
),
retention_pairs AS (
    SELECT
        current_month.month_start,
        current_month.delay_severity_cohort,
        current_month.lane_id,
        current_month.is_on_time_corridor AS current_is_on_time,
        next_month.is_on_time_corridor AS next_is_on_time
    FROM monthly_lane_cohorts AS current_month
    LEFT JOIN monthly_lane_cohorts AS next_month
      ON current_month.lane_id = next_month.lane_id
     AND next_month.month_start = DATE_ADD(current_month.month_start, INTERVAL 1 MONTH)
),
retention_rollup AS (
    SELECT
        month_start,
        delay_severity_cohort,
        COUNT(DISTINCT CASE WHEN current_is_on_time = 1 THEN lane_id END) AS on_time_corridors_this_month,
        COUNT(DISTINCT CASE WHEN current_is_on_time = 1 AND next_is_on_time = 1 THEN lane_id END) AS retained_on_time_corridors_next_month
    FROM retention_pairs
    GROUP BY month_start, delay_severity_cohort
)
SELECT
    month_start,
    delay_severity_cohort,
    on_time_corridors_this_month,
    retained_on_time_corridors_next_month,
    ROUND(
        100 * retained_on_time_corridors_next_month / NULLIF(on_time_corridors_this_month, 0),
        2
    ) AS on_time_corridor_retention_pct
FROM retention_rollup
ORDER BY month_start, delay_severity_cohort;

-- ---------------------------------------------------------------------------
-- 3A. Hub-level diagnostic: downstream delay contribution
-- ---------------------------------------------------------------------------
-- WHY this exists:
-- Corridor symptoms often originate upstream. This query reframes the analysis
-- from "which lane is bad" to "which hub is creating downstream operational
-- drag across many lanes."

CREATE OR REPLACE VIEW v_hub_downstream_delay_diagnostic AS
WITH hub_departures AS (
    SELECT
        source_center AS hub,
        COUNT(*) AS total_departures,
        COUNT(DISTINCT trip_uuid) AS trip_count,
        COUNT(DISTINCT destination_center) AS downstream_corridors,
        SUM(is_sla_breach) AS downstream_sla_breaches,
        ROUND(SUM(delay_minutes), 2) AS downstream_delay_minutes,
        ROUND(AVG(delay_minutes), 2) AS avg_downstream_delay_minutes,
        ROUND(100 * AVG(is_sla_breach), 2) AS downstream_sla_breach_rate_pct
    FROM v_delivery_segment_product_features
    GROUP BY source_center
),
ranked_hubs AS (
    SELECT
        *,
        DENSE_RANK() OVER (
            ORDER BY downstream_delay_minutes DESC, downstream_sla_breaches DESC
        ) AS downstream_delay_contribution_rank
    FROM hub_departures
)
SELECT *
FROM ranked_hubs;

-- ---------------------------------------------------------------------------
-- 3B. Hub efficiency score and drop alert
-- ---------------------------------------------------------------------------
-- WHY this exists:
-- A hub can look acceptable in aggregate while deteriorating week by week.
-- This view flags hubs whose on-time departure efficiency dropped by more than
-- 10 percent versus their previous observed period.

CREATE OR REPLACE VIEW v_hub_efficiency_drop_alerts AS
WITH weekly_hub_efficiency AS (
    SELECT
        source_center AS hub,
        week_start,
        COUNT(*) AS total_departures,
        SUM(is_on_time) AS on_time_departures,
        ROUND(SUM(is_on_time) / NULLIF(COUNT(*), 0), 4) AS hub_efficiency_score
    FROM v_delivery_segment_product_features
    GROUP BY source_center, week_start
),
weekly_with_lag AS (
    SELECT
        *,
        LAG(hub_efficiency_score) OVER (PARTITION BY hub ORDER BY week_start) AS previous_hub_efficiency_score
    FROM weekly_hub_efficiency
),
drop_flags AS (
    SELECT
        *,
        ROUND(
            100 * (previous_hub_efficiency_score - hub_efficiency_score) / NULLIF(previous_hub_efficiency_score, 0),
            2
        ) AS relative_efficiency_drop_pct,
        CASE
            WHEN hub_efficiency_score < previous_hub_efficiency_score * 0.90 THEN 1
            ELSE 0
        END AS efficiency_dropped_gt_10pct
    FROM weekly_with_lag
)
SELECT *
FROM drop_flags
WHERE previous_hub_efficiency_score IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 4A. Business Impact Summary Table: lane-level intervention backlog
-- ---------------------------------------------------------------------------
-- WHY this exists:
-- Product prioritization is impact vs effort. This turns SLA failures into a
-- practical lane backlog with INR impact, likely effort, and an ordered fix list.

CREATE OR REPLACE VIEW v_business_impact_summary_table AS
WITH params AS (
    SELECT
        150.00 AS assumed_sla_penalty_inr,
        0.70 AS assumed_fix_effectiveness_pct
),
lane_impact AS (
    SELECT
        lane_id,
        source_center,
        destination_center,
        route_type,
        COUNT(*) AS segment_count,
        COUNT(DISTINCT trip_uuid) AS trip_count,
        SUM(is_sla_breach) AS sla_breach_count,
        ROUND(100 * AVG(is_sla_breach), 2) AS sla_breach_rate_pct,
        ROUND(AVG(delay_minutes), 2) AS avg_delay_minutes,
        ROUND(SUM(delay_minutes), 2) AS total_delay_minutes
    FROM v_delivery_segment_product_features
    GROUP BY lane_id, source_center, destination_center, route_type
),
impact_scoring AS (
    SELECT
        lane_impact.*,
        ROUND(sla_breach_count * params.assumed_sla_penalty_inr, 0) AS total_projected_cost_of_delay_inr,
        ROUND(sla_breach_count * params.assumed_sla_penalty_inr * params.assumed_fix_effectiveness_pct, 0) AS expected_recovery_if_fixed_inr,
        CASE
            WHEN route_type = 'FTL' AND trip_count >= 25 THEN 'high'
            WHEN route_type = 'FTL' OR trip_count >= 25 THEN 'medium'
            ELSE 'low'
        END AS estimated_effort,
        CASE
            WHEN route_type = 'FTL' AND trip_count >= 25 THEN 3
            WHEN route_type = 'FTL' OR trip_count >= 25 THEN 2
            ELSE 1
        END AS effort_score
    FROM lane_impact
    CROSS JOIN params
),
prioritized_lanes AS (
    SELECT
        *,
        ROUND(expected_recovery_if_fixed_inr / NULLIF(effort_score, 0), 2) AS impact_vs_effort_score,
        DENSE_RANK() OVER (
            ORDER BY expected_recovery_if_fixed_inr / NULLIF(effort_score, 0) DESC,
                     sla_breach_rate_pct DESC,
                     trip_count DESC
        ) AS intervention_priority_rank
    FROM impact_scoring
)
SELECT
    intervention_priority_rank,
    lane_id,
    source_center,
    destination_center,
    route_type,
    trip_count,
    segment_count,
    sla_breach_count,
    sla_breach_rate_pct,
    avg_delay_minutes,
    total_delay_minutes,
    total_projected_cost_of_delay_inr,
    expected_recovery_if_fixed_inr,
    estimated_effort,
    impact_vs_effort_score
FROM prioritized_lanes;

-- ---------------------------------------------------------------------------
-- 4B. Expected INR recovery if the top 20 lanes are fixed
-- ---------------------------------------------------------------------------
-- WHY this exists:
-- Executives need the size of the prize. This query summarizes the intervention
-- backlog into one number that can be compared against staffing, routing, or
-- process-change costs.

CREATE OR REPLACE VIEW v_top20_lane_recovery_summary AS
WITH ranked_lanes AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            ORDER BY intervention_priority_rank, impact_vs_effort_score DESC, expected_recovery_if_fixed_inr DESC, lane_id
        ) AS top_lane_row_number
    FROM v_business_impact_summary_table
),
top20_lanes AS (
    SELECT *
    FROM ranked_lanes
    WHERE top_lane_row_number <= 20
),
summary AS (
    SELECT
        COUNT(*) AS lanes_in_scope,
        SUM(trip_count) AS trips_in_scope,
        SUM(sla_breach_count) AS sla_breaches_in_scope,
        ROUND(SUM(total_projected_cost_of_delay_inr), 0) AS projected_cost_of_delay_inr,
        ROUND(SUM(expected_recovery_if_fixed_inr), 0) AS expected_inr_recovery_if_top20_fixed
    FROM top20_lanes
)
SELECT *
FROM summary;

-- ---------------------------------------------------------------------------
-- 5A. Product Metrics View: North Star framing
-- ---------------------------------------------------------------------------
-- WHY this exists:
-- This is the common scorecard for a PM, operations lead, and data scientist:
-- reliability, delay depth, breach rate, and monetized opportunity.

CREATE OR REPLACE VIEW v_product_metrics_north_star AS
WITH params AS (
    SELECT 150.00 AS assumed_sla_penalty_inr
),
network_metrics AS (
    SELECT
        COUNT(*) AS segment_count,
        COUNT(DISTINCT trip_uuid) AS trip_count,
        COUNT(DISTINCT lane_id) AS active_corridor_count,
        ROUND(100 * AVG(is_on_time), 2) AS on_time_delivery_rate_pct,
        ROUND(AVG(delay_minutes), 2) AS mean_delay_per_segment_minutes,
        ROUND(100 * AVG(is_sla_breach), 2) AS sla_breach_rate_pct,
        SUM(is_sla_breach) AS sla_breach_count
    FROM v_delivery_segment_product_features
),
corridor_delay AS (
    SELECT
        lane_id,
        AVG(delay_minutes) AS corridor_avg_delay_minutes
    FROM v_delivery_segment_product_features
    GROUP BY lane_id
),
corridor_rollup AS (
    SELECT
        ROUND(AVG(corridor_avg_delay_minutes), 2) AS mean_delay_per_corridor_minutes
    FROM corridor_delay
)
SELECT
    network_metrics.segment_count,
    network_metrics.trip_count,
    network_metrics.active_corridor_count,
    network_metrics.on_time_delivery_rate_pct,
    corridor_rollup.mean_delay_per_corridor_minutes,
    network_metrics.mean_delay_per_segment_minutes,
    network_metrics.sla_breach_rate_pct,
    network_metrics.sla_breach_count,
    ROUND(network_metrics.sla_breach_count * params.assumed_sla_penalty_inr, 0) AS projected_sla_cost_inr
FROM network_metrics
CROSS JOIN corridor_rollup
CROSS JOIN params;

-- ---------------------------------------------------------------------------
-- 5B. Executive summary query: Problem -> Diagnosis -> Intervention -> Impact
-- ---------------------------------------------------------------------------
-- WHY this exists:
-- This is the single PM-ready query to paste into a stakeholder readout. It
-- translates operational telemetry into a narrative instead of a table dump.

WITH north_star AS (
    SELECT *
    FROM v_product_metrics_north_star
),
ranked_priority_lane AS (
    SELECT
        lane_id,
        sla_breach_rate_pct,
        trip_count,
        intervention_priority_rank,
        impact_vs_effort_score,
        expected_recovery_if_fixed_inr,
        ROW_NUMBER() OVER (
            ORDER BY intervention_priority_rank, impact_vs_effort_score DESC, expected_recovery_if_fixed_inr DESC, lane_id
        ) AS priority_row_number
    FROM v_business_impact_summary_table
),
highest_risk_lane AS (
    SELECT
        lane_id,
        sla_breach_rate_pct,
        trip_count
    FROM ranked_priority_lane
    WHERE priority_row_number = 1
),
ranked_top_hub AS (
    SELECT
        hub,
        downstream_delay_minutes,
        downstream_sla_breaches,
        downstream_delay_contribution_rank,
        ROW_NUMBER() OVER (
            ORDER BY downstream_delay_contribution_rank, downstream_delay_minutes DESC, hub
        ) AS hub_row_number
    FROM v_hub_downstream_delay_diagnostic
),
top_hub AS (
    SELECT
        hub,
        downstream_delay_minutes,
        downstream_sla_breaches
    FROM ranked_top_hub
    WHERE hub_row_number = 1
),
top20_recovery AS (
    SELECT *
    FROM v_top20_lane_recovery_summary
),
summary_rows AS (
    SELECT
        1 AS story_order,
        'Problem' AS story_stage,
        'SLA breach rate' AS metric_name,
        CONCAT(north_star.sla_breach_rate_pct, '%') AS metric_value,
        CONCAT('Network reliability is at ', north_star.on_time_delivery_rate_pct, '% on-time delivery.') AS business_readout
    FROM north_star

    UNION ALL

    SELECT
        2 AS story_order,
        'Diagnosis' AS story_stage,
        'Top delay hub' AS metric_name,
        top_hub.hub AS metric_value,
        CONCAT(
            'This hub contributes ',
            ROUND(top_hub.downstream_delay_minutes, 0),
            ' downstream delay minutes and ',
            top_hub.downstream_sla_breaches,
            ' SLA breaches.'
        ) AS business_readout
    FROM top_hub

    UNION ALL

    SELECT
        3 AS story_order,
        'Intervention' AS story_stage,
        'Priority lane' AS metric_name,
        highest_risk_lane.lane_id AS metric_value,
        CONCAT(
            'Start with the highest impact-vs-effort lane: ',
            highest_risk_lane.trip_count,
            ' trips and ',
            highest_risk_lane.sla_breach_rate_pct,
            '% breach rate.'
        ) AS business_readout
    FROM highest_risk_lane

    UNION ALL

    SELECT
        4 AS story_order,
        'Impact' AS story_stage,
        'Top 20 lane recovery' AS metric_name,
        CONCAT('INR ', FORMAT(top20_recovery.expected_inr_recovery_if_top20_fixed, 0)) AS metric_value,
        CONCAT(
            'Fixing the top 20 lane backlog addresses ',
            top20_recovery.sla_breaches_in_scope,
            ' observed SLA breaches in this sample window.'
        ) AS business_readout
    FROM top20_recovery
)
SELECT
    story_stage,
    metric_name,
    metric_value,
    business_readout
FROM summary_rows
ORDER BY story_order;
