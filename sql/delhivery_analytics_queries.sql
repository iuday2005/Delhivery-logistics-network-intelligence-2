-- Delhivery Logistics Network Intelligence
-- SQL analytics examples for logistics and product analysis.
-- Assumed table: delivery_segments
-- Columns mirror data/delivery_data.csv.

-- 1. Trip-level ETA error by route type.
SELECT
    route_type,
    COUNT(DISTINCT trip_uuid) AS trips,
    AVG(ABS(actual_time - osrm_time)) AS avg_eta_error_min,
    AVG(actual_time - osrm_time) AS avg_eta_bias_min
FROM delivery_segments
WHERE actual_time IS NOT NULL
  AND osrm_time IS NOT NULL
GROUP BY route_type
ORDER BY avg_eta_error_min DESC;

-- 2. Highest-delay corridors with minimum volume filter.
SELECT
    source_center,
    destination_center,
    route_type,
    COUNT(*) AS segment_count,
    COUNT(DISTINCT trip_uuid) AS trips,
    AVG(actual_time / NULLIF(osrm_time, 0)) AS avg_delay_ratio,
    AVG(CASE WHEN actual_time > 2 * osrm_time THEN 1 ELSE 0 END) AS severe_delay_rate
FROM delivery_segments
WHERE actual_time > 0
  AND osrm_time > 0
GROUP BY source_center, destination_center, route_type
HAVING COUNT(DISTINCT trip_uuid) >= 5
ORDER BY severe_delay_rate DESC, avg_delay_ratio DESC;

-- 3. Hub-level outgoing delay risk.
SELECT
    source_center AS hub,
    COUNT(*) AS outgoing_segments,
    COUNT(DISTINCT destination_center) AS outgoing_corridors,
    AVG(segment_actual_time / NULLIF(segment_osrm_time, 0)) AS outgoing_delay_ratio,
    AVG(CASE WHEN segment_actual_time > 2 * segment_osrm_time THEN 1 ELSE 0 END) AS outgoing_severe_rate
FROM delivery_segments
WHERE segment_actual_time > 0
  AND segment_osrm_time > 0
GROUP BY source_center
HAVING COUNT(*) >= 20
ORDER BY outgoing_severe_rate DESC, outgoing_delay_ratio DESC;

-- 4. Time-window delay pattern for dispatch planning.
SELECT
    EXTRACT(HOUR FROM trip_creation_time) AS trip_hour,
    route_type,
    COUNT(DISTINCT trip_uuid) AS trips,
    AVG(actual_time / NULLIF(osrm_time, 0)) AS avg_delay_ratio
FROM delivery_segments
WHERE actual_time > 0
  AND osrm_time > 0
GROUP BY EXTRACT(HOUR FROM trip_creation_time), route_type
ORDER BY trip_hour, route_type;

-- 5. Candidate control corridors for an A/B intervention design.
WITH corridor_stats AS (
    SELECT
        source_center,
        destination_center,
        route_type,
        COUNT(DISTINCT trip_uuid) AS trips,
        AVG(actual_time / NULLIF(osrm_time, 0)) AS avg_delay_ratio,
        AVG(actual_distance_to_destination) AS avg_distance
    FROM delivery_segments
    WHERE actual_time > 0
      AND osrm_time > 0
    GROUP BY source_center, destination_center, route_type
)
SELECT *
FROM corridor_stats
WHERE trips >= 5
ORDER BY route_type, avg_distance, avg_delay_ratio;
