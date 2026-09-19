param(
    [string]$MysqlExe = "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe",
    [string]$Database = "delhivery",
    [string]$User = "root",
    [string]$OutputDir = "D:\iit g\projects\delihivery\sql\product_analytics_layer\exports"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $MysqlExe)) {
    throw "mysql.exe not found at $MysqlExe"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$queries = @(
    @{
        Name = "product_metrics_north_star.tsv"
        Sql = "SELECT * FROM v_product_metrics_north_star;"
    },
    @{
        Name = "top10_highest_risk_lanes_by_sla_breach.tsv"
        Sql = "SELECT * FROM v_top10_highest_risk_lanes_by_sla_breach;"
    },
    @{
        Name = "corridor_delay_frequency_rank.tsv"
        Sql = "SELECT * FROM v_corridor_delay_frequency_rank;"
    },
    @{
        Name = "corridor_week_over_week_delay_trend.tsv"
        Sql = "SELECT * FROM v_corridor_week_over_week_delay_trend;"
    },
    @{
        Name = "sla_breach_cohort_analysis.tsv"
        Sql = "SELECT * FROM v_sla_breach_cohort_analysis;"
    },
    @{
        Name = "monthly_on_time_corridor_retention.tsv"
        Sql = "SELECT * FROM v_monthly_on_time_corridor_retention;"
    },
    @{
        Name = "hub_downstream_delay_diagnostic.tsv"
        Sql = "SELECT * FROM v_hub_downstream_delay_diagnostic;"
    },
    @{
        Name = "hub_efficiency_drop_alerts.tsv"
        Sql = "SELECT * FROM v_hub_efficiency_drop_alerts;"
    },
    @{
        Name = "business_impact_summary_table.tsv"
        Sql = "SELECT * FROM v_business_impact_summary_table;"
    },
    @{
        Name = "top20_lane_recovery_summary.tsv"
        Sql = "SELECT * FROM v_top20_lane_recovery_summary;"
    }
)

foreach ($query in $queries) {
    $outFile = Join-Path $OutputDir $query.Name
    & $MysqlExe -u $User -p --batch --raw $Database -e $query.Sql | Set-Content -Path $outFile -Encoding UTF8
    Write-Host "Exported $outFile"
}

Write-Host "Done. Results are in $OutputDir"
