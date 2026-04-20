# Run all three RAM pricing case studies sequentially.
# Assumes the wisk virtual environment is already activated.
#
# Usage (from repo root):
#   .\run_all_studies.ps1
#   .\run_all_studies.ps1 -OptimizerConfig configs/optimizer.json

param(
    [string]$OptimizerConfig = "configs/optimizer.json"
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Case studies: (city_pair, vertiport_config)
# ---------------------------------------------------------------------------
$studies = @(
    @{
        CityPair        = "UFL_Orlando_Thu"
        VertiportConfig = "external/replica_data_analytics/config/vertiport_configuration/UFL.json"
    },
    @{
        CityPair        = "Chicago_UIUC_Thu"
        VertiportConfig = "external/replica_data_analytics/config/vertiport_configuration/UIUC.json"
    },
    @{
        CityPair        = "TAMU_Houston_Thu"
        VertiportConfig = "external/replica_data_analytics/config/vertiport_configuration/TAMU.json"
    }
)

# ---------------------------------------------------------------------------
$totalStart = Get-Date
$results    = @()

foreach ($study in $studies) {
    $label = $study.CityPair
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $label" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan

    $start = Get-Date
    $exitCode = 0

    try {
        python run_pricing.py `
            --city_pair        $study.CityPair `
            --vertiport_config $study.VertiportConfig `
            --optimizer_config $OptimizerConfig

        if ($LASTEXITCODE -ne 0) { $exitCode = $LASTEXITCODE }
    } catch {
        Write-Host "ERROR: $_" -ForegroundColor Red
        $exitCode = 1
    }

    $elapsed = (Get-Date) - $start
    $status  = if ($exitCode -eq 0) { "OK" } else { "FAILED" }
    $color   = if ($exitCode -eq 0) { "Green" } else { "Red" }

    Write-Host ""
    Write-Host "  [$label]  $status  ($([int]$elapsed.TotalMinutes) min $($elapsed.Seconds) sec)" -ForegroundColor $color

    $results += [PSCustomObject]@{
        Study   = $label
        Status  = $status
        Minutes = [math]::Round($elapsed.TotalMinutes, 1)
    }
}

# ---------------------------------------------------------------------------
$totalElapsed = (Get-Date) - $totalStart
Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "  SUMMARY" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
foreach ($r in $results) {
    $color = if ($r.Status -eq "OK") { "Green" } else { "Red" }
    Write-Host ("  {0,-30} {1}  ({2} min)" -f $r.Study, $r.Status, $r.Minutes) -ForegroundColor $color
}
Write-Host ""
Write-Host ("  Total time: {0} min {1} sec" -f [int]$totalElapsed.TotalMinutes, $totalElapsed.Seconds) -ForegroundColor Cyan
Write-Host ""
