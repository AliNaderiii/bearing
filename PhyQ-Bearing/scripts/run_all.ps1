# Full pipeline on Windows (PowerShell). Resumable: re-running skips finished runs.
#   .\scripts\run_all.ps1              -> everything
#   .\scripts\run_all.ps1 -Seeds "0"   -> quick first pass with one seed
param([string]$Seeds = "")
Set-Location (Split-Path -Parent $PSScriptRoot)
# Always use the project's .venv Python (works even if you forgot to activate it)
$py = "python"
if (Test-Path ".venv\Scripts\python.exe") { $py = (Resolve-Path ".venv\Scripts\python.exe").Path }
Write-Host "Using Python: $py" -ForegroundColor DarkGray
$seedArgs = @()
if ($Seeds -ne "") { $seedArgs = @("--seeds") + $Seeds.Split(",") }

function Step($title, $cmd) {
    Write-Host "`n==================== $title ====================" -ForegroundColor Cyan
    & $py @cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nFAILED: $title  -> stopped here (nothing after this step was run)." -ForegroundColor Red
        Write-Host "Fix the message above (usually configs\paths.yaml), then run this script again." -ForegroundColor Red
        exit 1
    }
}
# 0) pre-flight: stop in seconds (not hours) if a dataset path is wrong
# The proposed model is PhyQ-TransNet exactly as designed by the author (configs/models/phyq_paper.yaml);
# no architecture search is run any more (scripts/search.py remains only as an optional tool).
Step "0/5 environment + paths"   @("scripts/check_env.py")
Step "0/5 data sanity check"     @("scripts/sanity_check_data.py", "cwru", "pu", "ims", "nrel", "sdust")
Step "1/5 unit tests"            @("-m", "pytest", "-q", "tests")
Step "2/5 main benchmark"        (@("scripts/run_suite.py", "configs/experiments/main_benchmark.yaml") + $seedArgs)
Step "3/5 CWRU cross-load"       (@("scripts/run_suite.py", "configs/experiments/cross_load.yaml") + $seedArgs)
Step "4/5 ablation"              (@("scripts/run_suite.py", "configs/experiments/ablation.yaml") + $seedArgs)
foreach ($e in "main_benchmark", "cross_load", "ablation") {
    Step "5/5 tables: $e" @("scripts/make_tables.py", $e)
}
& $py scripts/profile_models.py | Out-File -Encoding utf8 results/complexity.md
Write-Host "`nDONE. Send me the 'results' folder (zip it)." -ForegroundColor Green
