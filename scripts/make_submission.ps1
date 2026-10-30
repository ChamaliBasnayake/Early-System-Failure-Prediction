# Build the SUBMISSION folder: the repository plus the small artefacts the web app and the test evaluation need,
# minus raw data, caches, virtual environments and node_modules. Run inside (bgl) from the repository root:
#   Set-ExecutionPolicy -Scope Process Bypass; .\scripts\make_submission.ps1 -Out E:\submission\Early-System-Failure-Prediction
param([string]$Out = "E:\submission\Early-System-Failure-Prediction")
$ErrorActionPreference = "Stop"
if (Test-Path $Out) { Remove-Item $Out -Recurse -Force }
New-Item -ItemType Directory -Path $Out | Out-Null
# 1) code, configs, docs, results, tests, report (git-tracked content)
$exclude = @(".git", "node_modules", "__pycache__", ".pytest_cache", "dist", "app.db")
foreach ($d in "app", "configs", "docs", "experiments", "report", "results", "scripts", "src", "tests", "data\splits", "data\sample") {
  robocopy ".\$d" "$Out\$d" /E /XD $exclude /XF "*.pyc" "app.db" /NFL /NDL /NJH /NJS /NC /NS | Out-Null
}
foreach ($f in "README.md", "environment.yml", "docker-compose.yml", "pytest.ini", ".gitignore") { if (Test-Path $f) { Copy-Item $f $Out } }
# 2) artefacts required at request time by the application and by the evaluation scripts (NOT git-tracked: rebuilt by the pipeline)
New-Item -ItemType Directory -Path "$Out\data\interim" -Force | Out-Null
foreach ($f in "drain_miner.pkl", "masked_messages.parquet", "masked_embeddings_minilm.npy", "templates.csv") {
  if (Test-Path "data\interim\$f") { Copy-Item "data\interim\$f" "$Out\data\interim\" } else { Write-Host "MISSING data\interim\$f" -ForegroundColor Red }
}
New-Item -ItemType Directory -Path "$Out\results\checkpoints" -Force | Out-Null
foreach ($r in "sem_final_N1_s11", "sem_final_N5_s11", "sem_final_N15_s11", "sem_final_N30_s11", "sem_final_N5_s22", "sem_final_N5_s33", "lstm_v1_N5_s11") {
  if (Test-Path "results\checkpoints\$r.pt") { Copy-Item "results\checkpoints\$r.pt" "$Out\results\checkpoints\" } else { Write-Host "MISSING checkpoint $r" -ForegroundColor Yellow }
}
# 3) never ship secrets or the raw 743 MB log (download script + 3 MB sample are included)
Get-ChildItem $Out -Recurse -File | Select-String -Pattern "AIza[0-9A-Za-z_-]{30,}" -List | ForEach-Object { Write-Host "SECRET-LIKE STRING in $($_.Path)" -ForegroundColor Red }
Remove-Item "$Out\notebooks" -Recurse -Force -ErrorAction SilentlyContinue
$size = (Get-ChildItem $Out -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Submission folder built: {0}  ({1:N0} MB)" -f $Out, $size) -ForegroundColor Green
Write-Host "Add to it: the report PDF, the demonstration video, and the approval correspondence."
