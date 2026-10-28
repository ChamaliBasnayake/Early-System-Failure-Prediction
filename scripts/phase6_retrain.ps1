# Phase 6 — retraining for the generalisation experiments (validation only) + protocol controls.
#   leave-one-category-out: final config trained WITHOUT each evaluable category's positives (4 runs, N=5, seed 11)
#   machine-disjoint:       final config trained with 20 % of racks held out (1 run)
#   controls:               random-split XGBoost, transductive-template windows/features/XGBoost (comparison only)
# Run inside (bgl):  Set-ExecutionPolicy -Scope Process Bypass; .\scripts\phase6_retrain.ps1
$ErrorActionPreference = "Continue"
if (-not $env:CONDA_PREFIX) { Write-Host "Run 'conda activate bgl' first." -ForegroundColor Red; exit 1 }
$py = Join-Path $env:CONDA_PREFIX "python.exe"
foreach ($cat in "KERNMC", "KERNRTSP", "KERNSOCK", "KERNDTLB") {
  Write-Host "`n=== leave-one-category-out: without $cat ===" -ForegroundColor Cyan
  & "$py" -m src.models.train --config configs/lstm_semantic_v1.yaml --phase 6 --name "sem_final_no$cat" --N 5 --seed 11 --exclude-category $cat --purpose "held-out failure category $cat" --changed "training positives of $cat removed"
}
Write-Host "`n=== machine-disjoint: 20 % of racks held out ===" -ForegroundColor Cyan
& "$py" -m src.models.train --config configs/lstm_semantic_v1.yaml --phase 6 --name "sem_final_rackout" --N 5 --seed 11 --holdout-racks 0.2 --purpose "held-out racks (machine-disjoint)" --changed "20 % of racks removed from training"
Write-Host "`n=== protocol control: random window split (XGBoost) ===" -ForegroundColor Cyan
& "$py" -m src.models.baselines --random-split-control --n_configs 10
Write-Host "`n=== protocol control: transductive templates (windows, features, XGBoost) ===" -ForegroundColor Cyan
& "$py" -m src.data.windows --events _tc --tag tc
& "$py" -m src.features.build_features --tag tc --templates _tc
& "$py" -m src.models.baselines --tag tc --n_configs 10
Write-Host "`nPhase 6 retraining done. Now:  python -m src.models.evaluate_conditions --split validation" -ForegroundColor Green
