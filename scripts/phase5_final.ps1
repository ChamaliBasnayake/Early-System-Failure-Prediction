# Phase 5 (final) — train the FINAL ANN (semantic two-channel LSTM, v1 hyper-parameters) for all lead times × 3 seeds,
# and the INITIAL ANN (template LSTM v1) for the other lead times (seed 11; N=5 already has three seeds).
# Decision recorded before any test evaluation: validation performance of template and semantic models is equal
# within seed spread; the semantic model is chosen as final because it alone can represent message types unseen in
# training (61 % of test-period lines). Run inside the (bgl) env:  Set-ExecutionPolicy -Scope Process Bypass; .\scripts\phase5_final.ps1
$ErrorActionPreference = "Continue"
if (-not $env:CONDA_PREFIX) { Write-Host "Run 'conda activate bgl' first." -ForegroundColor Red; exit 1 }
$py = Join-Path $env:CONDA_PREFIX "python.exe"
foreach ($N in 1, 5, 15, 30) {
  foreach ($seed in 11, 22, 33) {
    Write-Host "`n=== FINAL sem_final N=$N seed=$seed ===" -ForegroundColor Cyan
    & "$py" -m src.models.train --config configs/lstm_semantic_v1.yaml --phase 5 --name sem_final --N $N --seed $seed --purpose "final ANN (semantic LSTM), N=$N seed=$seed" --changed "final configuration = v1 hyper-parameters + frozen MiniLM text embeddings"
  }
}
foreach ($N in 1, 15, 30) {
  Write-Host "`n=== INITIAL lstm_v1 N=$N seed=11 ===" -ForegroundColor Cyan
  & "$py" -m src.models.train --config configs/lstm_v1.yaml --phase 5 --name lstm_v1 --N $N --seed 11 --purpose "initial ANN (template LSTM v1) at N=$N for the earliness curve" --changed "N only"
}
Write-Host "`nFinal models trained. Next: Phase 6 (generalisation experiments on validation) and Phase 7 (single test evaluation)." -ForegroundColor Green
