# Phase 5 — optimisation study. Run from the project root inside the (bgl) environment:
#   conda activate bgl;  Set-ExecutionPolicy -Scope Process Bypass;  .\scripts\phase5_study.ps1
# Each run is logged (phase 5) with its name, purpose and what changed. Roughly 24 runs × 10-20 min on CPU.
# Everything here is selected and evaluated on VALIDATION only. The test set is not touched.
$ErrorActionPreference = "Continue"
# Use the Python of the ACTIVE conda environment (a fresh PowerShell would otherwise fall back to base, which has no torch).
if (-not $env:CONDA_PREFIX) { Write-Host "No conda environment active. Run 'conda activate bgl' first, then:  .\scripts\phase5_study.ps1" -ForegroundColor Red; exit 1 }
$py = Join-Path $env:CONDA_PREFIX "python.exe"
Write-Host "Using $py" -ForegroundColor DarkGray
function Run($name, $purpose, $changed, $extra) {
  Write-Host "`n=== $name : $purpose ===" -ForegroundColor Cyan
  $cmd = "& `"$py`" -m src.models.train --config configs/lstm_v1.yaml --phase 5 --name $name --purpose `"$purpose`" --changed `"$changed`" $extra"
  Write-Host $cmd; Invoke-Expression $cmd
}
# 1. Learning rate (reference 1e-3 = lstm_v1 seeds 11/22/33 from Phase 4)
Run "lr_1e-4"  "learning-rate study" "lr 1e-3 -> 1e-4" "--set train.lr=0.0001"
Run "lr_3e-4"  "learning-rate study" "lr 1e-3 -> 3e-4" "--set train.lr=0.0003"
Run "lr_3e-3"  "learning-rate study" "lr 1e-3 -> 3e-3" "--set train.lr=0.003"
Run "lr_1e-2"  "learning-rate study (expected to diverge)" "lr 1e-3 -> 1e-2" "--set train.lr=0.01"
# 2. Batch size (reference 128)
Run "bs_32"    "batch-size study" "batch 128 -> 32"  "--set train.batch_size=32"
Run "bs_512"   "batch-size study" "batch 128 -> 512" "--set train.batch_size=512"
# 3. Depth and width (reference 1 layer x 64)
Run "layers_2"   "depth study" "1 -> 2 LSTM layers" "--set model.layers=2"
Run "hidden_32"  "width study" "hidden 64 -> 32"   "--set model.hidden=32"
Run "hidden_128" "width study" "hidden 64 -> 128"  "--set model.hidden=128"
Run "pool_meanmax" "pooling alternative" "last state -> mean+max pooling" "--set model.pooling=meanmax"
# 4. Regularisation (reference dropout 0.3)
Run "drop_0.1"  "regularisation study" "dropout 0.3 -> 0.1" "--set model.dropout=0.1"
Run "drop_0.5"  "regularisation study" "dropout 0.3 -> 0.5" "--set model.dropout=0.5"
Run "no_layernorm" "normalisation ablation" "LayerNorm removed" "--set model.layernorm=false"
# 5. Optimisers (reference Adam 1e-3)
Run "sgd_0.01"  "optimiser study" "Adam -> SGD momentum 0.9, lr 0.01" "--set train.optimizer=sgd --set train.lr=0.01"
Run "sgd_0.1"   "optimiser study" "Adam -> SGD momentum 0.9, lr 0.1"  "--set train.optimizer=sgd --set train.lr=0.1"
Run "adamw_wd1e-2" "optimiser study" "Adam -> AdamW, weight decay 1e-2" "--set train.optimizer=adamw --set train.weight_decay=0.01"
# 6. Class weighting (reference sqrt ratio ~ 7.9)
Run "posw_ratio" "loss weighting (expected to over-fire)" "pos_weight sqrt -> full ratio (~65)" "--set train.pos_weight=ratio"
Run "posw_1"     "loss weighting" "pos_weight sqrt -> none (1)" "--set train.pos_weight=1"
# 7. Local window length (lecturer-approved 4th hyper-parameter; windows/features built in Phase 3c)
Run "w6h"   "window-length study" "local window 60 min -> 6 h"  "--tag w6h"
Run "w24h"  "window-length study" "local window 60 min -> 24 h" "--tag w24h"
# 8. Architecture alternatives
Write-Host "`n=== cnn_v1 : architecture alternative ===" -ForegroundColor Cyan
& "$py" -m src.models.train --config configs/cnn_v1.yaml --phase 5 --name cnn_v1 --purpose "architecture alternative" --changed "LSTM -> 1-D CNN (kernels 3/5/7 x 64), masked max-pool"
Write-Host "`n=== sem_v1 : transfer-learning variant (frozen MiniLM text embeddings) ===" -ForegroundColor Cyan
& "$py" -m src.models.train --config configs/lstm_semantic_v1.yaml --phase 5 --name sem_v1 --purpose "transfer-learning variant" --changed "learned template embedding -> frozen MiniLM text embeddings projected to 64-d"
Write-Host "`nStudy finished. Now run:  python -m src.models.study_summary" -ForegroundColor Green
