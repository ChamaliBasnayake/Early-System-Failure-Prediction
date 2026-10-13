# Phase 0 helper: keep conda environments, pip/HF caches and Docker data on the E: drive.
# Run once from PowerShell:  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

$ErrorActionPreference = "Stop"

$dirs = @("E:\conda\envs", "E:\conda\pkgs", "E:\cache\pip", "E:\cache\huggingface", "E:\Docker")
foreach ($d in $dirs) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null; Write-Host "created $d" } }

# conda: put new environments and downloaded packages on E:
conda config --add envs_dirs E:\conda\envs
conda config --add pkgs_dirs E:\conda\pkgs

# user-level environment variables (take effect in NEW terminals)
[Environment]::SetEnvironmentVariable("PIP_CACHE_DIR", "E:\cache\pip", "User")
[Environment]::SetEnvironmentVariable("HF_HOME", "E:\cache\huggingface", "User")

Write-Host ""
Write-Host "Done. Now:"
Write-Host "  1) Close and reopen PowerShell so the environment variables apply."
Write-Host "  2) In Docker Desktop > Settings > Resources > Advanced, set 'Disk image location' to E:\Docker."
Write-Host "  3) Run 'conda info' and confirm the E: paths are listed first."
