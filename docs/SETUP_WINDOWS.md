# Phase 0 — Setup on Windows 11

Follow these steps once. Allow about two hours including downloads.

## 1. Tools

| Tool | Notes |
|---|---|
| Anaconda (already installed) | Used only to create the `bgl` environment; never work in `base`. |
| Git for Windows | https://git-scm.com/download/win |
| Node.js 20 LTS | https://nodejs.org — needed from Phase 9 (React frontend). |
| Docker Desktop | https://www.docker.com/products/docker-desktop/ — enable WSL 2 when asked; needed from Phase 9. |
| VS Code + Python + Jupyter extensions | https://code.visualstudio.com |

Check in a new PowerShell window: `conda --version`, `git --version`, `node --version`, `docker --version`.

## 2. Keep everything off the C: drive

Run `scripts\setup_windows.ps1` from PowerShell (right-click → Run with PowerShell, or
`powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1`). It:

* points conda's environment and package folders to `E:\conda\envs` and `E:\conda\pkgs`;
* sets `PIP_CACHE_DIR=E:\cache\pip` and `HF_HOME=E:\cache\huggingface` (pretrained models from Phase 5 go here);
* creates `E:\Docker` for Docker's disk image.

Then, in Docker Desktop → Settings → Resources → Advanced, set **Disk image location** to `E:\Docker`
(do this before pulling any image; Docker moves the image for you).

Verify: `conda info` shows the E: paths under *envs directories* and *package cache*.

## 3. Repository

```powershell
cd E:\Projects\bgl-failure-prediction
git init
git add .
git commit -m "Phase 0: repository skeleton, environment, seeds, experiment log"
git branch -M main
git remote add origin https://github.com/<your-username>/bgl-failure-prediction.git
git push -u origin main
```

## 4. Environment

```powershell
conda env create -f environment.yml      # ~1 GB download, 10–20 min
conda activate bgl                       # prompt must show (bgl)
```

If `conda activate` fails in PowerShell: `conda init powershell`, close and reopen the window.
If Anaconda is older than 2023: `conda update -n base conda` first.

## 5. Smoke tests

```powershell
pytest -q        # expect: 5 passed
```

## 6. Read before moving on

`src/utils/seeds.py`, `src/utils/runlog.py`, `README.md`. Be able to explain why four random
number generators are seeded and why the log refuses `split="test"` before Phase 7.

## 7. Daily habit

```powershell
git add -A; git commit -m "<what you did>"; git push
```
