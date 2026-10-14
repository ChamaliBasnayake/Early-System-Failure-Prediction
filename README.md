# Early Failure Prediction from System Logs (BlueGene/L)

A leakage-controlled benchmark for predicting compute-node failures from the
minutes of log messages that precede them, with generalisation tests to unseen
failure types and unseen machines, and a fair comparison of a tuned classical
baseline, a sequence neural network and a foundation model (Gemini).

**User:** a system administrator operating a high-performance computing cluster.
**Decision:** when the model predicts a failure within the next N minutes, the
administrator drains or checkpoints the jobs on that node and removes it from
scheduling, instead of discovering the failure after it happens.

## Status

| Phase | State |
|---|---|
| 0 Environment & repository | done |
| 1 Data acquisition & EDA | done |
| 2 Dataset construction & split | in progress |
| 3 Baselines | — |
| 4 First neural model | — |
| 5 Optimisation study | — |
| 6 Generalisation experiments | — |
| 7 Final evaluation (test, once) | — |
| 8 Explainable AI | — |
| 9 Web application | — |
| 10 Generative AI comparison | — |
| 11 Report, video, presentation | — |

## Reproduce

Windows setup (tools, E: drive layout, Docker disk location): see `docs/SETUP_WINDOWS.md`.

```bash
conda env create -f environment.yml
conda activate bgl
pytest -q                         # Phase 0 smoke tests
# Phase 1 — data
python -m src.data.download           # BGL.zip from Zenodo, MD5 check, unzip (~700 MB)
python -m src.data.parse_bgl          # → data/interim/bgl_parsed.parquet
python -m src.data.eda                # → results/eda_summary.md + results/figures/eda_*.png
python -m src.data.failure_set        # → data/interim/failure_episodes.parquet + results/failure_set_summary.md
python -m src.data.make_sample        # → data/sample/bgl_sample.log (20k lines, committed)
python -m src.data.parse_bgl --sample && python -m src.data.eda --sample && python -m src.data.failure_set --sample
```

## Repository layout

```
configs/        experiment and pipeline configuration (YAML / INI)
data/raw        BGL.log — downloaded by src/data/download.py (git-ignored)
data/interim    parsed tables, templates, windows (git-ignored)
data/splits     chronological split definitions (committed)
data/sample     ~20k-line representative sample so the code can be tested
docs/           failure_definition.md, eda_findings.md, preprocessing.md, design_decisions.md
src/data        download, parse, templates, windows, splits, leakage checks
src/models      baselines, lstm, cnn, train, evaluate_final
src/xai         integrated gradients, perturbation checks
src/genai       Gemini evaluation and prompts
src/utils       seeds, run log, metrics, paths
experiments/    log.csv — THE experiment log (every run, including failures)
results/        metrics, curves, figures, final_test_results.json
app/backend     FastAPI service      app/frontend  React (Vite) client
report/         report source and figures
```

## Project rules (from the coursework brief)

1. One chronological train/validation/test split, created once in Phase 2 and reused by every model and the LLM.
2. The test set is read by exactly one script (`src/models/evaluate_final.py`), run once, output committed.
3. Every training/evaluation run appends a row to `experiments/log.csv` via `src.utils.runlog.log_run`.

## Dataset

BlueGene/L logs from the Loghub collection (Zenodo DOI 10.5281/zenodo.8196385), CC BY 4.0.
Cite: Zhu et al., "Loghub", ISSRE 2023; Oliner & Stearley, "What Supercomputers Say", DSN 2007.

## AI-tool declaration

See `docs/ai_tool_declaration.md` (to be completed; required by Section 7 of the brief).
