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
| 2 Dataset construction & split | done |
| 3 Baselines | done |
| 4 First neural model | done |
| 5 Optimisation study | done |
| 6 Generalisation experiments | done |
| 7 Final evaluation (test, once) | — |
| 8 Explainable AI | in progress |
| 9 Web application | in progress |
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

# Phase 2 — dataset construction (one split, templates fitted on train only, windows with gap, leakage report)
python -m src.data.splits             # → data/splits/split_v1.json (+ rollA, rollB), results/split_report.md
python -m src.data.templates          # → data/interim/drain_miner.pkl, templates.csv, events.parquet, results/template_report.md
python -m src.data.windows            # → data/interim/windows.parquet, results/windows_summary.md, results/class_distribution.csv
python -m src.data.leakage_checks     # → results/leakage_report.txt (exits non-zero on failure)
python -m src.data.unk_diagnostics        # which messages are unseen after the training cut (software-upgrade finding)
python -m src.data.precursor_diagnostics  # is there log content before failures, and at which hardware level?
python -m src.data.window_diagnostics     # content by class and channel
python -m src.data.templates --transductive-control   # leaky protocol of prior work — CONTROL ONLY (outputs *_tc)

# Phase 3 — baselines (validation only; every configuration appended to experiments/log.csv)
python -m src.features.build_features   # → data/interim/features.parquet
python -m src.models.baselines          # trivial, confound LR, 25-config XGBoost search per N → results/baselines_validation.csv

# Phase 3b — where does predictability live? SYSTEM-level variant (label: any target failure in the horizon)
python -m src.data.windows --unit system --tag system && python -m src.features.build_features --tag system --unit system
python -m src.models.baselines --tag system --n_configs 10
python -m src.models.feature_importance   # what the content model uses (template text, direction, group shares)
# Phase 3c — local window length study (the lecturer-approved 4th hyperparameter): W_local = 1 h (primary), 6 h, 24 h
python -m src.data.windows --local-window 360  --tag w6h  && python -m src.features.build_features --tag w6h  && python -m src.models.baselines --tag w6h  --n_configs 10
python -m src.data.windows --local-window 1440 --tag w24h && python -m src.features.build_features --tag w24h && python -m src.models.baselines --tag w24h --n_configs 10
# Phase 4 — first neural model (two-channel LSTM), three seeds, learning curves
python -m src.models.train --config configs/lstm_v1.yaml --seed 11
python -m src.models.train --config configs/lstm_v1.yaml --seed 22
python -m src.models.train --config configs/lstm_v1.yaml --seed 33
python -m src.models.plot_curves --runs lstm_v1_N5_s11 lstm_v1_N5_s22 lstm_v1_N5_s33 --out lstm_v1_N5
# Phase 5 — optimisation study (overnight batch; ~24 runs; validation only) + variants
python -m src.models.embed_messages                       # frozen MiniLM embeddings for every masked text (semantic variant)
powershell -ExecutionPolicy Bypass -File scripts\phase5_study.ps1
python -m src.models.study_summary                        # → results/optimisation_study.md + results/figures/study_*.png
# Phase 8 — explainable AI (Integrated Gradients; validated by perturbation; incorrect-prediction cases)
python -m src.xai.perturbation --run sem_final_N5_s11 --n 50       # → results/xai_perturbation.md + figures/xai_perturbation.png
python -m src.xai.cases --run sem_final_N5_s11                     # → results/xai_cases.md (write your interpretations in)
python scripts/make_demo_inputs.py --run sem_final_N5_s11          # → results/demo/*.log for the app demo and video
# Phase 9 — web application (development mode: SQLite; production: docker compose with PostgreSQL)
uvicorn app.backend.main:app --reload --port 8000        # backend, from the project root (loads Drain state + checkpoint once)
cd app/frontend && npm install && npm run dev            # frontend at http://localhost:5173 (proxies /api to :8000)
pytest -q tests/test_phase9.py                           # API tests (normal, invalid, empty, unknown midplane, feedback, history)
docker compose up --build                                # full stack: Postgres + backend (:8000) + nginx frontend (:8080)
# Phase 6 — generalisation experiments (validation) and protocol controls
powershell -ExecutionPolicy Bypass -File scripts\phase6_retrain.ps1     # or: Set-ExecutionPolicy -Scope Process Bypass; .\scripts\phase6_retrain.ps1
python -m src.models.evaluate_conditions --split validation             # → results/validation_eval/, freezes results/thresholds_final.json
# Phase 10 — Gemini as a predictor: prompt selection on VALIDATION (needs GEMINI_API_KEY user env var; new terminal)
python -m src.genai.gemini_eval --list-models                                # models available to this key
python -m src.genai.gemini_eval --split validation --prompts p1 --limit 10   # dry run: 10 calls (default model gemini-3.8-flash; --model to change)
python -m src.genai.gemini_eval --split validation                           # 4 prompts × 300 windows (cached; resumable)
python -m src.genai.gemini_eval --select p2                                  # freeze the prompt chosen from results/genai/gemini_validation_N5.md
# Phase 7 — THE single test evaluation (after Phase 10 prompt selection on validation)
python -m src.genai.gemini_eval --split test                                 # selected prompt only; refuses to run twice
python -m src.models.evaluate_conditions --split test                   # → results/test_eval/ (refuses to run twice)
python -m src.models.shift_analysis                                     # post-hoc descriptive: why validation and test differ
python -m src.models.error_buckets --run-prefix sem_final --N 5         # error analysis by input type / data quality (validation)
python -m src.models.plot_curves --runs sem_final_N5_s11 sem_final_N5_s22 sem_final_N5_s33 --out sem_final_N5   # final-model learning curves
python -m src.utils.apply_annotations                                   # apply experiments/annotations.csv (human 'learned' statements) to the log
# Protocol controls (comparison only, never for model selection)
python -m src.models.baselines --random-split-control --n_configs 10              # leaky random window split
python -m src.data.windows --events _tc --tag tc && python -m src.features.build_features --tag tc --templates _tc
python -m src.models.baselines --tag tc --n_configs 10                            # leaky transductive templates
# same four with --sample to exercise the pipeline on the committed sample
```

## Submission package
`scripts\make_submission.ps1` builds the folder that is submitted: all code, configs, docs, results, tests, the 3 MB sample log with sample splits,
plus the small runtime artefacts the application needs (`data/interim/{drain_miner.pkl, masked_messages.parquet, masked_embeddings_minilm.npy, templates.csv}`
and the final checkpoints in `results/checkpoints/`). The raw 743 MB log is not shipped; `python -m src.data.download` fetches it (Zenodo, CC BY 4.0)
and every pipeline step can be exercised on the sample with `--sample`. Phase 7 numbers are reproducible from the shipped checkpoints and scores with
`python -m src.models.evaluate_conditions --split test` after deleting `results/test_eval/final_test_results.json` (the guard that enforces the single read).

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
