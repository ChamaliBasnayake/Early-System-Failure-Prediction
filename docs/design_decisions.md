# Design decisions log

One short entry per decision: what was chosen, the alternative(s), and why.
Q2.a asks for depth, width, normalisation, regularisation, output layer and loss to be justified individually;
Q1 and Q3 reward documented data decisions.

## Phase 0
- **Three fixed seeds (11, 22, 33).** Alternative: one seed. Rejected because the brief requires mean and spread when results vary by seed.
- **CSV experiment log with a guard against logging the test split before Phase 7.** Alternative: free-form notes. Rejected because every reported number must be traceable to a row.
- **CPU-only PyTorch build.** Alternative: GPU. Not available; models are sized accordingly (template-ID sequences, hidden ≤ 256).
- **Existing Anaconda distribution reused; dedicated `bgl` env, caches and Docker data on the E: drive.** Alternative: fresh Miniconda. Unnecessary; isolation comes from the pinned environment file.

## Phase 1
- **Parquet columns written as plain strings, not categoricals.** Alternative: pandas category → Arrow dictionary. Rejected after an int16 dictionary-index overflow when a later chunk exceeded 32,767 distinct nodes; zstd compression keeps the file small anyway.
- **Target defined by behaviour (mass-incident rule), not by category list alone.** Alternative: all KERN* categories. Rejected because KERNSTOR's 63,491 episodes are 8 partition-wide events that would dominate the positive class and are not predictable per node.
- **Mount-failure categories (KERNMNTF, KERNMNT) excluded from the target.** Alternative: keep as a fifth evaluable category. Rejected for misalignment with the node-drain decision; they strengthen the hard-negative pool instead.
- **Support counted in incidents (system-wide, 300 s), not episodes.** Alternative: episode counts. Rejected because synchronous events inflate episode counts by three orders of magnitude.
- **Burst gap 300 s.** Alternative: 60 s. 300 s sits in the valley of the alert-gap histogram and is half the prediction horizon; 60 s risks splitting one storm that pauses briefly into two episodes.
- **Split cut dates placed by incident share (60/20/20).** Alternative: 70/15/15 by calendar days. Rejected because the quiet December–January tail would leave ~45 test incidents, too few for per-category reporting.

## Phase 2
- **Chronological split with cuts at the 60 % / 80 % incident quantiles (train 302 / validation 99 / test 101 incidents).** Alternative: 70/15/15 by days. Rejected: ~45 test incidents.
- **Drain3 fitted on the training period only; UNK for unmatched; transductive fit kept only as a labelled control.** Alternative: fit on all data (most prior work). Rejected as leakage (Le & Zhang 2022; Xiao et al. 2026). Consequence discovered: a BGL control-software upgrade in October 2005 (idoproxydb V1R1M2_254/255) makes 61 % of test-period lines unseen message types; the transductive control shows 0 % UNK and 530 vs 317 templates.
- **Every distinct masked message text stored with an id alongside the frozen template index.** Enables an inductive pretrained text encoder to embed unseen messages (Phase 5) without fitting anything on test data.
- **Unit of prediction changed from node to MIDPLANE; two input channels (unit-local + system-global, 60 min each).** Evidence: precursor diagnostics — the failing node is silent before 90–97 % of its failures, its midplane before 66 %, while the system is active before ~90 % with elevated volume (median 83 vs 12 lines/h). Alternatives: node unit with 10-min window (positives are empty: degenerate), system-only unit (loses the per-unit decision and the held-out-machine experiment). Midplane is also the unit BGL allocates jobs by, so the administrator decision becomes "withhold this midplane from the next allocation".
- **Negatives sampled uniformly over (unit, time), not at event times.** Alternative: activity-weighted sampling at event times. Rejected after it produced non-empty negatives vs. empty positives, a sampling confound that would be learned instead of failure signal.
- **Positive windows at five offsets per unit-episode (k = 0,2,4,6,8 min before the gap).** Gives ~5 training examples per rare episode; all metrics are also reported per incident so this cannot inflate results.
- **Exclusion zones**: failure visible in local window, 20 min after an episode, failure inside the gap, mass-incident lines in window+horizon, span crossing a split cut. Each is counted and reported.

## Phase 3
- **Primary metric PR-AUC; operating points at FAR 0.1 % and 1 % of negative windows (≈1.4 / 14 alarms per midplane-day); per-incident recall.** Alternative: accuracy or ROC-AUC. Rejected: at ~1.5 % prevalence they flatter; the administrator's cost is false alarms per unit per day.
- **Decisions use a strict inequality (score > threshold).** Found when a constant model showed recall 1.0 at "1 % FAR"; ties must never fire.
- **Failure-history features added as a separate group (hours since last unit/system failure, counts over 6 h / 24 h / 7 d, recent mass incidents and non-target alerts).** Alternative: content-only features. Kept both, as an explicit ablation: failures cluster in space and time, and the research question is how much log content adds beyond that clustering. Nothing after the window end t is used.
- **XGBoost budget: 25 random configurations on the full feature set; the best configuration re-fitted on content-only and history-only.** Keeps the tuning budget equal to the neural family while still answering the ablation question.
- **Baseline run repeated after the threshold fix and the history features; the superseded log rows (identical seeds/configs, same PR-AUC) were removed to avoid double counting.**

## Phase 5
- **Comparisons judged against the three-seed spread of v1 (±0.012), not against the seed-11 reference (0.044, an outlier).** Single-seed differences below that are reported as noise.
- **Final ANN = semantic two-channel LSTM with the v1 hyper-parameters (Adam 1e-3, batch 128, 1×64, dropout 0.3, LayerNorm, last state, √ratio weighting, 60-min window).** Alternatives within noise: hidden 128, 2 layers, mean+max pooling, dropout 0.5, AdamW. Chosen on mechanism, before any test evaluation: equal validation performance, and the only model able to represent the 61 % of test-period lines that the template model maps to UNK. The template LSTM v1 remains the "initial ANN".
- **Unsuccessful attempts kept and diagnosed:** lr 1e-4 / 3e-4 (under-fitting within the patience window), 6 h / 24 h local windows (dilution of the informative last hour), 1-D CNN (worst ROC: evidence is message identity, not local patterns), SGD (best bulk ranking / lowest validation loss but nothing at the head of the ranking: recall@1 % FAR = 0).
- **Wrong expectations recorded:** lr 1e-2 did not diverge (gradient clipping + LayerNorm); full-ratio class weighting did not over-fire; both are reported as such.


## Phase 6 — generalisation experiments and protocol controls (validation)
- **D6.1 Leave-one-category-out by removing the category's training positives only** (`train.py --exclude-category`). Negatives and other categories unchanged, so the only difference to the all-category model is the absence of that failure type. Evaluated on the held-out category's positives vs all negatives.
- **D6.2 Machine-disjoint by a seeded 20 % hold-out of racks from training** (`--holdout-racks 0.2`); rack list saved in the checkpoint. The all-rack model is scored on the same windows so the comparison isolates "never saw this rack".
- **D6.3 Thresholds frozen from validation** into `results/thresholds_final.json` by the validation run of `evaluate_conditions.py`; the test run reads them and refuses to run twice.
- **D6.4 Random-split and transductive-template controls are comparison-only** (`outcome=control` in the log) and never influence a model decision. The transductive control is additionally scored on test in Phase 7 because its effect is expected only where UNK is high.
- **D6.5 Per-category table reports incident recall with the incident count** so that single-incident categories (KERNRTSP on validation) are not over-read.
Findings: `docs/phase6_findings.md`.
