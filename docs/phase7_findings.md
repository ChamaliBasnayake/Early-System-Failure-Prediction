# Phase 7 — final test evaluation (single run, tag `final-eval-v1`)

Sources: `results/test_eval/conditions_test.md`, `results/test_eval/final_test_results.json`, `results/genai/gemini_test_N5.md`, `experiments/log.csv` (phase 7 rows). Thresholds frozen from validation (`results/thresholds_final.json`); nothing was tuned or re-run on test. Test period 2005-11-06 → 2006-01-04, 101 incidents, prevalence 1.8 % (N=1) to 3.1 % (N=30).

## F7.1 Headline: the tabular model generalised to the test period; the final ANN largely did not
Test PR-AUC (prevalence in brackets); ROC-AUC in the second block.

| N | trivial | confound LR | XGBoost content-only | XGBoost history-only | **XGBoost (content+history)** | initial ANN | **final ANN (3 seeds)** |
|---|---|---|---|---|---|---|---|
| 1 | 0.018 | 0.018 | 0.033 | 0.058 | **0.074** | 0.022 | 0.022 ± 0.001 |
| 5 | 0.018 | 0.021 | 0.044 | 0.048 | **0.075** | 0.020 ± 0.003 | 0.024 ± 0.002 |
| 15 | 0.025 | 0.023 | 0.033 | 0.082 | **0.092** | 0.024 | 0.032 ± 0.006 |
| 30 | 0.031 | 0.039 | 0.050 | 0.133 | **0.184** | 0.052 | 0.041 ± 0.004 |

ROC-AUC: XGBoost 0.81 / 0.79 / 0.75 / 0.76; history-only 0.74 / 0.68 / 0.69 / 0.74; content-only 0.70 / 0.73 / 0.64 / 0.68; final ANN 0.60 / 0.60 / 0.56 / 0.63; confound 0.53 / 0.55 / 0.50 / 0.61.
At 1 % FAR XGBoost catches 13–20 % of incidents; the ANN 4–6 %.

## F7.2 Why: a regime change between the validation and test periods, not a bug
`results/test_eval/shift_analysis.md` (descriptive, post-hoc): share of positive windows whose midplane had a target failure in the previous 7 days — train 27 %, **validation 20 %, test 42 %**; lift of that single fact over the base rate — validation 1.5×, **test 3.9×**; incidents per failing unit 2.6 / 1.4 / 1.7; share of incidents on the five worst units 22 % / 25 % / 34 %; KERNSOCK incidents 9 / 2 / 13. The validation period was the *least* recurrent of the three, so the Phase 3 evidence that "history does not transfer" was true of validation and false of test. In every split 82–88 % of positives have an empty local channel: prediction rests on the machine-wide channel.
On validation, failure history was useless (history-only ROC 0.43–0.52); on test it alone reaches ROC 0.74. The test period's failures recur on the same midplanes (KERNSOCK 13–21 incidents, KERNMC 36–54; see `shift_analysis.md` for the repeat-failure statistics). The ANN was deliberately built with time+volume context only, because on validation history did not transfer (Phase 3 decision D3.x) — so it could not use the one signal that dominated the test period. Content-only XGBoost (no history) also beats the ANN on test (ROC 0.64–0.73 vs 0.56–0.63), so template drift hurt the sequence model more than the count model as well.

## F7.3 Pre-registered expectations (report/expectation.md, 24 Oct 2026) — scored honestly
1. Modest PR-AUC everywhere — **correct** (best honest result 0.075 at N=5, 4.2× prevalence).
2. XGBoost at least competitive, likely better at N=1 — **correct, and stronger than expected**: on validation the ANN led by 0.008 PR-AUC; on test XGBoost leads at every N by 3×.
3. Confound above trivial but below content — **correct** (0.018–0.039; ROC ≤ 0.61).
4. Everything degrades on test; the semantic variant degrades least — **wrong in both parts**: XGBoost *improved* (validation 0.024 → test 0.075 at N=5) because the test period's failures are more recurrent and prevalence is higher; the ANN degraded (0.032 → 0.024); the semantic model was not materially more robust than the template model (0.024 vs 0.020 at N=5, ROC 0.60 vs 0.53) despite 66 % unseen templates.
5. Earliness flat or falling; KERNMC dominates — **half right**: PR-AUC *rises* with N for every model because prevalence rises with N (0.018 → 0.031), while ROC-AUC falls for XGBoost (0.81 → 0.76); KERNMC dominates (36–54 incidents), KERNSOCK second.

## F7.4 Template drift did not hurt the count model — the transductive control on test
XGBoost with honest (train-only) templates vs templates fitted on all data, same test windows: PR-AUC 0.074 vs 0.074 (N=1), 0.075 vs 0.060 (N=5), 0.092 vs 0.097 (N=15), 0.184 vs 0.096 (N=30), at a 66 % UNK rate. The honest model is as good or better: the volume of UNK lines after the October 2005 upgrade is itself informative, and the transductive template set changes the feature space rather than simply adding information. The widely assumed "unseen templates destroy the model" effect did not materialise for the tabular model; it is visible for the sequence ANN.

## F7.5 The protocol-control gap, final
Random window split (prior-work protocol): XGBoost PR-AUC 0.58 (validation). Honest chronological protocol: 0.024 (validation), 0.075 (test). The leaky number is 8–25× the honest ones on the same model and features.

## F7.6 Generalisation results confirmed on test
- Unseen failure types (N=5): the model that never saw KERNMC matches the all-category model on 38 KERNMC incidents (incident recall 0.079 = 0.079); the model that never saw KERNSOCK does *better* on 13 KERNSOCK incidents (0.077 vs 0; PR-AUC 0.012 vs 0.006). The usable precursor signal is generic ambient state, not category signatures. KERNDTLB (2) and KERNRTSP (1) not evaluable.
- Unseen machines: on 13 never-trained racks the held-out model reaches ROC 0.71 vs 0.68 for the all-rack model (24 positives) — no penalty for unseen hardware.
- Hard negatives: final-ANN FAR on alert-rich windows 0.0–0.45 % vs 0.7–2.2 % on ordinary negatives — the ANN never became an alert-chaser.

## F7.7 Gemini on the identical 300 test windows (prompt p2, selected on validation)
ROC-AUC 0.39 (Gemini) vs 0.64 (final ANN) vs 0.78 (XGBoost); inverted Gemini score 0.61 (diagnostic). Same mechanism as validation: silence read as health, alerts read as imminent failure. 0 parse errors, 6.8 s/window, 332 k input tokens (≈ $0.30).

## What the report must say
Under a leakage-controlled chronological protocol, the best predictor of BGL kernel failures 5 minutes ahead is gradient boosting on template-count and failure-history features (PR-AUC 0.075, ROC 0.79; 19 % of incidents at 14 alarms per midplane-day). The specialised sequence ANN, which led on validation, did not transfer to the test period (PR-AUC 0.024, ROC 0.60), because the test period rewarded failure recurrence, an input the ANN had been built without on validation evidence. A frontier LLM was below chance. All honest numbers are an order of magnitude below the 0.58 obtained under the random split used by most prior work. These are the results; the design decisions that produced them are logged, and the test set was read once.
