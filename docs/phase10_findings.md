# Phase 10 — Gemini as a zero-/few-shot predictor (validation; prompt selection)

Source: `results/genai/gemini_validation_N5.md`, `results/genai/gemini_validation_N5_summary.csv`, per-window files `results/genai/gemini_validation_N5_p*.csv`.
Model `gemini-3.8-flash` (Interactions API, thinking level `low`, seed 11), run 2026-10-28, Tier 1 key. 300 validation windows at N = 5: 150 positives (stratified by category), 100 uniform negatives, 50 hard negatives. The final ANN (mean of seeds 11/22/33) and XGBoost were scored on the identical windows.

## F10.1 Gemini ranks below chance on every prompt; the ANN and XGBoost rank well on the same windows
| model | ROC-AUC | recall @ 10 % FAR | incident recall @ 10 % FAR | share of windows with p ≤ 0.1 |
|---|---|---|---|---|
| Gemini p1 zero-shot | 0.339 | 0.007 | 0.017 | 95 % |
| Gemini p2 domain context | **0.362** | 0.007 | 0.017 | 99 % |
| Gemini p3 few-shot (4 train examples) | 0.346 | 0.033 | 0.067 | 95 % |
| Gemini p4 structured (+ summary statistics) | 0.311 | 0.020 | 0.033 | 99.7 % |
| final ANN (semantic LSTM) | **0.727** | 0.207 | 0.233 | — |
| XGBoost | 0.702 | 0.260 | 0.250 | — |

All 1,200 calls parsed (0 unparsable, 0 errors); mean latency 6.9–7.6 s per window against 0.36 ms for the ANN on CPU.

## F10.2 Mechanism: the LLM's prior is inverted for this protocol
Gemini assigns p ≈ 0.01 to silent windows ("no errors or abnormal events … healthy baseline") and p ≈ 0.85 to alert-rich windows ("severed CioStream link … imminent fatal node failure"). In BGL the failing unit is silent before 90–97 % of failures and alert-rich windows are, by construction, the hard negatives. Hence the negative Spearman correlation with the ANN (−0.41 to −0.54) and the sub-0.5 ROC. The ANN learned the opposite (hard-negative FAR 0.0 %, Phase 6). Explicit domain context (p2 states "quiet windows are NOT automatically safe") and few-shot examples did not override the prior.

## F10.3 Prompt design barely matters; few-shot triples the cost
ROC spread across prompts is 0.05. p3 uses 836 k input tokens against 220–282 k for the others. p4's extra summary statistics made the model *more* conservative (99.7 % of windows ≤ 0.1), not better.

## F10.4 Pre-registered selection
Rule fixed before the run (in the script's report): highest ROC-AUC, then recall at 10 % FAR, then unparsable rate → **p2_domain_context** frozen with `--select p2` for the single test run. Model and thinking level are stored with the selection and enforced on test.

## Caveats
- Balanced sample with 33 % hard negatives among negatives (population ≈ 10 %): harsher for an alert-chasing model than deployment would be; identical design on test; all models on identical windows.
- Gemini outputs are near-constant, so PR-AUC/recall values are dominated by ties; ROC-AUC is the informative metric here.
- The diagnostic `roc_auc of (1-p)` in the report shows whether Gemini's ranking carries information with the wrong sign; it is never used as a predictor.
- Model history: planned `gemini-2.5-flash` retired for new keys (404, 28 Oct 2026); `gemini-3.8-flash` free tier limited to 20 requests/day; run completed on a Tier 1 (paid) key; probe output recorded in `docs/ai_tool_declaration.md`.
