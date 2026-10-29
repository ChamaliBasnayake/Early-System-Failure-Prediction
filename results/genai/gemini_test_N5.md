# Gemini evaluation — test, N=5, 300 windows (150 positives / 150 negatives, of which 50 hard)

Model `models/gemini-3.8-flash`, thinking level `low`, seed 11, run 2026-10-29, JSON output, 60 most recent lines per channel; responses cached in results/genai/cache. The sample is balanced, so PR-AUC here is NOT comparable with the full-split PR-AUC elsewhere; compare models WITHIN this table only (all scored on the identical windows). FAR thresholds are set on the sample's own negatives. The sample's negatives are one-third hard negatives (alert-rich windows), against ~10 % in the population, so a model that fires on alerts is penalised more here than it would be in deployment; the same sample design is used on test and every model is scored on the identical windows. The column `roc_auc of (1-p)` is a diagnostic only (is the LLM's ranking informative but inverted?); it is never used as a model.

| model                                      |   n_scored |   roc_auc |   roc_auc of (1-p) [diagnostic] |   pr_auc |   recall@far0.01 |   recall@far0.1 |   incident_recall@far0.1 |   balanced_acc@0.5 |   unparsable |   errors |   mean latency s |   in tokens |   out tokens |   thinking tokens |
|:-------------------------------------------|-----------:|----------:|--------------------------------:|---------:|-----------------:|----------------:|-------------------------:|-------------------:|-------------:|---------:|-----------------:|------------:|-------------:|------------------:|
| Gemini — p2_domain_context                 |        300 |  0.387533 |                           0.612 | 0.424246 |        0         |       0.0133333 |                0.0178571 |           0.49     |            0 |        0 |             6.77 |      331858 |        13831 |               627 |
| final ANN (semantic LSTM, mean of 3 seeds) |        300 |  0.643089 |                         nan     | 0.612571 |        0.0333333 |       0.173333  |                0.25      |           0.503333 |          nan |      nan |           nan    |         nan |          nan |               nan |
| XGBoost (best validation config)           |        300 |  0.783133 |                         nan     | 0.781501 |        0.0866667 |       0.52      |                0.535714  |           0.5      |          nan |      nan |           nan    |         nan |          nan |               nan |

## Pairwise Spearman correlation of scores on the sample

|                                            |   p2_domain_context |   final ANN (semantic LSTM, mean of 3 seeds) |   XGBoost (best validation config) |
|:-------------------------------------------|--------------------:|---------------------------------------------:|-----------------------------------:|
| p2_domain_context                          |                1    |                                        -0.47 |                              -0.27 |
| final ANN (semantic LSTM, mean of 3 seeds) |               -0.47 |                                         1    |                               0.32 |
| XGBoost (best validation config)           |               -0.27 |                                         0.32 |                               1    |

## p2_domain_context: probability distribution and example reasons

Probability histogram by label (counts):

| probability   |   0 |   1 |
|:--------------|----:|----:|
| (-0.01, 0.1]  | 146 | 148 |
| (0.3, 0.5]    |   1 |   2 |
| (0.7, 0.9]    |   3 |   0 |

**Positives (failure followed) — most confident miss first:**

- p=0.01 [KERNSERV] The target midplane has been completely quiet for the past hour, and whole-machine activity primarily indicates control system restarts (BGLMaster, mmcs_db_server) rather than impending localized hardware failure.
- p=0.01 [KERNSOCK] The midplane has been completely quiet for the past hour and machine-wide activity reflects service actions and node card issues confined to midplane R43-M1.
- p=0.01 [KERNRTSA] No log lines are present for either the target midplane or the whole machine over the past hour, reflecting baseline nominal conditions without precursor signals.

**Negatives — most confident false alarm first:**

- p=0.78 [nan hard] A burst of DDR correctable errors, torus/tree link re-synch errors, and severe ciod socket disconnects/signals immediately prior to the window strongly indicates impending kernel/socket failure.
- p=0.75 [nan hard] Repeated CioStream link severed errors on this midplane combined with widespread machine-wide bursts of torus, tree re-synch, and DDR errors indicate severe network and communication degradation preceding a kernel failure.
- p=0.72 [nan hard] Repeated DDR corrections and bit steering combined with torus/tree receiver errors and severed CioStream socket connections strongly indicate imminent node or link drop failure.

## Cost

Tokens this run: 331,858 input, 14,458 output incl. thinking; ≈ $0.30 at $0.75/M input and $3.75/M output (ai.google.dev/pricing, 2026-10-29).
