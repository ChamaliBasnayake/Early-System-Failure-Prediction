# Gemini evaluation — validation, N=5, 300 windows (150 positives / 150 negatives, of which 50 hard)

Model `models/gemini-3.8-flash`, thinking level `low`, seed 11, run 2026-10-29, JSON output, 60 most recent lines per channel; responses cached in results/genai/cache. The sample is balanced, so PR-AUC here is NOT comparable with the full-split PR-AUC elsewhere; compare models WITHIN this table only (all scored on the identical windows). FAR thresholds are set on the sample's own negatives. The sample's negatives are one-third hard negatives (alert-rich windows), against ~10 % in the population, so a model that fires on alerts is penalised more here than it would be in deployment; the same sample design is used on test and every model is scored on the identical windows. The column `roc_auc of (1-p)` is a diagnostic only (is the LLM's ranking informative but inverted?); it is never used as a model.

| model                                      |   n_scored |   roc_auc |   roc_auc of (1-p) [diagnostic] |   pr_auc |   recall@far0.01 |   recall@far0.1 |   incident_recall@far0.1 |   balanced_acc@0.5 |   unparsable |   errors |   mean latency s |   in tokens |   out tokens |   thinking tokens |
|:-------------------------------------------|-----------:|----------:|--------------------------------:|---------:|-----------------:|----------------:|-------------------------:|-------------------:|-------------:|---------:|-----------------:|------------:|-------------:|------------------:|
| Gemini — p1_zero_shot                      |        300 |  0.339467 |                           0.661 | 0.390446 |       0          |      0.00666667 |                0.0166667 |           0.486667 |            0 |        0 |             7.29 |      219684 |        12244 |              3792 |
| Gemini — p2_domain_context                 |        300 |  0.3622   |                           0.638 | 0.413532 |       0.00666667 |      0.00666667 |                0.0166667 |           0.5      |            0 |        0 |             6.92 |      282084 |        14172 |               332 |
| Gemini — p3_few_shot                       |        300 |  0.345778 |                           0.654 | 0.4143   |       0.0133333  |      0.0333333  |                0.0666667 |           0.49     |            0 |        0 |             7.63 |      836484 |        13546 |              3936 |
| Gemini — p4_structured                     |        300 |  0.310889 |                           0.689 | 0.383613 |       0          |      0.02       |                0.0333333 |           0.5      |            0 |        0 |             7.31 |      268566 |        13703 |             17084 |
| final ANN (semantic LSTM, mean of 3 seeds) |        300 |  0.727378 |                         nan     | 0.665935 |       0.0266667  |      0.206667   |                0.233333  |           0.51     |          nan |      nan |           nan    |         nan |          nan |               nan |
| XGBoost (best validation config)           |        300 |  0.702067 |                         nan     | 0.67474  |       0.0533333  |      0.26       |                0.25      |           0.5      |          nan |      nan |           nan    |         nan |          nan |               nan |

## Pairwise Spearman correlation of scores on the sample

|                                            |   p1_zero_shot |   p2_domain_context |   p3_few_shot |   p4_structured |   final ANN (semantic LSTM, mean of 3 seeds) |   XGBoost (best validation config) |
|:-------------------------------------------|---------------:|--------------------:|--------------:|----------------:|---------------------------------------------:|-----------------------------------:|
| p1_zero_shot                               |           1    |                0.84 |          0.87 |            0.83 |                                        -0.54 |                              -0.33 |
| p2_domain_context                          |           0.84 |                1    |          0.83 |            0.83 |                                        -0.41 |                              -0.22 |
| p3_few_shot                                |           0.87 |                0.83 |          1    |            0.85 |                                        -0.47 |                              -0.27 |
| p4_structured                              |           0.83 |                0.83 |          0.85 |            1    |                                        -0.47 |                              -0.21 |
| final ANN (semantic LSTM, mean of 3 seeds) |          -0.54 |               -0.41 |         -0.47 |           -0.47 |                                         1    |                               0.47 |
| XGBoost (best validation config)           |          -0.33 |               -0.22 |         -0.27 |           -0.21 |                                         0.47 |                               1    |

## p1_zero_shot: probability distribution and example reasons

Probability histogram by label (counts):

| probability   |   0 |   1 |
|:--------------|----:|----:|
| (-0.01, 0.1]  | 136 | 149 |
| (0.1, 0.3]    |   6 |   0 |
| (0.3, 0.5]    |   3 |   0 |
| (0.7, 0.9]    |   5 |   1 |

**Positives (failure followed) — most confident miss first:**

- p=0.01 [KERNNOETH] No errors or abnormal events are logged on the midplane or system-wide, indicating normal healthy baseline operation.
- p=0.01 [KERNMC] No errors or abnormal events are logged on the midplane or system-wide, indicating normal healthy baseline operation.
- p=0.01 [KERNNOETH] No errors or abnormal events are logged on the midplane or system-wide, indicating normal healthy baseline operation.

**Negatives — most confident false alarm first:**

- p=0.85 [nan hard] The midplane has experienced a severed CioStream link coinciding with a fatal machine check interrupt (L2 dcache parity error) on the system, indicating a job-killing hardware crash that propagates or causes subsequent failures in the specified window.
- p=0.85 [nan hard] The midplane has experienced socket disconnection errors closely tied to a critical machine check interrupt and widespread severed link messages across the system, strongly indicating an imminent fatal node failure.
- p=0.85 [nan hard] The midplane has experienced DDR errors followed by severed CioStream links immediately following a severe machine check interrupt cascade across the system, indicating an active or imminent midplane crash.

## p2_domain_context: probability distribution and example reasons

Probability histogram by label (counts):

| probability   |   0 |   1 |
|:--------------|----:|----:|
| (-0.01, 0.1]  | 149 | 149 |
| (0.5, 0.7]    |   1 |   0 |
| (0.7, 0.9]    |   0 |   1 |

**Positives (failure followed) — most confident miss first:**

- p=0.01 [KERNMC] No log lines are present for either the target midplane or the whole machine over the past hour, reflecting baseline nominal conditions without precursor signals.
- p=0.01 [KERNMC] No log lines are present for either the target midplane or the whole machine over the past hour, reflecting baseline nominal conditions without precursor signals.
- p=0.01 [KERNNOETH] No log lines are present for either the target midplane or the whole machine over the past hour, reflecting baseline nominal conditions without precursor signals.

**Negatives — most confident false alarm first:**

- p=0.65 [nan] Persistent memory degradation with redundant bit steering activation, L3 EDRAM errors, and a heavy burst of 30 correctable errors on this midplane strongly foreshadows an imminent uncorrectable machine check or kernel failure.
- p=0.08 [nan hard] The midplane shows isolated socket disconnect errors during job teardown or failure on another midplane that underwent a machine check interrupt, showing no direct hardware distress or pending failure on this midplane.
- p=0.08 [nan hard] Midplane showed isolated ciod socket severed errors early on without ongoing local hardware degradation, making a target kernel failure in the 5-15 minute window unlikely.

## p3_few_shot: probability distribution and example reasons

Probability histogram by label (counts):

| probability   |   0 |   1 |
|:--------------|----:|----:|
| (-0.01, 0.1]  | 140 | 145 |
| (0.1, 0.3]    |   2 |   0 |
| (0.3, 0.5]    |   1 |   1 |
| (0.5, 0.7]    |   0 |   1 |
| (0.7, 0.9]    |   7 |   3 |

**Positives (failure followed) — most confident miss first:**

- p=0.01 [KERNNOETH] Both the midplane and the entire system are completely silent with no error activity or destabilizing conditions.
- p=0.01 [KERNMC] Both the midplane and the entire system are completely silent with no error activity or destabilizing conditions.
- p=0.01 [KERNPOW] Both the midplane and the entire system are completely silent with no error activity or destabilizing conditions.

**Negatives — most confident false alarm first:**

- p=0.85 [nan] The midplane has severe memory degradation with redundant bit steering, L3 EDRAM errors, and repeated correctable errors, coinciding with an active real-time signal panic dump across the machine.
- p=0.78 [nan hard] Midplane Lustre mount failures combined with repeated management daemon restarts and widespread Lustre mount failures across the machine closely match patterns preceding KERNSOCK failures.
- p=0.75 [nan hard] Midplane Lustre mount failures combined with machine-wide control daemon restarts (BGLMaster, mmcs_db_server, ciodb) and widespread DDR/torus errors strongly match prior patterns preceding kernel socket failures.

## p4_structured: probability distribution and example reasons

Probability histogram by label (counts):

| probability   |   0 |   1 |
|:--------------|----:|----:|
| (-0.01, 0.1]  | 149 | 150 |
| (0.1, 0.3]    |   1 |   0 |

**Positives (failure followed) — most confident miss first:**

- p=0.00 [KERNNOETH] No activity or warning indicators are observed on this midplane or machine-wide in the past hour, indicating a nominal baseline risk.
- p=0.00 [KERNNOETH] No activity or warning indicators are observed on this midplane or machine-wide in the past hour, indicating a nominal baseline risk.
- p=0.00 [KERNNOETH] No activity or warning indicators are observed on this midplane or machine-wide in the past hour, indicating a nominal baseline risk.

**Negatives — most confident false alarm first:**

- p=0.12 [nan] Repeated correctable memory errors and redundant bit steering on this midplane indicate degrading DRAM/EDRAM, elevating the risk of an uncorrectable fatal parity or machine check failure.
- p=0.08 [nan hard] The midplane shows isolated CioStream socket severed errors stemming from an application load abort or node disconnect rather than internal hardware parity or machine check triggers.
- p=0.08 [nan hard] Repeated ciod severed link errors reflect control network or I/O communication loss, often trailing or accompanying node crashes, though recent activity has quieted down.

## Cost

Tokens this run: 1,606,818 input, 78,809 output incl. thinking; pass --price-in/--price-out (USD per 1M tokens from ai.google.dev/pricing on the run day) to print the cost.
