# Error analysis by input type and data quality — sem_final (mean of seeds [11, 22, 33]), N=5, validation

Frozen 1 % FAR threshold 0.4100. 'PR-AUC / prevalence' is the lift over a constant predictor within the bucket (1.0 = chance).

## Local channel empty vs not

| bucket                  |   windows |   positives |   prevalence |   PR-AUC |   PR-AUC / prevalence |   ROC-AUC |   recall @ frozen 1 % thr |   FAR @ frozen 1 % thr |   incident recall |
|:------------------------|----------:|------------:|-------------:|---------:|----------------------:|----------:|--------------------------:|-----------------------:|------------------:|
| local channel EMPTY     |     12461 |         170 |       0.0136 |   0.0228 |                   1.7 |     0.672 |                     0     |                 0.0084 |             0     |
| local channel has lines |      3331 |          23 |       0.0069 |   0.1154 |                  16.7 |     0.696 |                     0.217 |                 0.0118 |             0.111 |

## Unknown-template rate of the global channel

| bucket      |   windows |   positives |   prevalence |   PR-AUC |   PR-AUC / prevalence |   ROC-AUC |   recall @ frozen 1 % thr |   FAR @ frozen 1 % thr |   incident recall |
|:------------|----------:|------------:|-------------:|---------:|----------------------:|----------:|--------------------------:|-----------------------:|------------------:|
| UNK 0 %     |     14455 |         186 |       0.0129 |   0.037  |                   2.9 |     0.693 |                     0.027 |                 0.0059 |             0.017 |
| UNK 0–10 %  |       935 |           6 |       0.0064 |   0.0092 |                   1.4 |     0.516 |                     0     |                 0.0161 |             0     |
| UNK 10–50 % |       266 |           1 |       0.0038 |   0.0042 |                   1.1 |     0.1   |                     0     |                 0.1358 |             0     |
| UNK > 50 %  |       136 |           0 |       0      | nan      |                 nan   |   nan     |                   nan     |                 0.0515 |           nan     |

## Global log volume in the window

| bucket                      |   windows |   positives |   prevalence |   PR-AUC |   PR-AUC / prevalence |   ROC-AUC |   recall @ frozen 1 % thr |   FAR @ frozen 1 % thr |   incident recall |
|:----------------------------|----------:|------------:|-------------:|---------:|----------------------:|----------:|--------------------------:|-----------------------:|------------------:|
| global volume Q1 (quietest) |      3948 |          75 |       0.019  |   0.0474 |                   2.5 |     0.706 |                     0     |                 0      |             0     |
| Q2                          |      3948 |          33 |       0.0084 |   0.023  |                   2.8 |     0.667 |                     0.03  |                 0.012  |             0.05  |
| Q3                          |      3948 |          54 |       0.0137 |   0.0622 |                   4.5 |     0.635 |                     0.074 |                 0.0123 |             0.043 |
| Q4 (busiest)                |      3948 |          31 |       0.0079 |   0.0144 |                   1.8 |     0.689 |                     0     |                 0.012  |             0     |

## Hour of day

| bucket    |   windows |   positives |   prevalence |   PR-AUC |   PR-AUC / prevalence |   ROC-AUC |   recall @ frozen 1 % thr |   FAR @ frozen 1 % thr |   incident recall |
|:----------|----------:|------------:|-------------:|---------:|----------------------:|----------:|--------------------------:|-----------------------:|------------------:|
| 00–05 UTC |      3498 |          12 |       0.0034 |   0.0192 |                   5.6 |     0.737 |                     0     |                 0.0049 |             0     |
| 06–11     |      3907 |          57 |       0.0146 |   0.0348 |                   2.4 |     0.737 |                     0     |                 0.0091 |             0     |
| 12–17     |      4052 |          81 |       0.02   |   0.058  |                   2.9 |     0.633 |                     0.062 |                 0.0176 |             0.037 |
| 18–23     |      4335 |          43 |       0.0099 |   0.0235 |                   2.4 |     0.734 |                     0     |                 0.0047 |             0     |

## Negatives: false-alarm rate at the frozen threshold by anchor type

| anchor   |   windows |    FAR |   mean score |
|:---------|----------:|-------:|-------------:|
| hard     |      1405 | 0      |       0.0045 |
| pos      |       455 | 0.0242 |       0.1135 |
| uniform  |     13739 | 0.0095 |       0.0853 |

## Positives by failure category

| category   |   incidents |   windows |   mean score |   incident recall @ thr |   share with empty local channel |
|:-----------|------------:|----------:|-------------:|------------------------:|---------------------------------:|
|            |           1 |         1 |       0.0931 |                       0 |                            1     |
| KERNBIT    |           1 |         1 |       0.0181 |                       0 |                            0     |
| KERNCON    |           2 |         4 |       0.1286 |                       0 |                            1     |
| KERNFLOAT  |           1 |         3 |       0.1315 |                       0 |                            1     |
| KERNMC     |          43 |       129 |       0.1134 |                       0 |                            0.907 |
| KERNMICRO  |           2 |        12 |       0.1429 |                       0 |                            1     |
| KERNNOETH  |           3 |        11 |       0.1088 |                       0 |                            1     |
| KERNPAN    |           5 |        13 |       0.0904 |                       0 |                            0.769 |
| KERNPOW    |           1 |         5 |       0.1555 |                       0 |                            1     |
| KERNPROG   |           1 |         2 |       0.1053 |                       0 |                            1     |
| KERNRTSP   |           1 |         9 |       0.4702 |                       1 |                            0.222 |
| KERNSOCK   |           2 |         2 |       0.1014 |                       0 |                            1     |
| KERNTERM   |           1 |         1 |       0.0679 |                       0 |                            1     |

## Reading
- If the EMPTY-local bucket has PR-AUC/prevalence near 1 and the non-empty bucket well above 1, the model's whole signal lives in windows where the midplane itself logged something; the empty bucket is the irreducible-error regime (no input).
- A FAR that is lower on hard negatives than on uniform negatives means the model is not an alert-chaser.
- UNK buckets on validation are small (2.8 % UNK overall); the test-period effect of UNK (66 %) is in docs/phase7_findings.md F7.4.
