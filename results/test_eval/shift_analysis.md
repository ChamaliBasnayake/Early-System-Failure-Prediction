# Validation → test shift (descriptive, post-hoc; N=5)

Computed after `final-eval-v1`; no model was trained or changed. Explains why failure-history features were useless on validation but strong on test, and why the ANN (time+volume context only, no history inputs) did not transfer.

|                                          | train                                          | validation                                   | test                                           |
|:-----------------------------------------|:-----------------------------------------------|:---------------------------------------------|:-----------------------------------------------|
| windows                                  | 42377                                          | 15792                                        | 17314                                          |
| prevalence                               | 0.0157                                         | 0.0122                                       | 0.0165                                         |
| positive incidents                       | 189                                            | 64                                           | 62                                             |
| failing units                            | 87                                             | 49                                           | 51                                             |
| incidents per failing unit               | 2.63                                           | 1.37                                         | 1.73                                           |
| share of incidents on top-5 units        | 0.22                                           | 0.25                                         | 0.34                                           |
| positives with unit failure in prev 24 h | 0.137                                          | 0.067                                        | 0.136                                          |
| negatives with unit failure in prev 24 h | 0.023                                          | 0.019                                        | 0.024                                          |
| positives with unit failure in prev 7 d  | 0.267                                          | 0.197                                        | 0.42                                           |
| negatives with unit failure in prev 7 d  | 0.114                                          | 0.127                                        | 0.103                                          |
| UNK rate (global lines)                  | 0.0                                            | 0.028                                        | 0.659                                          |
| median global lines/hour                 | 22.0                                           | 37.0                                         | 19.0                                           |
| positives with empty local channel       | 0.818                                          | 0.881                                        | 0.832                                          |
| top categories                           | KERNMC 118, KERNRTSP 33, KERNSOCK 9, KERNPOW 6 | KERNMC 43, KERNPAN 5, KERNNOETH 3, KERNCON 2 | KERNMC 38, KERNSOCK 13, KERNSERV 3, KERNDTLB 2 |

- validation: P(failure | unit failed in previous 7 d) = 0.0188 vs base rate 0.0122 → lift 1.5×; such windows are 12.8% of all windows.
- test: P(failure | unit failed in previous 7 d) = 0.0641 vs base rate 0.0165 → lift 3.9×; such windows are 10.8% of all windows.
