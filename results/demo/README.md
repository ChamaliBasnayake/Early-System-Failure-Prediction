# Demo inputs — sem_final_N5_s11, N=5 (from VALIDATION)

| file | midplane | label | score | anchor | category | lines in window | expected app behaviour |
|---|---|---|---|---|---|---|---|
| tp_R31-M1.log | R31-M1 | 1 | 0.921 | pos | KERNRTSP | 66 | alarm (true positive) |
| fn_R25-M0.log | R25-M0 | 1 | 0.000 | pos | KERNMC | 1 | no/low alarm although a failure followed (false negative) |
| fp_hard_R03-M0.log | R03-M0 | 0 | 0.284 | hard | — | 1194 | alarm although no failure followed (false positive) |
| neg_R31-M0.log | R31-M0 | 0 | 0.122 | uniform | — | 127 | no alarm |
| thunderbird_ood.log | — | — | — | — | — | 2000 | out-of-distribution: different system, high unknown-template rate |

For the demo: paste a file's content, select the midplane shown here, lead time N, and compare the app's decision with the label.