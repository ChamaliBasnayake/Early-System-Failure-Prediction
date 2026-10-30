# Incorrect-prediction case studies — sem_final_N5_s11, N=5 (auto-generated; interpretations to be written by hand)

1 % FAR threshold on validation: 0.4097. Attributions: Integrated Gradients (50 steps) relative to an empty window.

## Case 1 — false positive on a HARD negative (alert-rich window, no failure followed)
**correct at the 1 % FAR threshold (score 0.284 vs 0.410).** midplane **R03-M0**, window end 2005-10-06 17:56:22+00:00, label 0, anchor hard, category —, local items 1 (8 raw lines), global items 256 (1194 raw lines), UNK local 0 / global 0

- score **0.284** (baseline/empty-window score 0.037; completeness gap -0.117 logits)
- local channel (1 items): top pushing TOWARDS failure →
    - +0.657  ×8  `Lustre mount FAILED : <*> : block_id : location`
  top pushing AWAY →
- global channel (256 items): top pushing TOWARDS failure →
    - +0.276  ×1  `suppressing further interrupts of same type`
    - +0.244  ×1  `suppressing further interrupts of same type`
    - +0.217  ×1  `suppressing further interrupts of same type`
    - +0.213  ×1  `critical input interrupt (unit=<*> bit=<*>): warning for tree A0 wire`
    - +0.199  ×1  `suppressing further interrupts of same type`
    - +0.174  ×1  `suppressing further interrupts of same type`
  top pushing AWAY →
    - -0.492  ×128  `Lustre mount FAILED : <*> : block_id : location`
    - -0.183  ×1  `suppressing further interrupts of same type`
    - -0.018  ×1  `critical input interrupt (unit=<*> bit=<*>): warning for tree C1 wire`
- scalar context: dow_3 -0.496, hour_cos -0.155, dow_2 -0.091, n_units_active_global +0.072, distinct_local +0.057
- **Interpretation (write this yourself):** 

## Case 2 — false positive on an ordinary negative
**INCORRECT at the 1 % FAR threshold (score 0.908 vs 0.410).** midplane **R51-M0**, window end 2005-10-23 19:06:00+00:00, label 0, anchor uniform, category —, local items 1 (1 raw lines), global items 94 (130 raw lines), UNK local 0 / global 1

- score **0.908** (baseline/empty-window score 0.016; completeness gap +0.136 logits)
- local channel (1 items): top pushing TOWARDS failure →
    - +5.987  ×1  `<*> torus receiver z+ input pipe error(s) (dcr <*>) detected and corrected`
  top pushing AWAY →
- global channel (94 items): top pushing TOWARDS failure →
    - +0.312  ×1  `total of <*> ddr error(s) detected and corrected`
    - +0.123  ×1  `debug wait enable.................<*>`
    - +0.109  ×1  `CE sym <*>, at <*>, mask <*>`
    - +0.072  ×4  `CE sym <*>, at <*>, mask <*>`
    - +0.057  ×2  `total of <*> ddr error(s) detected and corrected`
    - +0.041  ×2  `CE sym <*>, at <*>, mask <*>`
  top pushing AWAY →
    - -0.240  ×1  `CE sym <*>, at <*>, mask <*>`
    - -0.092  ×2  `total of <*> ddr error(s) detected and corrected`
    - -0.063  ×1  `total of <*> ddr error(s) detected and corrected`
- scalar context: dow_6 +0.116, distinct_local +0.079, distinct_global +0.063, weekend +0.056, dow_3 -0.050
- **Interpretation (write this yourself):** 

## Case 3 — false negative (a real failure the model scored lowest)
**INCORRECT at the 1 % FAR threshold (score 0.000 vs 0.410).** midplane **R25-M0**, window end 2005-10-13 07:45:31+00:00, label 1, anchor pos, category KERNMC, local items 0 (0 raw lines), global items 1 (1 raw lines), UNK local 0 / global 0

- score **0.000** (baseline/empty-window score 0.018; completeness gap -0.168 logits)
- local channel: empty
- global channel (1 items): top pushing TOWARDS failure →
  top pushing AWAY →
    - -2.709  ×1  `ddr: activating redundant bit steering: rank=<*> symbol=<*>`
- scalar context: dow_3 -0.461, n_units_active_global -0.154, hour_cos -0.152, weekend -0.111, distinct_global -0.105
- **Interpretation (write this yourself):** 

## Case 4 — false negative WITH log activity on the midplane (lowest-scoring positive whose local channel is not empty)
**INCORRECT at the 1 % FAR threshold (score 0.003 vs 0.410).** midplane **R51-M1**, window end 2005-11-05 15:59:41+00:00, label 1, anchor pos, category KERNMC, local items 8 (10 raw lines), global items 139 (711 raw lines), UNK local 0 / global 0

- score **0.003** (baseline/empty-window score 0.012; completeness gap -0.153 logits)
- local channel (8 items): top pushing TOWARDS failure →
    - +1.358  ×1  `<*> torus receiver z- input pipe error(s) (dcr <*>) detected and corrected`
    - +0.496  ×1  `<*> torus receiver x- input pipe error(s) (dcr <*>) detected and corrected`
    - +0.309  ×1  `<*> torus receiver x+ input pipe error(s) (dcr <*>) detected and corrected`
    - +0.099  ×1  `<*> torus receiver y+ input pipe error(s) (dcr <*>) detected and corrected`
  top pushing AWAY →
    - -2.408  ×1  `<*> torus receiver z+ input pipe error(s) (dcr <*>) detected and corrected`
    - -1.311  ×1  `<*> torus receiver y- input pipe error(s) (dcr <*>) detected and corrected`
    - -1.009  ×3  `<*> tree receiver <*> in re-synch state event(s) (dcr <*>) detected`
- global channel (139 items): top pushing TOWARDS failure →
    - +1.610  ×1  `total of <*> ddr error(s) detected and corrected`
    - +0.659  ×1  `total of <*> ddr error(s) detected and corrected`
    - +0.277  ×225  `CE sym <*>, at <*>, mask <*>`
    - +0.185  ×4  `CE sym <*>, at <*>, mask <*>`
    - +0.085  ×1  `<*> tree receiver <*> in re-synch state event(s) (dcr <*>) detected`
    - +0.082  ×1  `<*> tree receiver <*> in re-synch state event(s) (dcr <*>) detected`
  top pushing AWAY →
    - -0.742  ×1  `<*> tree receiver <*> in re-synch state event(s) (dcr <*>) detected`
    - -0.320  ×143  `CE sym <*>, at <*>, mask <*>`
    - -0.282  ×1  `total of <*> ddr error(s) detected and corrected`
- scalar context: distinct_local +0.322, weekend +0.172, n_units_active_global +0.130, dow_5 +0.089, log_n_global +0.085
- **Interpretation (write this yourself):** 

## Contrast — true positive (highest-scoring real failure)
**correct at the 1 % FAR threshold (score 0.921 vs 0.410).** midplane **R31-M1**, window end 2005-10-23 17:13:50+00:00, label 1, anchor pos, category KERNRTSP, local items 1 (1 raw lines), global items 48 (66 raw lines), UNK local 0 / global 0

- score **0.921** (baseline/empty-window score 0.113; completeness gap +0.166 logits)
- local channel (1 items): top pushing TOWARDS failure →
    - +4.276  ×1  `<*> torus receiver z+ input pipe error(s) (dcr <*>) detected and corrected`
  top pushing AWAY →
- global channel (48 items): top pushing TOWARDS failure →
    - +0.107  ×2  `instruction cache parity error corrected`
    - +0.078  ×1  `suppressing further interrupts of same type`
    - +0.042  ×1  `total of <*> ddr error(s) detected and corrected`
    - +0.042  ×1  `total of <*> ddr error(s) detected and corrected`
    - +0.036  ×1  `total of <*> ddr error(s) detected and corrected`
    - +0.036  ×1  `suppressing further interrupts of same type`
  top pushing AWAY →
    - -0.120  ×1  `ddr: activating redundant bit steering: rank=<*> symbol=<*>`
    - -0.053  ×1  `ddr: activating redundant bit steering: rank=<*> symbol=<*>`
    - -0.053  ×2  `ddr: activating redundant bit steering: rank=<*> symbol=<*>`
- scalar context: hour_sin -0.097, dow_3 -0.063, distinct_local +0.060, dow_2 +0.058, dow_1 +0.051
- **Interpretation (write this yourself):** 

## Limitations of the method (state these in the report)
- Attribution measures sensitivity along one path from an EMPTY baseline; it is not a causal claim.
- Correlated items share attribution; a line can be important yet receive little attribution if a similar line already explains the change.
- A large share of attribution on the scalar context means the model leans on time/volume, which the perturbation test will also show.