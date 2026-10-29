# Incorrect-prediction case studies — sem_final_N5_s11, N=5 (auto-generated; interpretations written from the attributions, Integrated Gradients 50 steps)

1 % FAR threshold on validation: 0.4097. Attributions: Integrated Gradients (50 steps) relative to an empty window.

## Case 1 — false positive on a HARD negative (alert-rich window, no failure followed)
midplane **R03-M0**, window end 2005-10-06 17:56:22+00:00, label 0, anchor hard, category —, local items 1 (8 raw lines), global items 256 (1194 raw lines), UNK local 0 / global 0

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
- **Interpretation:** At the 1 % FAR threshold (0.41) this window is in fact rejected (score 0.28), so the model handled the hard negative correctly; the case is still instructive because of *how* it decided. The only local item, `Lustre mount FAILED` repeated 8 times on this midplane, pushes the score up (+0.66), but the same message repeated 128 times across the whole machine pushes it down (−0.49) and almost cancels it. The model has learned that a machine-wide Lustre mount storm is a filesystem event affecting everyone, not a sign that *this* midplane's kernel is about to fail. This is exactly the distinction the hard-negative set was built to test, and it is why the false-alarm rate on alert-rich windows is 0.0–0.5 % (Phases 6–7). The remaining positive contributions come from a handful of `suppressing further interrupts` and tree-wire interrupt lines (each ≈ +0.2), i.e. genuine hardware warnings elsewhere in the machine. The scalar context (day-of-week −0.50) also lowers the score: the model carries some schedule information, a limitation to keep in mind.

## Case 2 — false positive on an ordinary negative
midplane **R51-M0**, window end 2005-10-23 19:06:00+00:00, label 0, anchor uniform, category —, local items 1 (1 raw lines), global items 94 (130 raw lines), UNK local 0 / global 1

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
- **Interpretation:** This is the model's most confident mistake and the most informative case. The entire decision rests on one local line, `torus receiver z+ input pipe error(s) detected and corrected` (+5.99 logits), with the global DDR/CE correctable-error lines adding only ±0.3. The contrast case below shows the same message on R31-M1 (+4.28) preceding a real KERNRTSP failure. So the network has learned a rule close to "a torus receiver error on this midplane means a kernel failure within minutes". That rule is correct often enough to be the strongest precursor the model knows and wrong here: the error was corrected and nothing followed. It explains why the model's highest scores are dominated by false alarms (the 50 top-scoring negatives average 0.68 while the 50 top-scoring positives average 0.21) and therefore why PR-AUC stays low even though ROC-AUC is reasonable. For an operator this is also the most useful kind of explanation: the alarm can be traced to one line and judged on its merits.

## Case 3 — false negative (a real failure the model scored lowest)
midplane **R25-M0**, window end 2005-10-13 07:45:31+00:00, label 1, anchor pos, category KERNMC, local items 0 (0 raw lines), global items 1 (1 raw lines), UNK local 0 / global 0

- score **0.000** (baseline/empty-window score 0.018; completeness gap -0.168 logits)
- local channel: empty
- global channel (1 items): top pushing TOWARDS failure →
  top pushing AWAY →
    - -2.709  ×1  `ddr: activating redundant bit steering: rank=<*> symbol=<*>`
- scalar context: dow_3 -0.461, n_units_active_global -0.154, hour_cos -0.152, weekend -0.111, distinct_global -0.105
- **Interpretation:** The midplane logged nothing in the preceding hour and the whole machine produced a single line, `ddr: activating redundant bit steering`, which the model reads as a routine memory-repair action and uses to push the score *below* the empty-window baseline (0.000 vs 0.018). Nothing in the input distinguishes this hour from thousands of quiet hours, and the failure that followed (a KERNMC machine check) left no precursor in the logs. This is not a modelling error that a different architecture would fix: 82–88 % of positive windows have an empty local channel in every split (shift analysis), and the precursor diagnostics showed the failing node is silent before 90 %+ of failures. It is the clearest illustration of the ceiling on log-only prediction in this system, and of what the ANN lacked relative to XGBoost on the test period: a failure-history input (whether R25-M0 had failed recently) is the only information that could have raised this score.

## Contrast — true positive (highest-scoring real failure)
midplane **R31-M1**, window end 2005-10-23 17:13:50+00:00, label 1, anchor pos, category KERNRTSP, local items 1 (1 raw lines), global items 48 (66 raw lines), UNK local 0 / global 0

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
- **Interpretation:** The highest-scoring real failure is driven by the same local precursor as the false alarm in Case 2, `torus receiver z+ input pipe error(s)` (+4.28), supported by corrected instruction-cache parity errors and DDR corrections elsewhere in the machine (each ≈ +0.04 to +0.11) that together indicate a system under mild hardware stress. The `redundant bit steering` lines push slightly away, consistent with Case 3: the model treats them as repair actions rather than warnings. Placed next to Case 2, this pair shows precisely what the network has and has not learned: it has learned a real precursor pattern (local torus receiver error in a mildly stressed machine), and it has not learned anything that separates the occasions when that pattern is followed by a failure from those when it is not, because that information is probably not in the logs. The scalar context contributes little (|attribution| ≤ 0.1), so the decision is content-driven.

## Limitations of the method (state these in the report)
- Attribution measures sensitivity along one path from an EMPTY baseline; it is not a causal claim.
- Correlated items share attribution; a line can be important yet receive little attribution if a similar line already explains the change.
- A large share of attribution on the scalar context means the model leans on time/volume, which the perturbation test will also show.