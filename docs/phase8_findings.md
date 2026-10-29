# Phase 8 — explainability of the final ANN (sem_final_N5_s11, validation windows)

Sources: `results/xai_perturbation.md` / `.csv` / `figures/xai_perturbation.png`, `results/xai_cases.md`, `/api/explain` in the web application.
Method: Integrated Gradients implemented in plain PyTorch over the projected text embeddings of each collapsed log item (local and global channel) and the 17 scalar context features; baseline = empty window (zero text vectors, mean scalars); 50 steps; completeness checked (sum of attributions vs logit difference).

## F8.1 The attributions are faithful (deletion test)
| windows | n | mean score | drop after deleting top-1 attributed item | drop after deleting 1 random item | top hurts more in |
|---|---|---|---|---|---|
| positives above the 1 % FAR threshold | 5 | 0.68 | 0.42 | −0.00 | 100 % |
| 50 highest-scoring negatives | 50 | 0.68 | 0.31 | 0.05 | 80 % |
| 50 highest-scoring positives | 50 | 0.21 | 0.09 | 0.02 | 60 % (76 % at k=3) |
Deleting 5 attributed items removes 0.50–0.51 of a 0.68 score. The items IG highlights are the items the model uses. Completeness gap mean 0.14 logits; one window reached 2.7 (50 steps insufficient there) — stated as a limitation.

## F8.2 What the model has learned (case studies, `results/xai_cases.md`)
- **Strongest single precursor: "torus receiver z+ input pipe error(s) detected and corrected" on the midplane itself** (+4.3 to +6.0 logits of attribution). It precedes the highest-scoring real failure (KERNRTSP, R31-M1) and also the most confident false alarm (R51-M0). A genuine but unreliable precursor — the model's top alarms are mostly false for this reason.
- **Machine-wide storms are discounted, not chased:** in the hard-negative case, "Lustre mount FAILED" ×8 on the midplane pushes towards failure (+0.66) while the same message ×128 across the machine pushes away (−0.49). The model distinguishes a filesystem-wide event from a local kernel precursor — the behaviour the hard-negative FAR of 0.0–0.5 % (Phases 6–7) measures.
- **Silence cannot be explained away:** the lowest-scoring real failure (KERNMC, R25-M0) has an empty local channel and one machine-wide line; the score (0.000) is below the empty-window baseline (0.018). This is the 82–88 % of positives with an empty local channel (shift_analysis) seen in one case; no input-based model can catch it, and it is where the failure-history feature that XGBoost had would have helped.
- **Scalar context is secondary:** |attribution| of time/volume features ≤ 0.5 in every case, far below the leading text items; the model is content-driven.

## F8.3 Limitations of the method (to state in Q5)
Attribution is sensitivity along one path from an empty baseline, not a causal claim; correlated items share credit; the empty-window baseline itself scores 0.02–0.11, so "baseline" is not "zero risk"; 50 IG steps occasionally leave a large completeness gap.

## In the application
`/api/explain` runs the same IG code on the pasted log; the UI colours each raw line by the attribution of the collapsed item it belongs to, lists the global-channel items and the scalar contributions, and shows the completeness gap so the user can judge the explanation's reliability.
