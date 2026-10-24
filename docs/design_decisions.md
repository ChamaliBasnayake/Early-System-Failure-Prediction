# Design decisions log

One short entry per decision: what was chosen, the alternative(s), and why.
Q2.a asks for depth, width, normalisation, regularisation, output layer and loss to be justified individually;
Q1 and Q3 reward documented data decisions.

## Phase 0
- **Three fixed seeds (11, 22, 33).** Alternative: one seed. Rejected because the brief requires mean and spread when results vary by seed.
- **CSV experiment log with a guard against logging the test split before Phase 7.** Alternative: free-form notes. Rejected because every reported number must be traceable to a row.
- **CPU-only PyTorch build.** Alternative: GPU. Not available; models are sized accordingly (template-ID sequences, hidden ≤ 256).
- **Existing Anaconda distribution reused; dedicated `bgl` env, caches and Docker data on the E: drive.** Alternative: fresh Miniconda. Unnecessary; isolation comes from the pinned environment file.

## Phase 1
- **Parquet columns written as plain strings, not categoricals.** Alternative: pandas category → Arrow dictionary. Rejected after an int16 dictionary-index overflow when a later chunk exceeded 32,767 distinct nodes; zstd compression keeps the file small anyway.
- **Target defined by behaviour (mass-incident rule), not by category list alone.** Alternative: all KERN* categories. Rejected because KERNSTOR's 63,491 episodes are 8 partition-wide events that would dominate the positive class and are not predictable per node.
- **Mount-failure categories (KERNMNTF, KERNMNT) excluded from the target.** Alternative: keep as a fifth evaluable category. Rejected for misalignment with the node-drain decision; they strengthen the hard-negative pool instead.
- **Support counted in incidents (system-wide, 300 s), not episodes.** Alternative: episode counts. Rejected because synchronous events inflate episode counts by three orders of magnitude.
- **Burst gap 300 s.** Alternative: 60 s. 300 s sits in the valley of the alert-gap histogram and is half the prediction horizon; 60 s risks splitting one storm that pauses briefly into two episodes.
- **Split cut dates placed by incident share (60/20/20).** Alternative: 70/15/15 by calendar days. Rejected because the quiet December–January tail would leave ~45 test incidents, too few for per-category reporting.

## Phase 2
- **Chronological split with cuts at the 60 % / 80 % incident quantiles (train 302 / validation 99 / test 101 incidents).** Alternative: 70/15/15 by days. Rejected: ~45 test incidents.
- **Drain3 fitted on the training period only; UNK for unmatched; transductive fit kept only as a labelled control.** Alternative: fit on all data (most prior work). Rejected as leakage (Le & Zhang 2022; Xiao et al. 2026). Consequence discovered: a BGL control-software upgrade in October 2005 (idoproxydb V1R1M2_254/255) makes 61 % of test-period lines unseen message types; the transductive control shows 0 % UNK and 530 vs 317 templates.
- **Every distinct masked message text stored with an id alongside the frozen template index.** Enables an inductive pretrained text encoder to embed unseen messages (Phase 5) without fitting anything on test data.
- **Unit of prediction changed from node to MIDPLANE; two input channels (unit-local + system-global, 60 min each).** Evidence: precursor diagnostics — the failing node is silent before 90–97 % of its failures, its midplane before 66 %, while the system is active before ~90 % with elevated volume (median 83 vs 12 lines/h). Alternatives: node unit with 10-min window (positives are empty: degenerate), system-only unit (loses the per-unit decision and the held-out-machine experiment). Midplane is also the unit BGL allocates jobs by, so the administrator decision becomes "withhold this midplane from the next allocation".
- **Negatives sampled uniformly over (unit, time), not at event times.** Alternative: activity-weighted sampling at event times. Rejected after it produced non-empty negatives vs. empty positives, a sampling confound that would be learned instead of failure signal.
- **Positive windows at five offsets per unit-episode (k = 0,2,4,6,8 min before the gap).** Gives ~5 training examples per rare episode; all metrics are also reported per incident so this cannot inflate results.
- **Exclusion zones**: failure visible in local window, 20 min after an episode, failure inside the gap, mass-incident lines in window+horizon, span crossing a split cut. Each is counted and reported.
