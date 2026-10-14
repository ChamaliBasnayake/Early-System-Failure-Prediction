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
