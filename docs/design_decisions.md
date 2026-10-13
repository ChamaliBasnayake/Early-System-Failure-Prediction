# Design decisions log

One short entry per decision: what was chosen, the alternative(s), and why.
Q2.a asks for depth, width, normalisation, regularisation, output layer and loss to be justified individually.

## Phase 0
- **Three fixed seeds (11, 22, 33).** Alternative: one seed. Rejected because the brief requires mean and spread when results vary by seed.
- **CSV experiment log with a guard against logging the test split before Phase 7.** Alternative: free-form notes. Rejected because every reported number must be traceable to a row.
- **CPU-only PyTorch build.** Alternative: GPU. Not available; models are sized accordingly (template-ID sequences, hidden ≤ 256).
- **Existing Anaconda distribution reused; dedicated `bgl` env on E: drive.** Alternative: fresh Miniconda. Rejected as unnecessary; isolation comes from the pinned environment file, not from the distribution.
