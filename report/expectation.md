# Expectation statement (written BEFORE any model was trained — Question 2.b)

Date: 2026-10-24. Author: Chamali Basnayake.

**Primary metric:** average precision (PR-AUC) on the frozen validation split, per lead time N; with
recall at fixed false-alarm rates (0.1 % and 1 % of negative windows ≈ 1.4 and 14 alarms per midplane per
day) and per-incident recall as operating-point metrics. Thresholds are chosen on validation and frozen.

**What I expect, and why**
1. All models will achieve modest PR-AUC. The precursor diagnostics show that most failures have no local
   log precursor (the failing midplane is silent in 66 % of cases) and only mild system-wide elevation; the
   positive rate is ~1.5 %.
2. XGBoost on template-count features will be at least competitive with the sequence ANN on validation, and
   likely better at N = 1 min. Reasons: Xiao et al. (2026) found gradient boosting on counts/semantic features
   the strongest configuration under chronological BGL evaluation, with a small LSTM weakest; the signal here
   is "which messages appear", which counts capture, and ordering within a 60-minute window is probably
   secondary. The ANN may gain as N grows if subtle precursor sequences matter, but I do not expect a large gap.
3. The content-free confound baseline will be well above trivial (activity and time-of-day correlate with
   failures) but below the content models. If it is not, the content models have learned schedule.
4. Everything will degrade on the test period because of the October 2005 software upgrade (61 % of test
   lines are unseen templates). The variant using pretrained text embeddings of message text (Phase 5)
   should degrade least, because it can represent messages it has never seen.
5. The earliness curve will be flat or falling with N; per-category results will be dominated by KERNMC.

I will report whichever of these turn out wrong, with the evidence.
