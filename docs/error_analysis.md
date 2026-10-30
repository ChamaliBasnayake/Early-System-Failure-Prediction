# Error analysis of the final ANN (Question 3.a)

Sources: `results/error_buckets.md` (validation, mean of three seeds, frozen 1 % FAR threshold 0.410), `results/test_eval/conditions_test.md` (per-category and hard-negative tables on test), `results/test_eval/shift_analysis.md`, `results/xai_cases.md` (five attributed cases, four of them incorrect predictions).

## E1 The signal lives in the 12 % of positives whose midplane logged something
| local channel | windows | positives | PR-AUC | lift over prevalence | ROC | recall @ thr | incident recall |
|---|---|---|---|---|---|---|---|
| empty | 12,461 | 170 (88 %) | 0.023 | 1.7× | 0.672 | 0.00 | 0.00 |
| has lines | 3,331 | 23 (12 %) | **0.115** | **16.7×** | 0.696 | **0.22** | 0.11 |
When the midplane itself has logged in the last hour, the model is a useful ranker (one positive in nine at the top of the list, 22 % recall at 1 % FAR). When it has not — 88 % of real failures — the model is close to chance and never fires: every miss at the operating point is a silent-midplane failure. This is the irreducible-error regime for any log-only model and the single most important fact for the user: the system can only warn about failures that announce themselves.

## E2 Unseen templates raise the false-alarm rate
Validation UNK buckets: FAR at the frozen threshold rises from 0.6 % (no UNK lines) to 1.6 % (0–10 % UNK) and 13.6 % (10–50 % UNK, 266 windows); no positive in the UNK-rich buckets was caught. The semantic model embeds unseen texts but was never *trained* on them, so unfamiliar vectors push scores up. On the test period (66 % UNK) the ordinary-window FAR at the same threshold was 0.7–2.2 %: lower than this bucket suggests, because post-upgrade messages dominate every window there rather than marking unusual ones. Data-quality lesson: an unknown-template rate is a warning sign for the user, which is why the application displays it.

## E3 Quiet windows: best ranking, zero alarms
By global volume quartile, the quietest quartile has the highest ROC (0.706) and the most positives (75 of 193) but recall and FAR of exactly 0: the model never exceeds the threshold on a quiet machine. All alarms occur in the busier three quartiles, where the lift is modest (1.8–4.5×). The threshold, chosen for a 1 % FAR overall, is effectively infinite for quiet periods.

## E4 Schedule is a mild factor, not the driver
The 12–17 UTC bucket has the highest prevalence (2.0 %), the only non-zero recall (6 %) and the highest FAR (1.8 %), with the lowest ROC (0.633); the other three day-parts have ROC 0.73–0.74 and FAR ≤ 0.9 %. The model fires more in the busy afternoon, partly because that is when failures happen (confound baseline ROC ≤ 0.56) and partly because of content; the attribution of the scalar context in the case studies (|attr| ≤ 0.5) shows content dominates.

## E5 Not an alert-chaser
FAR at the frozen threshold: hard negatives (alert-rich, no failure) **0.0 %** (1,405 windows, mean score 0.005); uniform negatives 0.95 %; negatives anchored near a real failure but outside its horizon 2.4 % (mean score 0.11). The model scores windows *near* failures higher than random windows and alert storms lower than random windows — the opposite of the LLM's behaviour (Phase 10). Confirmed on test: 0.0–0.45 % on hard negatives vs 0.7–2.2 % on ordinary ones.

## E6 By failure category
Validation: only KERNRTSP (1 incident; local torus-receiver precursor) is caught; KERNMC (43 incidents, 91 % with an empty local channel) has mean score 0.11 and 0 % incident recall. Test: KERNMC 2–8 %, KERNSOCK 0–5 %, single-incident categories 0 or 1. The category table is the same fact as E1 seen by type: categories whose failing midplane is silent beforehand are not predicted.

## E7 What the attributed cases add (`results/xai_cases.md`)
- The strongest learned precursor is a *single* corrected torus-receiver error on the midplane (+4 to +6 logits) — real but unreliable (Case 2 false alarm vs the true positive).
- The same message carries −2.4 when all six torus directions report corrected errors at once (Case 4): the model has learned that a whole-neighbourhood link event is benign, and misses the minority of cases where it precedes a machine check. The network reads configurations of messages, not message identities.
- A machine-wide Lustre storm is discounted (Case 1): filesystem event, not kernel precursor.
- Silence cannot be explained away (Case 3): score below the empty-window baseline; only a failure-history input could have helped, which the ANN did not have and XGBoost did (Phase 7).

## Consequences
1. Deploy with the unknown-template rate and the "no local lines" condition shown to the user (done in the app); a silent midplane is "no information", not "safe".
2. The threshold policy should be volume-aware or the operating point should be set per load regime; a single global 1 % FAR threshold yields zero alarms in quiet periods.
3. Future work with the strongest evidence behind it: add failure-recurrence inputs (Phase 7) and treat the "all-directions link event" pattern explicitly.
