# EDA: three findings that constrain modelling (Question 1)

Numbers come from `results/eda_summary.md`, `results/category_evidence.md` and
`results/failure_set_summary.md` (full BGL log, 4,747,963 lines, 2005-06-03 → 2006-01-04).

## 1. Replication, not abundance
348,460 alert lines (7.3 % of all lines) look like a rich failure signal, but alerts arrive in bursts
and in lock-step across nodes: 48 % of consecutive same-node alerts are ≤1 s apart, and whole
partitions log the same event in the same minute (KERNSTOR: 63,491 node episodes but 8 system-wide
incidents; APPSEV on exactly 1,024 nodes). After collapsing bursts and removing mass-synchronous
events, independent node failures number **502 incidents** over seven months.

*Consequences:* events are collapsed into episodes (300 s) and incidents (300 s, system-wide);
near-identical windows from the same instant are de-duplicated; every result is reported per
incident as well as per window. Otherwise one failure is counted thousands of times, and copies
of it land on both sides of the train/test boundary.

## 2. Not every alert is a node failure
`APP*` alerts are user-program errors written by every node of a job; `KERNMNTF`/`KERNMNT` are
filesystem mount failures whose cause is the I/O server; mass-synchronous `KERN*` events are
partition-level. Only node-local kernel/hardware faults — KERNMC (machine check), KERNRTSP (panic),
KERNSOCK (control-network socket lost), KERNDTLB (TLB error storm) and smaller categories — match the
administrator's decision to drain a node.

*Consequences:* a behaviour-based target definition (`docs/failure_definition.md`); a hard-negative
pool of alert-rich windows that are *not* followed by a node failure (96,367 lines); per-category
evaluation; an "insufficient support" rule for categories with fewer than 30 incidents.

## 3. Strong time dependence
Daily log volume varies ~100-fold (June–July heavy, October–January quiet); alert lines peak at
09:00 and 17:00–23:00 local time; 30 % of lines repeat the previous line on the same node; incidents
per month fall from 109 (August) to 2 (January).

*Consequences:* a strictly chronological split, with the two cut dates placed so that each period
holds a fixed share of incidents (60/20/20); a content-free baseline using only time-of-day,
day-of-week and log volume, to prove that a content model learns failure symptoms rather than
schedule; collapsing of consecutive duplicate lines into (template, repeat-count) before modelling;
and the expectation that performance may differ between the busy summer and the quiet winter,
which the rolling-origin evaluation will test.

## Data-quality notes for Q1
- 0 malformed lines dropped by the parser; 316 lines carry shifted fields (level values such as
  "Kill", "microseconds"), all non-alert and harmless.
- Timestamps are taken from the epoch field (UTC); the human-readable field in each line is US/Pacific.
- Top 1 % of nodes produce 17 % of all lines; 69,252 distinct node identifiers across 67 racks.
