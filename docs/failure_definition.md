# Failure definition (frozen at end of Phase 1)

**Status:** FROZEN — 2026-10-14. Any later change requires a new row in the Changes table and a note in the report.

## Rule
A *target failure event* is a BGL alert line (alert tag ≠ `-`) whose category begins with `KERN`
(kernel-level), excluding `KERNMNTF` and `KERNMNT`, and which is not part of a *mass-synchronous
incident* — defined as more than 32 distinct nodes (one BlueGene/L node card) logging the same
category within 60 s. Target events on the same node within 300 s form one *episode*. Episodes
anywhere in the system within 300 s form one *incident*, which is the unit used for support counting
and for per-incident evaluation. A category is *evaluable* when it has at least 30 incidents.

Configuration: `configs/failure_definition.yaml`. Implementation: `src/data/failure_set.py`.

## Evidence (results/eda_alert_component_level.csv, results/category_evidence.md)
- All 41 alert categories carry level FATAL or FAILURE, so the alert tag alone marks a fatal event;
  a level filter adds nothing.
- `APP*` categories are user-application errors written by every node of the affected job
  (e.g. APPSEV on exactly 1,024 nodes, one partition). They are not node failures and would not
  lead an administrator to drain a node. They form the hard-negative pool (96,367 lines, 22 categories).
- Several `KERN*` categories are mass-synchronous. KERNSTOR has 63,491 node episodes but only
  8 system-wide incidents, with a median of 1,874 nodes logging it in the same minute; KERNMNTF,
  KERNTERM and KERNREC behave the same way. These are partition- or job-level events. The
  32-node / 60 s rule removes 110 such incidents (98,364 lines).
- `KERNMNTF` / `KERNMNT` are Lustre mount failures ("Lustre mount FAILED : bglio818 : point /p/gb1").
  The fault is on the I/O server; the remedy is not to drain the compute node. Excluded.
- The 300 s burst gap lies in the valley of the alert-gap histogram (results/figures/eda_04_alert_gaps.png),
  between the ≤1 s burst spike (48% of consecutive same-node alerts) and the ~40 min mode.
- 179 episodes contain both KERNMC (machine check) and KERNRTSP (panic): a machine check followed
  within minutes by a panic. Hypothesis for later phases: machine-check lines are precursors of panics.

## Result
- Target events: 153,729 lines · episodes: 937 · incidents: 502 · nodes with ≥1 episode: 705
- Mass incidents excluded: 110 (98,364 lines) · non-target alert lines: 96,367
- Evaluable categories (incidents): KERNMC 290 · KERNRTSP 50 · KERNSOCK 38 · KERNDTLB 30
- Insufficient support (<30 incidents): KERNTERM 22, KERNCON 16, KERNPAN 14, KERNPOW 10, KERNNOETH 10,
  and ten further categories with ≤5 incidents each.
- Incidents per month: Jun 40, Jul 56, Aug 109, Sep 97, Oct 88, Nov 69, Dec 41, Jan 2.

## Declaration
This definition was derived from the whole log. It is a label definition, not a model decision,
and it was frozen before any modelling. The chronological split (Phase 2) is placed so that
60 % / 20 % / 20 % of incidents fall in train / validation / test; only the position of the two
cut dates uses incident counts, never window contents or labels.

## Changes
| Date | Version | Change | Reason |
|---|---|---|---|
| 2026-10-13 | v1 | KERN* categories, level FATAL/FAILURE, 60 s burst gap | initial rule |
| 2026-10-14 | v2 | Mass-incident exclusion (>32 nodes / 60 s); support counted in incidents; burst gap 300 s | incident analysis: KERNSTOR = 8 incidents across 22,529 nodes |
| 2026-10-14 | v3 | KERNMNTF and KERNMNT excluded | filesystem faults do not match the node-drain decision |
