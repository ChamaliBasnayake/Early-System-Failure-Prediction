# Preprocessing record (every transformation that affects model input)

| Step | Script | Transformation | Parameters | Phase |
|---|---|---|---|---|
| Parse | `src/data/parse_bgl.py` | timestamps from the epoch field (UTC; the human-readable field is US/Pacific); each line split into 9 metadata fields + message (`split(maxsplit=9)`); lines with <9 fields dropped (0 in BGL); `is_alert = alert_tag != '-'`; rack / midplane / node-card derived from the node id; columns stored as plain strings (zstd Parquet) | — | 1 |
| Failure set | `src/data/failure_set.py` | target event = KERN* alert line excluding KERNMNTF/KERNMNT and excluding mass-synchronous incidents (>32 nodes within 60 s, same category); node episodes = target events within 300 s; incidents = episodes system-wide within 300 s; APP*/LINK*/MAS*/MON*/MMCS alerts kept as the non-target pool | `configs/failure_definition.yaml` | 1 |
| Masking | `src/data/templates.py` | numbers, hex values, node ids, paths and addresses replaced by placeholders before template mining | regex list in the script | 2 |
| Templates | `src/data/templates.py` | Drain3 fitted on the training period only; validation/test matched in inference mode; unmatched → `UNK`; UNK rate reported per split | `configs/drain3.ini` | 2 |
| Windows | `src/data/windows.py` | per node: W-minute window of (template, repeat-count, Δt) ending at t, capped at the last 256 events; label per lead time N: failure episode starts in (t+N, t+N+H]; windows overlapping a target episode, a mass incident or the gap zone are excluded | `configs/windows.yaml` | 2 |
| Negatives | `src/data/windows.py` | negatives sampled at event times (activity-weighted); hard negatives anchored on non-target alerts; train negatives sub-sampled to a fixed ratio (seeded); exact-duplicate windows removed across and within splits | `configs/windows.yaml` | 2 |

Personal data: none. BGL contains machine identifiers and system messages only; no anonymisation is required.
