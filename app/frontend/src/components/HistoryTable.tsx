import { Chip, Table, TableBody, TableCell, TableHead, TableRow, Typography } from '@mui/material'
import type { HistoryRow } from '../lib/api'
export default function HistoryTable({ rows }: { rows: HistoryRow[] }) {
  if (!rows.length) return <Typography color="text.secondary">No predictions stored yet.</Typography>
  return (<Table size="small"><TableHead><TableRow><TableCell>#</TableCell><TableCell>time</TableCell><TableCell>midplane</TableCell><TableCell>N</TableCell><TableCell>score</TableCell><TableCell>spread</TableCell><TableCell>alarm</TableCell><TableCell>UNK %</TableCell><TableCell>latency</TableCell><TableCell>feedback</TableCell></TableRow></TableHead>
    <TableBody>{rows.map(r => <TableRow key={r.id}><TableCell>{r.id}</TableCell><TableCell>{new Date(r.created_at).toLocaleString()}</TableCell><TableCell>{r.unit}</TableCell><TableCell>{r.lead_time}</TableCell><TableCell>{r.score.toFixed(3)}</TableCell><TableCell>{r.mc_spread?.toFixed(3) ?? '—'}</TableCell>
      <TableCell>{r.alarm_01pct ? <Chip size="small" color="error" label="0.1 %" /> : r.alarm_1pct ? <Chip size="small" color="warning" label="1 %" /> : <Chip size="small" label="none" />}</TableCell><TableCell>{(r.unk_rate_global * 100).toFixed(0)}</TableCell><TableCell>{r.latency_ms.toFixed(0)} ms</TableCell>
      <TableCell>{r.feedback ? (r.feedback.correct ? 'correct' : 'incorrect') : '—'}</TableCell></TableRow>)}</TableBody></Table>)
}
