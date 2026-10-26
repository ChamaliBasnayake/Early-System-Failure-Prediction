import { useState } from 'react'
import { Alert, Box, Button, Card, CardContent, Chip, Collapse, LinearProgress, Stack, Table, TableBody, TableCell, TableHead, TableRow, Typography } from '@mui/material'
import type { UnitResult } from '../lib/api'

export default function ResultCard({ r, onFeedback }: { r: UnitResult; onFeedback: (id: number, ok: boolean) => void }) {
  const [open, setOpen] = useState(false); const [fb, setFb] = useState<boolean | null>(null)
  const pct = Math.min(100, Math.round(r.score * 100)); const a01 = r.alarm_at_far['0.001']; const a1 = r.alarm_at_far['0.01']
  const colour = a01 ? 'error' : a1 ? 'warning' : 'success'
  return (
    <Card variant="outlined" sx={{ borderColor: a01 ? 'error.main' : a1 ? 'warning.main' : 'divider' }}>
      <CardContent>
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
          <Typography variant="h6">{r.unit}</Typography>
          <Chip color={colour} label={a01 ? 'ALARM (0.1 % FAR)' : a1 ? 'Attention (1 % FAR)' : 'No alarm'} />
          {r.out_of_distribution && <Chip color="default" variant="outlined" label="out of distribution" />}
          <Box flex={1} />
          <Typography variant="body2" color="text.secondary">window {new Date(r.window_start).toLocaleTimeString()} – {new Date(r.window_end).toLocaleTimeString()}</Typography>
        </Stack>
        <Box mt={1.5}><Typography variant="body2">Model score <b>{r.score.toFixed(3)}</b>{r.mc_spread !== null && <> &nbsp;· MC spread ±{r.mc_spread.toFixed(3)} ({r.mc_samples} passes)</>} &nbsp;· thresholds: 0.1 % FAR {r.thresholds['0.001']?.toFixed(3)}, 1 % FAR {r.thresholds['0.01']?.toFixed(3)}</Typography>
          <LinearProgress variant="determinate" value={pct} color={colour} sx={{ height: 10, borderRadius: 5, mt: 0.5 }} /></Box>
        <Typography variant="body1" mt={1.5}><b>Recommendation:</b> {r.recommendation}</Typography>
        <Typography variant="body2" color="text.secondary" mt={1}>Local lines (this midplane): {r.n_local} · global lines: {r.n_global} · unknown-template rate local {(r.unk_rate_local * 100).toFixed(0)} % / global {(r.unk_rate_global * 100).toFixed(0)} % · texts never seen in training: {r.n_new_texts}</Typography>
        {r.out_of_distribution && <Alert severity="warning" sx={{ mt: 1 }}>More than half of the lines belong to message types the model never saw in training (or the window is empty). The score is unreliable for this input.</Alert>}
        <Stack direction="row" spacing={1} mt={1.5}>
          <Button size="small" onClick={() => setOpen(o => !o)}>{open ? 'Hide' : 'Show'} local lines ({r.local_lines.length})</Button>
          <Box flex={1} />
          <Typography variant="body2" sx={{ alignSelf: 'center' }}>Was this prediction correct?</Typography>
          <Button size="small" variant={fb === true ? 'contained' : 'outlined'} onClick={() => { setFb(true); onFeedback(r.prediction_id, true) }}>Yes</Button>
          <Button size="small" variant={fb === false ? 'contained' : 'outlined'} color="error" onClick={() => { setFb(false); onFeedback(r.prediction_id, false) }}>No</Button>
        </Stack>
        <Collapse in={open}><Table size="small" sx={{ mt: 1 }}><TableHead><TableRow><TableCell>time</TableCell><TableCell>node</TableCell><TableCell>tpl</TableCell><TableCell>message</TableCell></TableRow></TableHead>
          <TableBody>{r.local_lines.map((l, i) => <TableRow key={i} sx={{ bgcolor: l.template_index === 1 ? '#fff4e5' : undefined }}><TableCell>{new Date(l.timestamp).toLocaleTimeString()}</TableCell><TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{l.node}</TableCell><TableCell>{l.template_index === 1 ? 'UNK' : l.template_index}</TableCell><TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{l.message.slice(0, 140)}</TableCell></TableRow>)}</TableBody></Table>
          <Typography variant="caption" color="text.secondary">Highlighted rows are message types unseen in training. Attribution highlighting (which lines drove the score) arrives with the explanation feature (Phase 8).</Typography></Collapse>
      </CardContent>
    </Card>
  )
}
