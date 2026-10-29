import { Box, Chip, Stack, Table, TableBody, TableCell, TableHead, TableRow, Typography, Alert } from '@mui/material'

export type Explanation = {
  unit: string; meaning: string; score: number; baseline_score: number; completeness_gap: number; steps: number
  items_local: { item: number; template_index: number; count: number; attribution: number; text: string }[]
  items_global: { item: number; template_index: number; count: number; attribution: number; text: string }[]
  scalars: { feature: string; attribution: number }[]
  sum_local: number; sum_global: number; sum_scalars: number
  lines: { timestamp: string; node: string; template_index: number; message: string; item: number | null }[]
}
const colour = (a: number, max: number) => { const t = Math.min(1, Math.abs(a) / (max || 1)); return a >= 0 ? `rgba(220, 60, 60, ${0.12 + 0.7 * t})` : `rgba(60, 120, 220, ${0.12 + 0.7 * t})` }

export default function ExplainView({ ex }: { ex: Explanation }) {
  const maxL = Math.max(...ex.items_local.map(i => Math.abs(i.attribution)), 1e-6); const maxG = Math.max(...ex.items_global.map(i => Math.abs(i.attribution)), 1e-6)
  const attrOfLine = (it: number | null) => it === null ? 0 : (ex.items_local[it]?.attribution ?? 0)
  const topG = [...ex.items_global].sort((a, b) => b.attribution - a.attribution).slice(0, 8)
  const share = (x: number) => { const tot = Math.abs(ex.sum_local) + Math.abs(ex.sum_global) + Math.abs(ex.sum_scalars) || 1; return `${(100 * Math.abs(x) / tot).toFixed(0)} %` }
  return (
    <Stack spacing={1.5} mt={1}>
      <Alert severity="info" icon={false}><b>How to read this.</b> {ex.meaning}</Alert>
      <Stack direction="row" spacing={1} flexWrap="wrap">
        <Chip label={`score ${ex.score.toFixed(3)} vs empty window ${ex.baseline_score.toFixed(3)}`} />
        <Chip label={`attribution share — this midplane's lines ${share(ex.sum_local)} · system lines ${share(ex.sum_global)} · time/volume context ${share(ex.sum_scalars)}`} variant="outlined" />
        <Chip label={`completeness gap ${ex.completeness_gap.toFixed(3)} logits (${ex.steps} steps)`} variant="outlined" size="small" />
      </Stack>
      <Typography variant="subtitle2">This midplane's lines, coloured by attribution (red pushes towards failure, blue away)</Typography>
      {ex.lines.length === 0 ? <Typography variant="body2" color="text.secondary">The midplane logged nothing in the window; the score comes from the system channel and the context features.</Typography> :
        <Box sx={{ maxHeight: 320, overflow: 'auto', border: '1px solid #ddd', borderRadius: 1 }}>
          <Table size="small" stickyHeader><TableHead><TableRow><TableCell>time</TableCell><TableCell>node</TableCell><TableCell>attr</TableCell><TableCell>message</TableCell></TableRow></TableHead>
            <TableBody>{ex.lines.map((l, i) => { const a = attrOfLine(l.item); return (<TableRow key={i} sx={{ bgcolor: l.item === null ? undefined : colour(a, maxL) }}>
              <TableCell>{new Date(l.timestamp).toLocaleTimeString()}</TableCell><TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{l.node}</TableCell>
              <TableCell>{l.item === null ? '—' : a.toFixed(3)}</TableCell><TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{l.message.slice(0, 140)}{l.template_index === 1 && <Chip size="small" label="UNK" sx={{ ml: 1 }} />}</TableCell></TableRow>) })}</TableBody></Table></Box>}
      <Typography variant="subtitle2">System-wide messages that pushed the score up most</Typography>
      <Table size="small"><TableBody>{topG.filter(i => i.attribution > 0).map(i => <TableRow key={i.item} sx={{ bgcolor: colour(i.attribution, maxG) }}><TableCell>{i.attribution.toFixed(3)}</TableCell><TableCell>×{i.count}</TableCell><TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{i.text.slice(0, 120)}</TableCell></TableRow>)}</TableBody></Table>
      <Typography variant="subtitle2">Context features</Typography>
      <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 12 }}>{ex.scalars.slice(0, 6).map(s => `${s.feature} ${s.attribution >= 0 ? '+' : ''}${s.attribution.toFixed(3)}`).join('   ')}</Typography>
    </Stack>
  )
}
