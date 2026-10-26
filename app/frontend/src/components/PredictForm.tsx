import { useState } from 'react'
import { Box, Button, FormControl, FormControlLabel, InputLabel, MenuItem, Select, Stack, Switch, TextField, Typography } from '@mui/material'
import UploadFileIcon from '@mui/icons-material/UploadFile'

type Props = { onSubmit: (log: string, unit: string | null, N: number, mc: number | null) => void; busy: boolean; units: string[] }
export default function PredictForm({ onSubmit, busy, units }: Props) {
  const [log, setLog] = useState(''); const [unit, setUnit] = useState<string>(''); const [N, setN] = useState(5); const [mc, setMc] = useState(true)
  const onFile = (f: File | null) => { if (!f) return; const r = new FileReader(); r.onload = () => setLog(String(r.result ?? '')); r.readAsText(f) }
  return (
    <Stack spacing={2}>
      <Typography variant="body2" color="text.secondary">Paste (or upload) the last ~60 minutes of BlueGene/L RAS log lines. The system detects the midplanes present and scores each one, or you can pick one.</Typography>
      <TextField multiline minRows={10} maxRows={18} value={log} onChange={e => setLog(e.target.value)} placeholder="- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected" inputProps={{ style: { fontFamily: 'monospace', fontSize: 12 } }} />
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems="center">
        <Button component="label" variant="outlined" startIcon={<UploadFileIcon />}>Upload .log<input hidden type="file" accept=".log,.txt" onChange={e => onFile(e.target.files?.[0] ?? null)} /></Button>
        <FormControl size="small" sx={{ minWidth: 180 }}><InputLabel>Midplane</InputLabel>
          <Select label="Midplane" value={unit} onChange={e => setUnit(e.target.value)}><MenuItem value="">All detected</MenuItem>{units.map(u => <MenuItem key={u} value={u}>{u}</MenuItem>)}</Select></FormControl>
        <FormControl size="small" sx={{ minWidth: 160 }}><InputLabel>Lead time</InputLabel>
          <Select label="Lead time" value={N} onChange={e => setN(Number(e.target.value))}>{[1, 5, 15, 30].map(n => <MenuItem key={n} value={n}>{n} min</MenuItem>)}</Select></FormControl>
        <FormControlLabel control={<Switch checked={mc} onChange={e => setMc(e.target.checked)} />} label="Monte-Carlo spread (20 passes)" />
        <Box flex={1} />
        <Button variant="contained" disabled={busy || !log.trim()} onClick={() => onSubmit(log, unit || null, N, mc ? null : 0)}>{busy ? 'Scoring…' : 'Predict'}</Button>
      </Stack>
    </Stack>
  )
}
