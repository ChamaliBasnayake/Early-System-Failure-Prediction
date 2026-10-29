import { useEffect, useState } from 'react'
import { Alert, AppBar, Box, Container, Paper, Stack, Tab, Tabs, Toolbar, Typography } from '@mui/material'
import PredictForm from './components/PredictForm'
import ResultCard from './components/ResultCard'
import HistoryTable from './components/HistoryTable'
import { api, type HistoryRow, type PredictResponse } from './lib/api'

export default function App() {
  const [tab, setTab] = useState(0); const [busy, setBusy] = useState(false); const [err, setErr] = useState<string | null>(null)
  const [res, setRes] = useState<PredictResponse | null>(null); const [hist, setHist] = useState<HistoryRow[]>([]); const [model, setModel] = useState<any>(null)
  const [lastReq, setLastReq] = useState<{ log: string; N: number } | null>(null)
  useEffect(() => { api.model().then(setModel).catch(() => setModel(null)) }, [])
  useEffect(() => { if (tab === 1) api.history().then(setHist).catch(e => setErr(String(e.message))) }, [tab])
  const submit = async (log: string, unit: string | null, N: number, mc: number | null) => {
    setBusy(true); setErr(null)
    try { setRes(await api.predict(log, unit, N, mc)); setLastReq({ log, N }) } catch (e: any) { setErr(e.message) } finally { setBusy(false) }
  }
  return (<>
    <AppBar position="static"><Toolbar><Typography variant="h6">BGL early failure prediction</Typography><Box flex={1} /><Typography variant="body2">{model?.version ?? 'backend offline'}</Typography></Toolbar></AppBar>
    <Container maxWidth="lg" sx={{ py: 3 }}>
      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}><Tab label="Predict" /><Tab label="History" /><Tab label="About the model" /></Tabs>
      {err && <Alert severity="error" sx={{ mb: 2 }} onClose={() => setErr(null)}>{err}</Alert>}
      {tab === 0 && <Stack spacing={3}>
        <Paper sx={{ p: 2 }}><PredictForm onSubmit={submit} busy={busy} units={res?.units_detected ?? []} /></Paper>
        {res && <>
          <Alert severity={res.fallback_lead_time ? 'warning' : 'info'}>Scored {res.results.length} midplane(s) in {res.latency_ms.toFixed(0)} ms with <b>{res.model_run}</b> at lead time {res.lead_time_used} min{res.fallback_lead_time && <> (no model for {res.lead_time_requested} min yet — fallback)</>}. Parsed {res.n_parsed} of {res.n_input_lines} lines ({res.n_malformed} malformed).</Alert>
          <Alert severity="info" icon={false}><b>What the score means.</b> {res.score_meaning}</Alert>
          {res.results.map(r => <ResultCard key={r.prediction_id} r={r} onFeedback={(id, ok) => api.feedback(id, ok).catch(e => setErr(e.message))} onExplain={(u) => api.explain(lastReq!.log, u, lastReq!.N)} />)}
          <Paper variant="outlined" sx={{ p: 2 }}><Typography variant="subtitle2" gutterBottom>Limitations</Typography><ul style={{ margin: 0 }}>{res.limitations.map((l, i) => <li key={i}><Typography variant="body2">{l}</Typography></li>)}</ul></Paper>
        </>}
      </Stack>}
      {tab === 1 && <Paper sx={{ p: 2 }}><HistoryTable rows={hist} /></Paper>}
      {tab === 2 && <Paper sx={{ p: 2 }}>{model ? <>
        <Typography variant="subtitle1" gutterBottom>{model.version}</Typography>
        <Typography variant="body2" paragraph>{model.score_meaning}</Typography>
        <Typography variant="subtitle2">Loaded checkpoints</Typography><pre style={{ fontSize: 12 }}>{JSON.stringify(model.models, null, 2)}</pre>
        <Typography variant="subtitle2">Limitations</Typography><ul>{model.limitations.map((l: string, i: number) => <li key={i}><Typography variant="body2">{l}</Typography></li>)}</ul>
      </> : <Typography>Backend offline.</Typography>}</Paper>}
    </Container>
  </>)
}
