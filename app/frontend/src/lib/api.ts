export type UnitResult = {
  prediction_id: number; unit: string; score: number; mc_spread: number | null; mc_samples: number
  alarm_at_far: Record<string, boolean>; thresholds: Record<string, number>; recommendation: string
  n_local: number; n_global: number; unk_rate_local: number; unk_rate_global: number; n_new_texts: number; out_of_distribution: boolean; new_texts_embedded: number; encoder: string
  window_start: string; window_end: string; local_lines: { timestamp: string; node: string; unit: string; template_index: number; message: string }[]
}
export type PredictResponse = {
  model_run: string; model_version: string; lead_time_requested: number; lead_time_used: number; fallback_lead_time: boolean
  n_input_lines: number; n_parsed: number; n_malformed: number; units_detected: string[]; latency_ms: number
  score_meaning: string; limitations: string[]; results: UnitResult[]
}
export type HistoryRow = { id: number; created_at: string; unit: string; lead_time: number; score: number; mc_spread: number | null; alarm_1pct: boolean; alarm_01pct: boolean; n_local: number; n_global: number; unk_rate_global: number; model_run: string; latency_ms: number; feedback: { correct: boolean; note: string | null } | null }

async function j<T>(r: Response): Promise<T> { if (!r.ok) { const e = await r.json().catch(() => ({ detail: r.statusText })); throw new Error(e.detail ?? r.statusText) } return r.json() }
export const api = {
  health: () => fetch('/api/health').then(j<any>),
  model: () => fetch('/api/model').then(j<any>),
  predict: (log_text: string, unit: string | null, lead_time: number, mc_samples: number | null) =>
    fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ log_text, unit, lead_time, mc_samples }) }).then(j<PredictResponse>),
  explain: (log_text: string, unit: string, lead_time: number) =>
    fetch('/api/explain', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ log_text, unit, lead_time }) }).then(j<any>),
  feedback: (prediction_id: number, correct: boolean, note?: string) =>
    fetch('/api/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prediction_id, correct, note }) }).then(j<any>),
  history: () => fetch('/api/history?limit=100').then(j<HistoryRow[]>),
}
