// Small wrapper around the backend REST API. All calls go to /api/... (proxied to the FastAPI server).

export type LogLine = {
  timestamp: string
  node: string
  unit: string
  template_index: number
  message: string
}

export type UnitResult = {
  prediction_id: number
  unit: string
  score: number
  mc_spread: number | null
  mc_samples: number
  alarm_at_far: Record<string, boolean>
  thresholds: Record<string, number>
  recommendation: string
  n_local: number
  n_global: number
  unk_rate_local: number
  unk_rate_global: number
  n_new_texts: number
  out_of_distribution: boolean
  new_texts_embedded: number
  encoder: string
  window_start: string
  window_end: string
  local_lines: LogLine[]
}

export type PredictResponse = {
  model_run: string
  model_version: string
  lead_time_requested: number
  lead_time_used: number
  fallback_lead_time: boolean
  n_input_lines: number
  n_parsed: number
  n_malformed: number
  units_detected: string[]
  latency_ms: number
  score_meaning: string
  limitations: string[]
  results: UnitResult[]
}

export type HistoryRow = {
  id: number
  created_at: string
  unit: string
  lead_time: number
  score: number
  mc_spread: number | null
  alarm_1pct: boolean
  alarm_01pct: boolean
  unk_rate_global: number
  latency_ms: number
  feedback: { correct: boolean; note: string | null } | null
}

export type Explanation = {
  unit: string
  meaning: string
  score: number
  baseline_score: number
  completeness_gap: number
  steps: number
  items_local: { item: number; count: number; attribution: number; text: string }[]
  items_global: { item: number; count: number; attribution: number; text: string }[]
  scalars: { feature: string; attribution: number }[]
  sum_local: number
  sum_global: number
  sum_scalars: number
  lines: { timestamp: string; node: string; template_index: number; message: string; item: number | null }[]
}

// Turn a fetch Response into JSON, or throw an Error with the backend's message.
async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail ?? response.statusText)
  }
  return response.json()
}

function postJson<T>(url: string, body: object): Promise<T> {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(readJson<T>)
}

export function getModelInfo() {
  return fetch('/api/model').then(readJson<any>)
}

export function predict(logText: string, unit: string | null, leadTime: number, mcSamples: number | null) {
  return postJson<PredictResponse>('/api/predict', { log_text: logText, unit, lead_time: leadTime, mc_samples: mcSamples })
}

export function explain(logText: string, unit: string, leadTime: number) {
  return postJson<Explanation>('/api/explain', { log_text: logText, unit, lead_time: leadTime })
}

export function sendFeedback(predictionId: number, correct: boolean) {
  return postJson<any>('/api/feedback', { prediction_id: predictionId, correct })
}

export function getHistory() {
  return fetch('/api/history?limit=100').then(readJson<HistoryRow[]>)
}
