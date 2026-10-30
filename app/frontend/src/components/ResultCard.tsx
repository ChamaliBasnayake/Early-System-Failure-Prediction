import { useState } from 'react'
import { explain, sendFeedback, type Explanation, type UnitResult } from '../lib/api'
import ExplainView from './ExplainView'

type Props = {
  result: UnitResult
  logText: string
  leadTime: number
}

export default function ResultCard({ result, logText, leadTime }: Props) {
  const [feedback, setFeedback] = useState<boolean | null>(null)
  const [explanation, setExplanation] = useState<Explanation | null>(null)
  const [explaining, setExplaining] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Two operating points, both chosen on validation data:
  //   0.1 % FAR ≈ 1.4 false alarms per midplane per day  -> "ALARM"
  //   1 % FAR   ≈ 14 false alarms per midplane per day   -> "Attention"
  const alarm = result.alarm_at_far['0.001']
  const attention = result.alarm_at_far['0.01']
  const level = alarm ? 'alarm' : attention ? 'attention' : ''
  const label = alarm ? 'ALARM' : attention ? 'Attention' : 'No alarm'
  const percent = Math.min(100, Math.round(result.score * 100))

  async function handleExplain() {
    setExplaining(true)
    setError(null)
    try {
      setExplanation(await explain(logText, result.unit, leadTime))
    } catch (e: any) {
      setError(e.message)
    } finally {
      setExplaining(false)
    }
  }

  function handleFeedback(correct: boolean) {
    setFeedback(correct)
    sendFeedback(result.prediction_id, correct).catch(e => setError(e.message))
  }

  return (
    <div className={`box result ${level}`}>
      <div>
        <b style={{ fontSize: 18 }}>Midplane {result.unit}</b>
        <span className={`badge ${level}`}>{label}</span>
        {result.out_of_distribution && <span className="badge grey">input unlike training data</span>}
      </div>

      <p style={{ margin: '10px 0 2px' }}>
        Risk score <b>{result.score.toFixed(2)}</b> out of 1
        {result.mc_spread !== null && <span className="muted"> &nbsp;(stability ±{result.mc_spread.toFixed(2)})</span>}
      </p>
      <div className="bar"><div style={{ width: `${percent}%` }} /></div>
      <p className="muted" style={{ marginTop: 2 }}>
        Attention above {result.thresholds['0.01']?.toFixed(2)}, alarm above {result.thresholds['0.001']?.toFixed(2)}.
        The score ranks risk; it is not a probability.
      </p>

      <p><b>Recommendation:</b> {result.recommendation}</p>

      <p className="muted">
        Last 60 minutes: {result.n_local} lines from this midplane, {result.n_global} from the whole machine;
        {' '}{(result.unk_rate_global * 100).toFixed(0)} % of messages were never seen in training.
      </p>

      {result.out_of_distribution && (
        <div className="warning">
          Most of these messages are unknown to the model (or the window is empty), so the score is not reliable here.
        </div>
      )}

      <div className="row">
        <button className="small" disabled={explaining} onClick={handleExplain}>
          {explaining ? 'Explaining…' : explanation ? 'Refresh explanation' : 'Explain this score'}
        </button>
        <span style={{ marginLeft: 'auto' }} className="muted">Was this prediction correct?</span>
        <button className={`small ${feedback === true ? 'selected' : ''}`} onClick={() => handleFeedback(true)}>Yes</button>
        <button className={`small ${feedback === false ? 'selected' : ''}`} onClick={() => handleFeedback(false)}>No</button>
      </div>

      {error && <div className="error" style={{ marginTop: 10 }}>{error}</div>}

      {explanation && <ExplainView explanation={explanation} />}
    </div>
  )
}
