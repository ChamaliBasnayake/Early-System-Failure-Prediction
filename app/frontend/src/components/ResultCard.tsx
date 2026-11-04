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
  const [showExplanation, setShowExplanation] = useState(false)
  const [explaining, setExplaining] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Two operating points, both chosen on validation data:
  //   0.1 % FAR ≈ 1.4 false alarms per midplane per day  -> "ALARM"
  //   1 % FAR   ≈ 14 false alarms per midplane per day   -> "Attention"
  const alarm = result.alarm_at_far['0.001']
  const attention = result.alarm_at_far['0.01']
  const level = alarm ? 'alarm' : attention ? 'attention' : 'ok'
  const label = alarm ? 'ALARM' : attention ? 'Attention' : 'No alarm'
  const attentionAt = result.thresholds['0.01'] ?? 0
  const alarmAt = result.thresholds['0.001'] ?? 0

  async function handleExplain() {
    if (explanation) {                 // already fetched: just show / hide it
      setShowExplanation(!showExplanation)
      return
    }
    setExplaining(true)
    setError(null)
    try {
      setExplanation(await explain(logText, result.unit, leadTime))
      setShowExplanation(true)
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
    <section className={`panel result result-${level}`}>
      <div className="result-head">
        <div>
          <span className="eyebrow">Midplane</span>
          <h2>{result.unit}</h2>
        </div>
        <div className="badges">
          {result.out_of_distribution && <span className="badge badge-grey">Input unlike training data</span>}
          <span className={`badge badge-${level}`}>{label}</span>
        </div>
      </div>

      <div className="result-body">
        <div className="score">
          <span className="score-value">{result.score.toFixed(2)}</span>
          <span className="score-label">risk score (0–1)</span>
          {result.mc_spread !== null && <span className="score-label">stability ±{result.mc_spread.toFixed(2)}</span>}
        </div>

        <div className="score-detail">
          {/* bar with the two thresholds marked on it */}
          <div className="meter">
            <div className={`meter-fill meter-${level}`} style={{ width: `${Math.min(100, result.score * 100)}%` }} />
            <div className="meter-mark" style={{ left: `${attentionAt * 100}%` }} title="Attention threshold" />
            <div className="meter-mark meter-mark-alarm" style={{ left: `${alarmAt * 100}%` }} title="Alarm threshold" />
          </div>
          <div className="meter-legend">
            <span>Attention above {attentionAt.toFixed(2)} (~14 false alarms / day)</span>
            <span>Alarm above {alarmAt.toFixed(2)} (~1.4 / day)</span>
          </div>
          <p className="recommendation"><b>Recommendation:</b> {result.recommendation}</p>
          <p className="muted small">
            The score ranks risk; it is not a probability. Last 60 minutes: {result.n_local} lines from this midplane,
            {' '}{result.n_global} from the whole machine; {(result.unk_rate_global * 100).toFixed(0)} % of messages never seen in training.
          </p>
        </div>
      </div>

      {result.out_of_distribution && (
        <div className="alert alert-warning">
          Most of these messages are unknown to the model (or the window is empty), so the score is not reliable here.
        </div>
      )}

      <div className="result-actions">
        <button className="btn btn-secondary" disabled={explaining} onClick={handleExplain}>
          {explaining ? 'Explaining…' : showExplanation ? 'Hide explanation' : 'Explain this score'}
        </button>
        <div className="feedback">
          <span className="muted">Was this prediction correct?</span>
          <button className={`btn btn-chip ${feedback === true ? 'chip-yes' : ''}`} onClick={() => handleFeedback(true)}>Yes</button>
          <button className={`btn btn-chip ${feedback === false ? 'chip-no' : ''}`} onClick={() => handleFeedback(false)}>No</button>
        </div>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {explanation && showExplanation && <ExplainView explanation={explanation} />}
    </section>
  )
}
