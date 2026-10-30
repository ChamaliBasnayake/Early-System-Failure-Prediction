import { useEffect, useState } from 'react'
import PredictForm from './components/PredictForm'
import ResultCard from './components/ResultCard'
import HistoryTable from './components/HistoryTable'
import { getModelInfo, getHistory, predict, type HistoryRow, type PredictResponse } from './lib/api'

type Page = 'predict' | 'history' | 'about'

export default function App() {
  const [page, setPage] = useState<Page>('predict')
  const [modelInfo, setModelInfo] = useState<any>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [lastLog, setLastLog] = useState('')
  const [lastLeadTime, setLastLeadTime] = useState(5)
  const [history, setHistory] = useState<HistoryRow[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showLimits, setShowLimits] = useState(false)

  // Ask the backend which model is loaded, once at start-up.
  useEffect(() => {
    getModelInfo().then(setModelInfo).catch(() => setModelInfo(null))
  }, [])

  // Reload the history table every time the user opens that page.
  useEffect(() => {
    if (page === 'history') {
      getHistory().then(setHistory).catch(e => setError(e.message))
    }
  }, [page])

  async function handlePredict(logText: string, unit: string | null, leadTime: number, mcSamples: number | null) {
    setBusy(true)
    setError(null)
    try {
      const response = await predict(logText, unit, leadTime, mcSamples)
      setResult(response)
      setLastLog(logText)
      setLastLeadTime(leadTime)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <header>
        <h1>BlueGene/L early failure prediction</h1>
        <span>{modelInfo ? modelInfo.version : 'backend offline'}</span>
      </header>

      <main>
        <nav>
          <button className={page === 'predict' ? 'active' : ''} onClick={() => setPage('predict')}>Predict</button>
          <button className={page === 'history' ? 'active' : ''} onClick={() => setPage('history')}>History</button>
          <button className={page === 'about' ? 'active' : ''} onClick={() => setPage('about')}>About the model</button>
        </nav>

        {error && <div className="error">{error}</div>}

        {page === 'predict' && (
          <>
            <div className="box">
              <PredictForm onSubmit={handlePredict} busy={busy} units={result ? result.units_detected : []} />
            </div>

            {result && (
              <>
                <div className="info">
                  Scored {result.results.length} midplane(s) for failures starting {result.lead_time_used} minutes ahead
                  ({result.n_parsed} of {result.n_input_lines} lines understood, {result.latency_ms.toFixed(0)} ms).
                  {result.fallback_lead_time && <b> No model for {result.lead_time_requested} min — the closest one was used.</b>}
                </div>

                {result.results.map(r => (
                  <ResultCard key={r.prediction_id} result={r} logText={lastLog} leadTime={lastLeadTime} />
                ))}

                <div className="box">
                  <button className="small" onClick={() => setShowLimits(!showLimits)}>
                    {showLimits ? 'Hide' : 'Show'} model limitations
                  </button>
                  {showLimits && (
                    <ul>
                      {result.limitations.map((text, i) => <li key={i}>{text}</li>)}
                    </ul>
                  )}
                </div>
              </>
            )}
          </>
        )}

        {page === 'history' && (
          <div className="box">
            <HistoryTable rows={history} />
          </div>
        )}

        {page === 'about' && (
          <div className="box">
            {modelInfo ? (
              <>
                <h3>{modelInfo.version}</h3>
                <p>{modelInfo.score_meaning}</p>
                <h3>Loaded checkpoints</h3>
                <pre style={{ fontSize: 12 }}>{JSON.stringify(modelInfo.models, null, 2)}</pre>
                <h3>Limitations</h3>
                <ul>
                  {modelInfo.limitations.map((text: string, i: number) => <li key={i}>{text}</li>)}
                </ul>
              </>
            ) : (
              <p>Backend offline.</p>
            )}
          </div>
        )}
      </main>
    </>
  )
}
