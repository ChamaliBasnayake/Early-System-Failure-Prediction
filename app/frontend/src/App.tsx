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

  // Any change to the input makes the old result out of date, so remove it.
  function clearResult() {
    setResult(null)
    setError(null)
    setShowLimits(false)
  }

  async function handlePredict(logText: string, unit: string | null, leadTime: number, mcSamples: number | null) {
    clearResult()
    setBusy(true)
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
      <header className="topbar">
        <div className="brand">
          <div className="logo">BG/L</div>
          <div>
            <h1>Early failure prediction</h1>
            <p>IBM BlueGene/L · RAS log monitor for system administrators</p>
          </div>
        </div>
        <span className={`status ${modelInfo ? 'online' : 'offline'}`}>
          {modelInfo ? 'Model loaded · sem_final' : 'Backend offline'}
        </span>
      </header>

      <main>
        <nav className="tabs">
          <button className={page === 'predict' ? 'active' : ''} onClick={() => setPage('predict')}>Predict</button>
          <button className={page === 'history' ? 'active' : ''} onClick={() => setPage('history')}>History</button>
          <button className={page === 'about' ? 'active' : ''} onClick={() => setPage('about')}>About the model</button>
        </nav>

        {error && <div className="alert alert-error">{error}</div>}

        {/* The form stays mounted when switching tabs, so the pasted log is kept. */}
        <div style={{ display: page === 'predict' ? 'block' : 'none' }}>
          <section className="panel">
            <PredictForm onSubmit={handlePredict} onInputChange={clearResult} busy={busy} />
          </section>

          {busy && <div className="alert alert-info">Scoring… the model is running on the backend.</div>}

          {result && (
            <>
              <div className="summary">
                <div><b>{result.results.length}</b><span>midplane(s) scored</span></div>
                <div><b>{result.lead_time_used} min</b><span>lead time</span></div>
                <div><b>{result.n_parsed} / {result.n_input_lines}</b><span>lines understood</span></div>
                <div><b>{result.latency_ms.toFixed(0)} ms</b><span>response time</span></div>
              </div>
              {result.fallback_lead_time && (
                <div className="alert alert-warning">No model for {result.lead_time_requested} min — the closest one was used.</div>
              )}

              {result.results.map(r => (
                <ResultCard key={r.prediction_id} result={r} logText={lastLog} leadTime={lastLeadTime} />
              ))}

              <section className="panel">
                <button className="btn btn-ghost" onClick={() => setShowLimits(!showLimits)}>
                  {showLimits ? 'Hide' : 'Show'} model limitations
                </button>
                {showLimits && (
                  <ul className="limits">
                    {result.limitations.map((text, i) => <li key={i}>{text}</li>)}
                  </ul>
                )}
              </section>
            </>
          )}
        </div>

        {page === 'history' && (
          <section className="panel">
            <h2>Prediction history</h2>
            <HistoryTable rows={history} />
          </section>
        )}

        {page === 'about' && (
          <section className="panel">
            {modelInfo ? (
              <>
                <h2>{modelInfo.version}</h2>
                <p>{modelInfo.score_meaning}</p>
                <h3>Loaded models</h3>
                <table>
                  <thead>
                    <tr><th>Lead time</th><th>Checkpoint</th><th>Parameters</th><th>Attention above</th><th>Alarm above</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(modelInfo.models).map(([n, m]: [string, any]) => (
                      <tr key={n}>
                        <td>{n} min</td>
                        <td className="mono">{m.run}</td>
                        <td>{m.n_params?.toLocaleString()}</td>
                        <td>{m.thresholds['0.01']?.toFixed(2)}</td>
                        <td>{m.thresholds['0.001']?.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <h3>Limitations</h3>
                <ul className="limits">
                  {modelInfo.limitations.map((text: string, i: number) => <li key={i}>{text}</li>)}
                </ul>
              </>
            ) : (
              <p>Backend offline.</p>
            )}
          </section>
        )}
      </main>
    </>
  )
}
