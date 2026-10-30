import { useState } from 'react'

type Props = {
  onSubmit: (logText: string, unit: string | null, leadTime: number, mcSamples: number | null) => void
  busy: boolean
  units: string[]
}

export default function PredictForm({ onSubmit, busy, units }: Props) {
  const [logText, setLogText] = useState('')
  const [unit, setUnit] = useState('')
  const [leadTime, setLeadTime] = useState(5)
  const [useMonteCarlo, setUseMonteCarlo] = useState(true)

  // Read an uploaded .log file into the text box.
  function handleFile(file: File | undefined) {
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => setLogText(String(reader.result ?? ''))
    reader.readAsText(file)
  }

  return (
    <div>
      <p className="muted">
        Paste or upload the last ~60 minutes of BlueGene/L RAS log lines. Every midplane found in the log is scored,
        or you can choose one.
      </p>

      <textarea
        rows={10}
        value={logText}
        onChange={e => setLogText(e.target.value)}
        placeholder="- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
      />

      <div className="row">
        <input type="file" accept=".log,.txt" onChange={e => handleFile(e.target.files?.[0])} />

        <label>
          Midplane:{' '}
          <select value={unit} onChange={e => setUnit(e.target.value)}>
            <option value="">All detected</option>
            {units.map(u => <option key={u} value={u}>{u}</option>)}
          </select>
        </label>

        <label>
          Lead time:{' '}
          <select value={leadTime} onChange={e => setLeadTime(Number(e.target.value))}>
            {[1, 5, 15, 30].map(n => <option key={n} value={n}>{n} min</option>)}
          </select>
        </label>

        <label>
          <input type="checkbox" checked={useMonteCarlo} onChange={e => setUseMonteCarlo(e.target.checked)} />
          {' '}Monte-Carlo spread (20 passes)
        </label>

        <button
          className="primary"
          disabled={busy || logText.trim() === ''}
          onClick={() => onSubmit(logText, unit === '' ? null : unit, leadTime, useMonteCarlo ? null : 0)}
        >
          {busy ? 'Scoring…' : 'Predict'}
        </button>
      </div>
    </div>
  )
}
