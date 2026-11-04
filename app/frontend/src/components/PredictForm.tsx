import { useMemo, useState } from 'react'

type Props = {
  onSubmit: (logText: string, unit: string | null, leadTime: number, mcSamples: number | null) => void
  onInputChange: () => void   // called whenever the input changes, so old results are cleared
  busy: boolean
}

// Find the midplanes mentioned in the pasted log, e.g. "R02-M1" from the node id "R02-M1-N0-C:J12-U11".
function findMidplanes(logText: string): string[] {
  const found = new Set<string>()
  for (const match of logText.matchAll(/\b(R\d{2}-M\d)-/g)) {
    found.add(match[1])
  }
  return Array.from(found).sort()
}

export default function PredictForm({ onSubmit, onInputChange, busy }: Props) {
  const [logText, setLogText] = useState('')
  const [fileName, setFileName] = useState('')
  const [unit, setUnit] = useState('')
  const [leadTime, setLeadTime] = useState(5)
  const [useMonteCarlo, setUseMonteCarlo] = useState(true)

  const midplanes = useMemo(() => findMidplanes(logText), [logText])
  const lineCount = logText.trim() === '' ? 0 : logText.trim().split('\n').length

  function changeText(text: string) {
    setLogText(text)
    // if the chosen midplane is not in the new log any more, go back to "All detected"
    if (unit !== '' && !findMidplanes(text).includes(unit)) setUnit('')
    onInputChange()
  }

  // Read an uploaded .log file into the text box.
  function handleFile(input: HTMLInputElement) {
    const file = input.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      setFileName(file.name)
      changeText(String(reader.result ?? ''))
    }
    reader.readAsText(file)
    input.value = ''   // allows the same file to be chosen again
  }

  function handleClear() {
    setLogText('')
    setFileName('')
    setUnit('')
    onInputChange()
  }

  return (
    <div>
      <div className="form-header">
        <div>
          <h2>Log input</h2>
          <p className="muted">Paste or upload the last ~60 minutes of BlueGene/L RAS log lines.</p>
        </div>
        <div className="form-header-actions">
          <label className="btn btn-secondary">
            Upload .log
            <input type="file" accept=".log,.txt" hidden onChange={e => handleFile(e.target)} />
          </label>
          <button className="btn btn-ghost" onClick={handleClear} disabled={logText === ''}>Clear</button>
        </div>
      </div>

      <textarea
        rows={10}
        value={logText}
        onChange={e => changeText(e.target.value)}
        placeholder="- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
      />
      <p className="muted small">
        {lineCount} lines · {midplanes.length} midplane(s) found{fileName && <> · from {fileName}</>}
      </p>

      <div className="controls">
        <label className="field">
          <span>Midplane</span>
          <select value={unit} onChange={e => { setUnit(e.target.value); onInputChange() }}>
            <option value="">All detected ({midplanes.length})</option>
            {midplanes.map(u => <option key={u} value={u}>{u}</option>)}
          </select>
        </label>

        <label className="field">
          <span>Lead time</span>
          <select value={leadTime} onChange={e => { setLeadTime(Number(e.target.value)); onInputChange() }}>
            {[1, 5, 15, 30].map(n => <option key={n} value={n}>{n} min</option>)}
          </select>
        </label>

        <label className="checkbox">
          <input type="checkbox" checked={useMonteCarlo} onChange={e => { setUseMonteCarlo(e.target.checked); onInputChange() }} />
          Monte-Carlo stability (20 passes)
        </label>

        <button
          className="btn btn-primary"
          disabled={busy || logText.trim() === ''}
          onClick={() => onSubmit(logText, unit === '' ? null : unit, leadTime, useMonteCarlo ? null : 0)}
        >
          {busy ? 'Scoring…' : 'Predict'}
        </button>
      </div>
    </div>
  )
}
