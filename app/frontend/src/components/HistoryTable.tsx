import type { HistoryRow } from '../lib/api'

export default function HistoryTable({ rows }: { rows: HistoryRow[] }) {
  if (rows.length === 0) {
    return <p className="muted">No predictions stored yet.</p>
  }
  return (
    <table>
      <thead>
        <tr>
          <th>#</th><th>time</th><th>midplane</th><th>lead</th><th>score</th><th>spread</th><th>alarm</th><th>UNK %</th><th>latency</th><th>feedback</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(r => (
          <tr key={r.id}>
            <td>{r.id}</td>
            <td>{new Date(r.created_at).toLocaleString()}</td>
            <td>{r.unit}</td>
            <td>{r.lead_time} min</td>
            <td>{r.score.toFixed(3)}</td>
            <td>{r.mc_spread === null ? '—' : r.mc_spread.toFixed(3)}</td>
            <td>{r.alarm_01pct ? 'ALARM 0.1 %' : r.alarm_1pct ? 'attention 1 %' : 'none'}</td>
            <td>{(r.unk_rate_global * 100).toFixed(0)}</td>
            <td>{r.latency_ms.toFixed(0)} ms</td>
            <td>{r.feedback ? (r.feedback.correct ? 'correct' : 'incorrect') : '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
