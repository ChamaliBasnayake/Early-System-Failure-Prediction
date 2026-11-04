import type { HistoryRow } from '../lib/api'

export default function HistoryTable({ rows }: { rows: HistoryRow[] }) {
  if (rows.length === 0) {
    return <p className="muted">No predictions stored yet.</p>
  }
  return (
    <div className="scroll scroll-tall">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Time</th><th>Midplane</th><th>Lead</th><th className="num">Score</th><th className="num">Stability</th>
            <th>Alarm</th><th className="num">Unknown %</th><th className="num">Latency</th><th>Feedback</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.id}>
              <td>{r.id}</td>
              <td className="nowrap">{new Date(r.created_at).toLocaleString()}</td>
              <td className="mono">{r.unit}</td>
              <td>{r.lead_time} min</td>
              <td className="num">{r.score.toFixed(3)}</td>
              <td className="num">{r.mc_spread === null ? '—' : '±' + r.mc_spread.toFixed(3)}</td>
              <td>
                {r.alarm_01pct ? <span className="badge badge-alarm">ALARM</span>
                  : r.alarm_1pct ? <span className="badge badge-attention">Attention</span>
                  : <span className="badge badge-ok">None</span>}
              </td>
              <td className="num">{(r.unk_rate_global * 100).toFixed(0)}</td>
              <td className="num">{r.latency_ms.toFixed(0)} ms</td>
              <td>{r.feedback ? (r.feedback.correct ? '✔ correct' : '✘ incorrect') : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
