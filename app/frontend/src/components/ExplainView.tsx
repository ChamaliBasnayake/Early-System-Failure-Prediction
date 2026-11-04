import { useState } from 'react'
import type { Explanation } from '../lib/api'

// Red = pushed the score up (towards failure), blue = pulled it down. Stronger colour = bigger effect.
function colour(attribution: number, max: number) {
  const strength = Math.min(1, Math.abs(attribution) / (max || 1))
  const alpha = 0.10 + 0.55 * strength
  return attribution >= 0 ? `rgba(214, 69, 61, ${alpha})` : `rgba(52, 112, 178, ${alpha})`
}

export default function ExplainView({ explanation }: { explanation: Explanation }) {
  const [showDetails, setShowDetails] = useState(false)

  const maxLocal = Math.max(...explanation.items_local.map(i => Math.abs(i.attribution)), 1e-6)
  const maxGlobal = Math.max(...explanation.items_global.map(i => Math.abs(i.attribution)), 1e-6)

  // A raw log line gets the effect of the collapsed item it belongs to.
  function effectOfLine(item: number | null) {
    if (item === null) return 0
    return explanation.items_local[item]?.attribution ?? 0
  }

  const topGlobal = [...explanation.items_global]
    .filter(i => i.attribution > 0)
    .sort((a, b) => b.attribution - a.attribution)
    .slice(0, 5)

  const total = Math.abs(explanation.sum_local) + Math.abs(explanation.sum_global) + Math.abs(explanation.sum_scalars) || 1
  const share = (x: number) => `${(100 * Math.abs(x) / total).toFixed(0)} %`

  return (
    <div className="explain">
      <div className="explain-legend">
        <span><i className="swatch swatch-up" /> pushed the score up</span>
        <span><i className="swatch swatch-down" /> pulled it down</span>
        <span className="muted">Without any log lines the score would be {explanation.baseline_score.toFixed(2)}.</span>
      </div>

      <h3>Messages from this midplane</h3>
      {explanation.lines.length === 0 ? (
        <p className="empty-note">This midplane logged nothing in the last 60 minutes. The score comes from machine-wide messages and the time/volume context.</p>
      ) : (
        <div className="scroll">
          <table>
            <thead>
              <tr><th>Time</th><th>Message</th><th className="num">Effect</th></tr>
            </thead>
            <tbody>
              {explanation.lines.map((line, i) => {
                const effect = effectOfLine(line.item)
                return (
                  <tr key={i} style={{ background: line.item === null ? undefined : colour(effect, maxLocal) }}>
                    <td className="nowrap">{new Date(line.timestamp).toLocaleTimeString()}</td>
                    <td className="mono">{line.message.slice(0, 120)}{line.template_index === 1 ? ' [unknown type]' : ''}</td>
                    <td className="num">{line.item === null ? '' : (effect >= 0 ? '+' : '') + effect.toFixed(2)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      <h3>Machine-wide messages that raised the score most</h3>
      {topGlobal.length === 0 ? (
        <p className="empty-note">None.</p>
      ) : (
        <table>
          <tbody>
            {topGlobal.map(item => (
              <tr key={item.item} style={{ background: colour(item.attribution, maxGlobal) }}>
                <td className="mono">{item.text.slice(0, 110)}</td>
                <td className="num">×{item.count}</td>
                <td className="num">+{item.attribution.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <button className="btn btn-ghost" onClick={() => setShowDetails(!showDetails)}>
        {showDetails ? 'Hide' : 'Show'} technical details
      </button>
      {showDetails && (
        <p className="tech">
          Method: Integrated Gradients, {explanation.steps} steps, baseline = empty window. Completeness gap {explanation.completeness_gap.toFixed(3)} logits
          (how much of the score change the attributions fail to account for). Share of attribution: this midplane {share(explanation.sum_local)},
          machine-wide {share(explanation.sum_global)}, time/volume context {share(explanation.sum_scalars)}.
          Context features: {explanation.scalars.slice(0, 5).map(s => `${s.feature} ${s.attribution >= 0 ? '+' : ''}${s.attribution.toFixed(2)}`).join(', ')}.
          The attribution shows what the model reacted to; it does not prove the cause of a failure.
        </p>
      )}
    </div>
  )
}
