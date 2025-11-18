
interface SiteScore {
  site_id: number
  rank: number
  total_score: number
  components: Record<string, number>
  explanation: string
}

interface ExplainabilityPanelProps {
  site: SiteScore | null
  weights: Record<string, number>
}

export default function ExplainabilityPanel({ site, weights }: ExplainabilityPanelProps) {
  if (!site) {
    return (
      <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
        <h3 className="font-semibold mb-2">Score Explanation</h3>
        <p className="text-sm text-gray-400">Select a site to see score breakdown</p>
      </div>
    )
  }

  // Sort components by contribution (weighted value)
  const componentEntries = Object.entries(site.components)
    .map(([key, value]) => {
      const weight = weights[key] || 0
      const contribution = value * weight
      return { key, value, weight, contribution }
    })
    .sort((a, b) => b.contribution - a.contribution)

  const maxContribution = Math.max(...componentEntries.map(c => Math.abs(c.contribution)), 0.01)

  return (
    <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
      <h3 className="font-semibold mb-3">Score Explanation</h3>
      
      {/* Site Info */}
      <div className="mb-4 pb-4 border-b border-gray-700">
        <div className="text-sm">
          <div className="flex justify-between mb-1">
            <span className="text-gray-400">Site ID:</span>
            <span className="font-semibold">{site.site_id}</span>
          </div>
          <div className="flex justify-between mb-1">
            <span className="text-gray-400">Rank:</span>
            <span className="font-semibold">#{site.rank}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Total Score:</span>
            <span className="font-semibold text-blue-400">{(site.total_score * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* Plain Language Explanation */}
      <div className="mb-4 pb-4 border-b border-gray-700">
        <h4 className="text-sm font-semibold mb-2 text-gray-300">Summary</h4>
        <p className="text-sm text-gray-300 leading-relaxed">{site.explanation}</p>
      </div>

      {/* Component Breakdown */}
      <div>
        <h4 className="text-sm font-semibold mb-3 text-gray-300">Component Contributions</h4>
        <div className="space-y-3">
          {componentEntries.map(({ key, value, weight, contribution }) => {
            const displayName = key
              .replace(/_/g, ' ')
              .replace(/\b\w/g, l => l.toUpperCase())
            
            const barWidth = Math.abs(contribution) / maxContribution * 100
            const isPositive = contribution >= 0

            return (
              <div key={key} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">{displayName}</span>
                  <span className="text-gray-300">
                    {(contribution * 100).toFixed(1)}% contribution
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        isPositive ? 'bg-green-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                  <div className="text-xs text-gray-500 w-20 text-right">
                    Value: {typeof value === 'number' ? value.toFixed(2) : value}
                  </div>
                </div>
                <div className="text-xs text-gray-500">
                  Weight: {(weight * 100).toFixed(0)}%
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

