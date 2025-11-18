import { useState } from 'react'

interface SiteScore {
  site_id: number
  rank: number
  total_score: number
  components: Record<string, number>
  explanation: string
  geometry: any
  centroid_lat: number
  centroid_lon: number
  area_km2: number
}

interface SiteScoresListProps {
  sites: SiteScore[]
  selectedSite: number | null
  onSelectSite: (siteId: number) => void
}

export default function SiteScoresList({ sites, selectedSite, onSelectSite }: SiteScoresListProps) {
  const [expandedSite, setExpandedSite] = useState<number | null>(null)

  return (
    <div className="p-4">
      <h3 className="font-semibold mb-3">Site Rankings</h3>
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {sites.map((site) => (
          <div
            key={site.site_id}
            className={`p-3 rounded border-2 cursor-pointer transition-colors ${
              selectedSite === site.site_id
                ? 'border-blue-500 bg-blue-900/30'
                : 'border-gray-700 bg-gray-700/50 hover:bg-gray-700'
            }`}
            onClick={() => onSelectSite(site.site_id)}
          >
            <div className="flex justify-between items-start mb-2">
              <div>
                <div className="font-semibold text-sm">
                  Rank #{site.rank} - Site {site.site_id}
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  Score: {(site.total_score * 100).toFixed(1)}%
                </div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  setExpandedSite(expandedSite === site.site_id ? null : site.site_id)
                }}
                className="text-gray-400 hover:text-gray-300 text-xs"
              >
                {expandedSite === site.site_id ? '▼' : '▶'}
              </button>
            </div>
            <div className="text-xs text-gray-300 mb-2">{site.explanation}</div>
            {expandedSite === site.site_id && (
              <div className="mt-2 pt-2 border-t border-gray-700 text-xs">
                <div className="text-gray-400 mb-1">Component Breakdown:</div>
                <div className="space-y-1">
                  {Object.entries(site.components).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-gray-400 capitalize">{key.replace('_', ' ')}:</span>
                      <span className="text-gray-300">
                        {typeof value === 'number' ? value.toFixed(2) : value}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="mt-2 pt-2 border-t border-gray-700">
                  <div className="text-gray-400">Area: {site.area_km2.toFixed(2)} km²</div>
                  <div className="text-gray-400">
                    Location: {site.centroid_lat.toFixed(4)}°, {site.centroid_lon.toFixed(4)}°
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

