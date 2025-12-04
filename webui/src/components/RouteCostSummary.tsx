import { Activity, AlertTriangle, CheckCircle, TrendingUp } from 'lucide-react'

interface RouteCostSummaryProps {
  distance_m: number
  slope_cost: number
  roughness_cost: number
  shadow_cost: number
  energy_estimate_j: number
}

export default function RouteCostSummary({
  distance_m,
  slope_cost,
  roughness_cost,
  shadow_cost,
  energy_estimate_j
}: RouteCostSummaryProps) {
  const totalCost = slope_cost + roughness_cost + shadow_cost
  const maxComponent = Math.max(slope_cost, roughness_cost, shadow_cost, 0.01)

  // Risk Assessment Logic
  let riskLevel: 'LOW' | 'MED' | 'HIGH' = 'LOW'
  let riskColor = 'text-green-400'
  if (totalCost > 50 || slope_cost > 30) {
    riskLevel = 'MED'
    riskColor = 'text-yellow-400'
  }
  if (totalCost > 80 || slope_cost > 50 || roughness_cost > 40) {
    riskLevel = 'HIGH'
    riskColor = 'text-red-500'
  }

  return (
    <div className="glass-panel rounded-lg p-4 text-white font-mono text-sm">
      <div className="flex items-center justify-between mb-4 border-b border-gray-700 pb-2">
        <h3 className="font-bold tracking-wider text-cyan-400 flex items-center gap-2">
            <Activity size={16} />
            TRAVERSE_METRICS
        </h3>
        <div className={`flex items-center gap-1 font-bold ${riskColor}`}>
            {riskLevel === 'HIGH' && <AlertTriangle size={14} />}
            {riskLevel === 'LOW' && <CheckCircle size={14} />}
            <span>RISK: {riskLevel}</span>
        </div>
      </div>
      
      <div className="space-y-4">
        {/* Distance Display */}
        <div className="bg-gray-900/50 p-2 rounded border border-gray-700/50 flex justify-between items-center">
            <span className="text-gray-400 text-xs">DISTANCE</span>
            <span className="text-lg font-bold">{(distance_m / 1000).toFixed(2)} <span className="text-xs font-normal text-gray-500">km</span></span>
        </div>

        {/* Cost Component Breakdown */}
        <div className="space-y-2">
          <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Impedance Factors</div>
          
          {/* Slope */}
          <div className="group">
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400 group-hover:text-white transition-colors">Slope Grade</span>
              <span className="text-cyan-300">{slope_cost.toFixed(1)}</span>
            </div>
            <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden border border-gray-700">
              <div
                className="h-full bg-gradient-to-r from-cyan-900 to-cyan-400"
                style={{ width: `${Math.min((slope_cost / maxComponent) * 100, 100)}%` }}
              />
            </div>
          </div>

          {/* Roughness */}
          <div className="group">
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400 group-hover:text-white transition-colors">Terrain Roughness</span>
              <span className="text-orange-300">{roughness_cost.toFixed(1)}</span>
            </div>
            <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden border border-gray-700">
              <div
                className="h-full bg-gradient-to-r from-orange-900 to-orange-400"
                style={{ width: `${Math.min((roughness_cost / maxComponent) * 100, 100)}%` }}
              />
            </div>
          </div>

          {/* Shadow/Solar */}
          <div className="group">
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400 group-hover:text-white transition-colors">Occlusion/Shadow</span>
              <span className="text-purple-300">{shadow_cost.toFixed(1)}</span>
            </div>
            <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden border border-gray-700">
              <div
                className="h-full bg-gradient-to-r from-purple-900 to-purple-400"
                style={{ width: `${Math.min((shadow_cost / maxComponent) * 100, 100)}%` }}
              />
            </div>
          </div>
        </div>

        {/* Energy & Total */}
        <div className="pt-3 border-t border-gray-700 grid grid-cols-2 gap-4">
          <div>
            <span className="text-[10px] text-gray-500 block">EST. ENERGY</span>
            <span className="text-base font-bold text-blue-400">{(energy_estimate_j / 1000).toFixed(1)} <span className="text-xs">kJ</span></span>
          </div>
          <div className="text-right">
            <span className="text-[10px] text-gray-500 block">AGGREGATE COST</span>
            <span className="text-base font-bold text-white">{totalCost.toFixed(1)}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
