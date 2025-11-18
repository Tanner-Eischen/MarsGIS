
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

  return (
    <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
      <h3 className="font-semibold mb-4">Route Cost Summary</h3>
      
      <div className="space-y-4">
        {/* Distance */}
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-400">Total Distance</span>
            <span className="font-semibold">{(distance_m / 1000).toFixed(2)} km</span>
          </div>
        </div>

        {/* Cost Components */}
        <div className="space-y-2">
          <div className="text-xs text-gray-400 mb-2">Cost Components:</div>
          
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">Slope Cost</span>
              <span className="text-gray-300">{slope_cost.toFixed(2)}</span>
            </div>
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-yellow-500"
                style={{ width: `${(slope_cost / maxComponent) * 100}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">Roughness Cost</span>
              <span className="text-gray-300">{roughness_cost.toFixed(2)}</span>
            </div>
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-orange-500"
                style={{ width: `${(roughness_cost / maxComponent) * 100}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400">Shadow Cost</span>
              <span className="text-gray-300">{shadow_cost.toFixed(2)}</span>
            </div>
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500"
                style={{ width: `${(shadow_cost / maxComponent) * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* Energy Estimate */}
        <div className="pt-2 border-t border-gray-700">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Estimated Energy</span>
            <span className="font-semibold text-blue-400">
              {(energy_estimate_j / 1000).toFixed(1)} kJ
            </span>
          </div>
        </div>

        {/* Total Cost */}
        <div className="pt-2 border-t border-gray-700">
          <div className="flex justify-between">
            <span className="text-sm font-semibold">Total Cost</span>
            <span className="text-lg font-bold text-green-400">{totalCost.toFixed(2)}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

