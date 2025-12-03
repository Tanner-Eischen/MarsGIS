interface SolarImpactPanelProps {
  impacts: {
    power_generation_kwh_per_day: number
    power_surplus_kwh_per_day: number
    mission_duration_extension_days: number
    cost_savings_usd: number
    battery_reduction_kwh: number
  } | null
  dailyPowerNeeds: number
}

export default function SolarImpactPanel({ impacts, dailyPowerNeeds }: SolarImpactPanelProps) {
  if (!impacts) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Mission Impact</h3>
        <p className="text-gray-400">Run analysis to see mission impact calculations.</p>
      </div>
    )
  }

  const powerGeneration = impacts.power_generation_kwh_per_day
  const powerSurplus = impacts.power_surplus_kwh_per_day
  const durationExtension = impacts.mission_duration_extension_days
  const costSavings = impacts.cost_savings_usd
  const batteryReduction = impacts.battery_reduction_kwh

  const powerSurplusPercent = dailyPowerNeeds > 0 
    ? (powerSurplus / dailyPowerNeeds) * 100 
    : 0

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h3 className="text-xl font-semibold mb-4">Mission Impact</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Power Generation Card */}
        <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-medium text-gray-300">Power Generation</h4>
            <span className="text-green-400">↑</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {powerGeneration.toFixed(1)} <span className="text-sm text-gray-400">kWh/day</span>
          </div>
          <div className="mt-2 text-xs text-gray-400">
            vs {dailyPowerNeeds.toFixed(1)} kWh/day needed
          </div>
          {powerSurplus > 0 && (
            <div className="mt-2 text-xs text-green-400">
              +{powerSurplus.toFixed(1)} kWh/day surplus ({powerSurplusPercent.toFixed(1)}%)
            </div>
          )}
          {powerSurplus < 0 && (
            <div className="mt-2 text-xs text-red-400">
              {powerSurplus.toFixed(1)} kWh/day deficit
            </div>
          )}
        </div>

        {/* Mission Duration Card */}
        <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-medium text-gray-300">Mission Extension</h4>
            {durationExtension > 0 ? (
              <span className="text-green-400">↑</span>
            ) : (
              <span className="text-gray-400">—</span>
            )}
          </div>
          <div className="text-2xl font-bold text-white">
            {durationExtension > 0 ? '+' : ''}{durationExtension.toFixed(0)} <span className="text-sm text-gray-400">days</span>
          </div>
          <div className="mt-2 text-xs text-gray-400">
            Extended mission duration
          </div>
          {durationExtension > 0 && (
            <div className="mt-2 text-xs text-green-400">
              Additional operational time
            </div>
          )}
        </div>

        {/* Cost Savings Card */}
        <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-medium text-gray-300">Cost Savings</h4>
            {costSavings > 0 ? (
              <span className="text-green-400">↑</span>
            ) : (
              <span className="text-gray-400">—</span>
            )}
          </div>
          <div className="text-2xl font-bold text-white">
            ${(costSavings / 1000).toFixed(1)}k
          </div>
          <div className="mt-2 text-xs text-gray-400">
            Total savings
          </div>
          {batteryReduction > 0 && (
            <div className="mt-2 text-xs text-green-400">
              {batteryReduction.toFixed(1)} kWh battery reduction
            </div>
          )}
        </div>
      </div>

      {/* Summary */}
      <div className="mt-4 p-4 bg-gray-700 rounded-lg border border-gray-600">
        <h4 className="text-sm font-medium text-gray-300 mb-2">Summary</h4>
        <ul className="text-xs text-gray-400 space-y-1">
          <li>• Solar panels generate {powerGeneration.toFixed(1)} kWh/day</li>
          {powerSurplus > 0 && (
            <li>• Power surplus enables {durationExtension.toFixed(0)} additional mission days</li>
          )}
          {batteryReduction > 0 && (
            <li>• Reduced battery capacity by {batteryReduction.toFixed(1)} kWh</li>
          )}
          {costSavings > 0 && (
            <li>• Total cost savings: ${costSavings.toLocaleString()}</li>
          )}
        </ul>
      </div>
    </div>
  )
}

