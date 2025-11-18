import { useState, useEffect } from 'react'

interface AdvancedWeightsPanelProps {
  weights: Record<string, number>
  onChange: (weights: Record<string, number>) => void
  presetWeights: Record<string, number>
}

export default function AdvancedWeightsPanel({ weights, onChange, presetWeights }: AdvancedWeightsPanelProps) {
  const [localWeights, setLocalWeights] = useState<Record<string, number>>(weights)

  useEffect(() => {
    setLocalWeights(weights)
  }, [weights])

  const updateWeight = (criterion: string, value: number) => {
    const newWeights = { ...localWeights, [criterion]: value }
    setLocalWeights(newWeights)
    onChange(newWeights)
  }

  const totalWeight = Object.values(localWeights).reduce((sum, w) => sum + w, 0)
  const isValid = Math.abs(totalWeight - 1.0) < 0.01

  const criteria = [
    { key: 'slope', label: 'Slope Safety' },
    { key: 'roughness', label: 'Surface Roughness' },
    { key: 'elevation', label: 'Elevation' },
    { key: 'solar_exposure', label: 'Solar Exposure' },
    { key: 'science_value', label: 'Science Value' },
  ]

  return (
    <div className="mt-3 space-y-3">
      {criteria.map(({ key, label }) => (
        <div key={key}>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-300">{label}</span>
            <span className="text-gray-400">
              {((localWeights[key] || presetWeights[key] || 0) * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={localWeights[key] ?? presetWeights[key] ?? 0}
            onChange={(e) => updateWeight(key, parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      ))}
      <div className="pt-2 border-t border-gray-700">
        <div className="flex justify-between text-xs">
          <span className={isValid ? 'text-green-400' : 'text-yellow-400'}>
            Total Weight: {(totalWeight * 100).toFixed(1)}%
          </span>
          {!isValid && (
            <span className="text-yellow-400">Should sum to 100%</span>
          )}
        </div>
      </div>
    </div>
  )
}

