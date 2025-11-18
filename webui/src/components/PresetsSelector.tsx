
interface Preset {
  id: string
  name: string
  description: string
  scope: string
  weights: Record<string, number>
}

interface PresetsSelectorProps {
  presets: Preset[]
  selected: string
  onSelect: (presetId: string) => void
}

export default function PresetsSelector({ presets, selected, onSelect }: PresetsSelectorProps) {
  return (
    <div className="p-4 border-b border-gray-700">
      <h3 className="font-semibold mb-3">Preset Selection</h3>
      <div className="space-y-2">
        {presets.map((preset) => (
          <button
            key={preset.id}
            onClick={() => onSelect(preset.id)}
            className={`w-full text-left p-3 rounded border-2 transition-colors ${
              selected === preset.id
                ? 'border-blue-500 bg-blue-900/30'
                : 'border-gray-700 bg-gray-700/50 hover:bg-gray-700'
            }`}
          >
            <div className="font-semibold text-sm">{preset.name}</div>
            <div className="text-xs text-gray-400 mt-1">{preset.description}</div>
          </button>
        ))}
      </div>
    </div>
  )
}

