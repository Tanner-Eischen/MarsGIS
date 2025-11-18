import PresetsSelector from './PresetsSelector'

interface Preset {
  id: string
  name: string
  description: string
  scope: string
  weights: Record<string, number>
}

interface RoutePresetsSelectorProps {
  presets: Preset[]
  selected: string
  onSelect: (presetId: string) => void
}

export default function RoutePresetsSelector({ presets, selected, onSelect }: RoutePresetsSelectorProps) {
  // Filter to route presets only
  const routePresets = presets.filter(p => p.scope === 'route')
  
  return <PresetsSelector presets={routePresets} selected={selected} onSelect={onSelect} />
}

