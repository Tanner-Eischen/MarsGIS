import { useEffect, useState } from 'react'
import TerrainMap from '../components/TerrainMap'
import SolarImpactPanel from '../components/SolarImpactPanel'
import { analyzeSolarPotential, SolarAnalysisResponse, getExampleROIs, ExampleROIItem } from '../services/api'

interface ROI {
  lat_min: number
  lat_max: number
  lon_min: number
  lon_max: number
}

interface Props {
  roi?: ROI
  onRoiChange?: (roi: ROI) => void
  dataset?: string
  onDatasetChange?: (dataset: string) => void
}

export default function SolarAnalysis({ 
  roi: propRoi, 
  onRoiChange, 
  dataset: propDataset, 
  onDatasetChange 
}: Props) {
  
  const [localRoi, setLocalRoi] = useState<ROI>({
    lat_min: 18.0,
    lat_max: 18.6,
    lon_min: 77.0,
    lon_max: 77.8,
  })
  const [localDataset, setLocalDataset] = useState('mola')

  const roi = propRoi || localRoi
  const setRoi = onRoiChange || setLocalRoi
  const dataset = propDataset || localDataset
  const setDataset = onDatasetChange || setLocalDataset

  const [examples, setExamples] = useState<ExampleROIItem[]>([])
  useEffect(() => {
    const s = localStorage.getItem('solar.roi')
    const d = localStorage.getItem('solar.dataset')
    if (s && !propRoi) { try { const o = JSON.parse(s); if (o && o.lat_min !== undefined) setRoi(o) } catch {} }
    if (d && !propDataset) setDataset(d)
  }, [])
  
  useEffect(() => { localStorage.setItem('solar.roi', JSON.stringify(roi)) }, [roi])
  useEffect(() => { localStorage.setItem('solar.dataset', dataset) }, [dataset])
  
  // Panel configuration
  const [panelEfficiency, setPanelEfficiency] = useState(0.25)
  const [panelArea, setPanelArea] = useState(100.0)
  
  // Mission parameters
  const [batteryCapacity, setBatteryCapacity] = useState(50.0)
  const [dailyPowerNeeds, setDailyPowerNeeds] = useState(20.0)
  const [batteryCostPerKwh] = useState(1000.0)
  const [missionDuration, setMissionDuration] = useState(500.0)
  
  // Sun position (static for now)
  const [sunAzimuth, setSunAzimuth] = useState(0.0)
  const [sunAltitude, setSunAltitude] = useState(45.0)
  
  // Results
  const [results, setResults] = useState<SolarAnalysisResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await analyzeSolarPotential({
        roi,
        dataset,
        sun_azimuth: sunAzimuth,
        sun_altitude: sunAltitude,
        panel_efficiency: panelEfficiency,
        panel_area_m2: panelArea,
        battery_capacity_kwh: batteryCapacity,
        daily_power_needs_kwh: dailyPowerNeeds,
        battery_cost_per_kwh: batteryCostPerKwh,
        mission_duration_days: missionDuration,
      })
      
      setResults(response)
    } catch (err) {
      console.error('Solar analysis failed:', err)
      setError(err instanceof Error ? err.message : 'Failed to analyze solar potential')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 text-sm">
      
      <div className="glass-panel p-4 rounded-lg border border-amber-500/20">
        <div className="flex items-center justify-between mb-4">
           <h3 className="text-xs font-bold text-amber-400 uppercase tracking-wide">Analysis Configuration</h3>
           <button
            onClick={handleAnalyze}
            disabled={loading}
            className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 text-white rounded text-xs font-bold uppercase tracking-wider disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_10px_rgba(245,158,11,0.2)]"
          >
            {loading ? 'CALCULATING...' : 'RUN ANALYSIS'}
          </button>
        </div>

        {error && (
            <div className="p-2 mb-3 bg-red-900/50 border border-red-700 rounded text-xs text-red-300">
            {error}
            </div>
        )}

        <div className="space-y-4">
            {/* ROI Section */}
            <div>
                <div className="flex items-center justify-between mb-2">
                    <label className="text-[10px] font-bold text-gray-500 uppercase">Region of Interest</label>
                    <button
                        onClick={async () => { const data = await getExampleROIs(); setExamples(data) }}
                        className="text-[10px] text-amber-400 hover:text-amber-300 underline"
                    >
                        Load Example
                    </button>
                </div>
                {examples.length > 0 && (
                    <select
                        onChange={(e) => {
                        const sel = examples.find(x => x.id === e.target.value)
                        if (sel) { setRoi(sel.bbox); setDataset(sel.dataset) }
                        }}
                        className="w-full bg-gray-800 text-white px-2 py-1 rounded text-[10px] border border-gray-600 mb-2"
                        defaultValue=""
                    >
                        <option value="" disabled>Select...</option>
                        {examples.map(x => (
                        <option key={x.id} value={x.id}>{x.name}</option>
                        ))}
                    </select>
                )}
                <div className="grid grid-cols-2 gap-2">
                    <input type="number" value={roi.lat_min} onChange={(e) => setRoi({ ...roi, lat_min: parseFloat(e.target.value) })} className="bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs" step="0.1" placeholder="Lat Min"/>
                    <input type="number" value={roi.lat_max} onChange={(e) => setRoi({ ...roi, lat_max: parseFloat(e.target.value) })} className="bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs" step="0.1" placeholder="Lat Max"/>
                    <input type="number" value={roi.lon_min} onChange={(e) => setRoi({ ...roi, lon_min: parseFloat(e.target.value) })} className="bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs" step="0.1" placeholder="Lon Min"/>
                    <input type="number" value={roi.lon_max} onChange={(e) => setRoi({ ...roi, lon_max: parseFloat(e.target.value) })} className="bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs" step="0.1" placeholder="Lon Max"/>
                </div>
            </div>
            <div className="pt-2 border-t border-gray-700/50">
              <label className="block text-[10px] font-bold text-gray-500 uppercase mb-2">Source Data</label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                className="w-full bg-gray-800 border border-gray-600 text-white px-2 py-1.5 rounded text-xs focus:border-amber-500 focus:outline-none"
              >
                <option value="mola">MOLA (Global)</option>
                <option value="mola_200m">MOLA 200m (Global)</option>
                <option value="hirise">HiRISE (High Res)</option>
                <option value="ctx">CTX (Medium Res)</option>
              </select>
            </div>

            {/* Panel Config */}
            <div className="pt-2 border-t border-gray-700/50">
                <label className="block text-[10px] font-bold text-gray-500 uppercase mb-2">Array Specs</label>
                <div className="space-y-2">
                    <div>
                        <div className="flex justify-between text-[10px] text-gray-400">
                            <span>Efficiency</span>
                            <span>{(panelEfficiency * 100).toFixed(0)}%</span>
                        </div>
                        <input type="range" min="0.1" max="0.5" step="0.01" value={panelEfficiency} onChange={(e) => setPanelEfficiency(parseFloat(e.target.value))} className="w-full accent-amber-500 h-1.5"/>
                    </div>
                    <div>
                        <div className="flex justify-between text-[10px] text-gray-400">
                            <span>Area</span>
                            <span>{panelArea.toFixed(0)} m²</span>
                        </div>
                        <input type="range" min="10" max="500" step="10" value={panelArea} onChange={(e) => setPanelArea(parseFloat(e.target.value))} className="w-full accent-amber-500 h-1.5"/>
                    </div>
                </div>
            </div>

             {/* Mission Config */}
             <div className="pt-2 border-t border-gray-700/50">
                <label className="block text-[10px] font-bold text-gray-500 uppercase mb-2">Mission Specs</label>
                <div className="space-y-2">
                    <div>
                        <div className="flex justify-between text-[10px] text-gray-400">
                            <span>Battery</span>
                            <span>{batteryCapacity.toFixed(0)} kWh</span>
                        </div>
                        <input type="range" min="10" max="200" step="10" value={batteryCapacity} onChange={(e) => setBatteryCapacity(parseFloat(e.target.value))} className="w-full accent-amber-500 h-1.5"/>
                    </div>
                    <div>
                        <div className="flex justify-between text-[10px] text-gray-400">
                            <span>Daily Load</span>
                            <span>{dailyPowerNeeds.toFixed(1)} kWh</span>
                        </div>
                        <input type="range" min="5" max="50" step="1" value={dailyPowerNeeds} onChange={(e) => setDailyPowerNeeds(parseFloat(e.target.value))} className="w-full accent-amber-500 h-1.5"/>
                    </div>
                </div>
            </div>
        </div>
      </div>

      {/* Results Panel */}
        {results && (
        <div className="glass-panel p-4 rounded-lg border border-amber-500/20">
            <h3 className="text-xs font-bold mb-3 text-amber-400 tracking-wider uppercase">Solar Metrics</h3>
            
            <div className="grid grid-cols-2 gap-2 mb-4">
                <div className="bg-gray-900 p-2 rounded border border-gray-700">
                    <div className="text-[10px] text-gray-500 uppercase">Mean Potential</div>
                    <div className="text-lg font-bold text-white">{(results.statistics.mean * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-gray-900 p-2 rounded border border-gray-700">
                    <div className="text-[10px] text-gray-500 uppercase">Irradiance</div>
                    <div className="text-lg font-bold text-amber-400">{results.statistics.mean_irradiance_w_per_m2.toFixed(0)} <span className="text-xs text-gray-500 font-normal">W/m²</span></div>
                </div>
            </div>

            <div className="text-xs text-gray-300 space-y-1">
                <div className="flex justify-between">
                    <span>Power Gen (Daily):</span>
                    <span className="font-mono text-white">{results.mission_impacts.power_generation_kwh_per_day.toFixed(1)} kWh</span>
                </div>
                <div className="flex justify-between">
                    <span>Surplus:</span>
                    <span className={`font-mono ${results.mission_impacts.power_surplus_kwh_per_day >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {results.mission_impacts.power_surplus_kwh_per_day.toFixed(1)} kWh
                    </span>
                </div>
            </div>
        </div>
        )}
    </div>
  )
}
