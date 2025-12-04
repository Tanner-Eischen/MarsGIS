import { useState } from 'react'
import NavigationPlanning from './NavigationPlanning'
import { Map, Flag, Activity } from 'lucide-react'

// Placeholder for Mission Scenarios until fully migrated/re-implemented inline
function MissionScenariosComponent() {
    return (
        <div className="glass-panel p-8 rounded-lg text-center text-gray-400">
            <Flag className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-bold text-white mb-2">MISSION_SCENARIOS</h3>
            <p className="text-sm">Scenario builder is currently under maintenance.</p>
        </div>
    )
}

export default function MissionControl() {
  const [activeTab, setActiveTab] = useState<'planning' | 'scenarios'>('planning')

  return (
    <div className="h-full flex flex-col bg-gray-900 text-white min-h-screen">
      {/* Header / Tabs */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
            <Activity className="text-green-500" />
            <h1 className="text-2xl font-bold tracking-wider">MISSION_CONTROL</h1>
        </div>
        
        <div className="flex bg-gray-900 rounded-lg p-1 border border-gray-700">
          <button
            onClick={() => setActiveTab('planning')}
            className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === 'planning'
                ? 'bg-green-600 text-white shadow-lg shadow-green-900/20'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >
            <Map className="w-4 h-4 mr-2" />
            Route Planning
          </button>
          <button
            onClick={() => setActiveTab('scenarios')}
            className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === 'scenarios'
                ? 'bg-purple-600 text-white shadow-lg shadow-purple-900/20'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >
            <Flag className="w-4 h-4 mr-2" />
            Mission Scenarios
          </button>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 p-6 overflow-auto">
        {activeTab === 'planning' ? (
            <div className="max-w-7xl mx-auto">
                <NavigationPlanning />
            </div>
        ) : (
            <div className="max-w-7xl mx-auto">
                <MissionScenariosComponent />
            </div>
        )}
      </div>
    </div>
  )
}
