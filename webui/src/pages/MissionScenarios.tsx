import { useState } from 'react'
import MissionLandingWizard from '../components/MissionLandingWizard'
import RoverTraverseWizard from '../components/RoverTraverseWizard'

export default function MissionScenarios() {
  const [activeTab, setActiveTab] = useState<'landing' | 'traverse'>('landing')

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Mission Scenarios</h2>
      
      {/* Tab Navigation */}
      <div className="bg-gray-800 rounded-lg p-1 flex gap-2">
        <button
          onClick={() => setActiveTab('landing')}
          className={`flex-1 px-4 py-2 rounded-md font-semibold transition-colors ${
            activeTab === 'landing'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Landing Site Wizard
        </button>
        <button
          onClick={() => setActiveTab('traverse')}
          className={`flex-1 px-4 py-2 rounded-md font-semibold transition-colors ${
            activeTab === 'traverse'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Rover Traverse Wizard
        </button>
      </div>

      {/* Wizard Content */}
      <div className="bg-gray-800 rounded-lg p-6">
        {activeTab === 'landing' ? (
          <MissionLandingWizard />
        ) : (
          <RoverTraverseWizard />
        )}
      </div>
    </div>
  )
}

