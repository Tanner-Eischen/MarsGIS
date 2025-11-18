import { useState } from 'react'

interface SaveProjectModalProps {
  roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }
  dataset: string
  presetId: string
  selectedSites: number[]
  onClose: () => void
  onSave: () => void
}

export default function SaveProjectModal({
  roi,
  dataset,
  presetId,
  selectedSites,
  onClose,
  onSave
}: SaveProjectModalProps) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')

  const handleSave = async () => {
    if (!name.trim()) {
      alert('Project name is required')
      return
    }

    const projectData = {
      name,
      description,
      roi,
      dataset,
      preset_id: presetId,
      selected_sites: selectedSites,
      routes: [],
      metadata: {}
    }

    try {
      const response = await fetch('http://localhost:5000/api/v1/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(projectData)
      })

      if (response.ok) {
        onSave()
        onClose()
      } else {
        const error = await response.json()
        alert(`Failed to save project: ${error.detail || 'Unknown error'}`)
      }
    } catch (error) {
      console.error('Failed to save project:', error)
      alert('Failed to save project')
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-full max-w-md">
        <h3 className="text-xl font-semibold mb-4">Save as Project</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Project Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
              placeholder="My Mars Mission"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full bg-gray-700 text-white px-4 py-2 rounded-md"
              rows={3}
              placeholder="Project description..."
            />
          </div>
          <div className="text-xs text-gray-400">
            This will save the current ROI, dataset, preset, and selected sites.
          </div>
        </div>
        <div className="flex gap-2 mt-6">
          <button
            onClick={onClose}
            className="flex-1 bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded font-semibold"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="flex-1 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-semibold"
          >
            Save Project
          </button>
        </div>
      </div>
    </div>
  )
}

