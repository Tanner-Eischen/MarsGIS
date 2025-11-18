import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

interface Project {
  id: string
  name: string
  description: string
  created_at: string
  roi: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }
  dataset: string
  preset_id?: string
  selected_sites: number[]
}

export default function Projects() {
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    loadProjects()
  }, [])

  const loadProjects = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/v1/projects')
      if (response.ok) {
        const data = await response.json()
        setProjects(data)
      }
    } catch (error) {
      console.error('Failed to load projects:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleOpenProject = async (projectId: string) => {
    try {
      const response = await fetch(`http://localhost:5000/api/v1/projects/${projectId}`)
      if (response.ok) {
        const project = await response.json()
        // Navigate to Decision Lab with project data
        navigate('/decision-lab', { state: { project } })
      }
    } catch (error) {
      console.error('Failed to open project:', error)
      alert('Failed to open project')
    }
  }

  const handleDeleteProject = async (projectId: string) => {
    if (!confirm('Are you sure you want to delete this project?')) {
      return
    }

    try {
      const response = await fetch(`http://localhost:5000/api/v1/projects/${projectId}`, {
        method: 'DELETE'
      })
      if (response.ok) {
        loadProjects()
      } else {
        alert('Failed to delete project')
      }
    } catch (error) {
      console.error('Failed to delete project:', error)
      alert('Failed to delete project')
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold">Projects</h2>
        <button
          onClick={() => setShowCreateModal(true)}
          className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-semibold"
        >
          New Project
        </button>
      </div>

      {loading ? (
        <div className="text-center py-8">Loading projects...</div>
      ) : projects.length === 0 ? (
        <div className="text-center py-8 text-gray-400">
          No projects yet. Create one to save your analyses.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {projects.map(project => (
            <div
              key={project.id}
              className="bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-blue-500 transition-colors"
            >
              <h3 className="text-xl font-semibold mb-2">{project.name}</h3>
              <p className="text-sm text-gray-400 mb-4 line-clamp-2">
                {project.description || 'No description'}
              </p>
              <div className="text-xs text-gray-500 mb-4">
                Created: {new Date(project.created_at).toLocaleDateString()}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => handleOpenProject(project.id)}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 px-3 py-2 rounded text-sm font-semibold"
                >
                  Open
                </button>
                <button
                  onClick={() => handleDeleteProject(project.id)}
                  className="bg-red-600 hover:bg-red-700 px-3 py-2 rounded text-sm font-semibold"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {showCreateModal && (
        <CreateProjectModal
          onClose={() => setShowCreateModal(false)}
          onSave={() => {
            setShowCreateModal(false)
            loadProjects()
          }}
        />
      )}
    </div>
  )
}

function CreateProjectModal({ onClose, onSave }: { onClose: () => void; onSave: () => void }) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')

  const handleSave = async () => {
    if (!name.trim()) {
      alert('Project name is required')
      return
    }

    // Get current analysis state (simplified - would get from context/state)
    const projectData = {
      name,
      description,
      roi: { lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 },
      dataset: 'mola',
      preset_id: 'balanced',
      selected_sites: [],
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
      } else {
        alert('Failed to create project')
      }
    } catch (error) {
      console.error('Failed to create project:', error)
      alert('Failed to create project')
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-full max-w-md">
        <h3 className="text-xl font-semibold mb-4">Create New Project</h3>
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
            Save
          </button>
        </div>
      </div>
    </div>
  )
}

