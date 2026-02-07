import { useState, useEffect } from 'react'
import { apiFetch } from '../lib/apiBase'

interface ExampleROI {
  id: string
  name: string
  description: string
  bbox: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }
  dataset: string
}

interface ExamplesDrawerProps {
  onSelectExample: (example: ExampleROI) => void
  isOpen: boolean
  onClose: () => void
}

export default function ExamplesDrawer({ onSelectExample, isOpen, onClose }: ExamplesDrawerProps) {
  const [examples, setExamples] = useState<ExampleROI[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (isOpen) {
      apiFetch('/examples/rois')
        .then(res => res.json())
        .then(data => {
          setExamples(data)
          setLoading(false)
        })
        .catch(err => {
          console.error('Failed to load examples:', err)
          setLoading(false)
        })
    }
  }, [isOpen])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-semibold">Example Regions</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl"
          >
            ×
          </button>
        </div>

        {loading ? (
          <div className="text-center py-8">Loading examples...</div>
        ) : examples.length === 0 ? (
          <div className="text-center py-8 text-gray-400">No examples available</div>
        ) : (
          <div className="space-y-3">
            {examples.map(example => (
              <div
                key={example.id}
                className="bg-gray-700 rounded-lg p-4 hover:bg-gray-600 cursor-pointer transition-colors"
                onClick={() => {
                  onSelectExample(example)
                  onClose()
                }}
              >
                <h4 className="font-semibold text-lg mb-1">{example.name}</h4>
                <p className="text-sm text-gray-300">{example.description}</p>
                <div className="text-xs text-gray-400 mt-2">
                  Dataset: {example.dataset.toUpperCase()}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

