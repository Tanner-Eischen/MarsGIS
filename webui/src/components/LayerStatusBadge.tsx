import React from 'react'

interface LayerStatusBadgeProps {
  status: 'loading' | 'cached' | 'error' | 'idle'
  dataset?: 'mola' | 'hirise' | 'ctx'
}

export default function LayerStatusBadge({ status, dataset }: LayerStatusBadgeProps) {
  const getStatusClasses = () => {
    switch (status) {
      case 'loading':
        return 'bg-yellow-600/20 text-yellow-400 border-yellow-600/30'
      case 'cached':
        return 'bg-green-600/20 text-green-400 border-green-600/30'
      case 'error':
        return 'bg-red-600/20 text-red-400 border-red-600/30'
      case 'idle':
      default:
        return 'bg-gray-600/20 text-gray-400 border-gray-600/30'
    }
  }

  const getStatusText = () => {
    switch (status) {
      case 'loading':
        return 'Loading'
      case 'cached':
        return 'Cached'
      case 'error':
        return 'Error'
      case 'idle':
      default:
        return ''
    }
  }

  const getDatasetIcon = () => {
    switch (dataset) {
      case 'mola':
        return '🛰️'
      case 'hirise':
        return '🔭'
      case 'ctx':
        return '📷'
      default:
        return ''
    }
  }

  if (status === 'idle') {
    return null
  }

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full border ${getStatusClasses()}`}
    >
      {getDatasetIcon() && <span>{getDatasetIcon()}</span>}
      <span>{getStatusText()}</span>
    </span>
  )
}

