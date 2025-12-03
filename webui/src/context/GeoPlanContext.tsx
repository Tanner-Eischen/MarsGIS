import { createContext, useContext, useMemo, useState } from 'react'
import type { SiteCandidate } from '../services/api'

type SiteType = 'landing' | 'construction'

interface GeoPlanState {
  landingSites: SiteCandidate[]
  constructionSites: SiteCandidate[]
  recommendedLandingSiteId: number | null
  publishSites: (type: SiteType, sites: SiteCandidate[]) => void
  setRecommendedLandingSiteId: (id: number | null) => void
}

const GeoPlanContext = createContext<GeoPlanState | null>(null)

export function GeoPlanProvider({ children }: { children: React.ReactNode }) {
  const [landingSites, setLandingSites] = useState<SiteCandidate[]>([])
  const [constructionSites, setConstructionSites] = useState<SiteCandidate[]>([])
  const [recommendedLandingSiteId, setRecommendedLandingSiteId] = useState<number | null>(null)

  const publishSites = (type: SiteType, sites: SiteCandidate[]) => {
    if (type === 'landing') setLandingSites(sites)
    else setConstructionSites(sites)
  }

  const value = useMemo(() => ({
    landingSites,
    constructionSites,
    recommendedLandingSiteId,
    publishSites,
    setRecommendedLandingSiteId,
  }), [landingSites, constructionSites, recommendedLandingSiteId])

  return <GeoPlanContext.Provider value={value}>{children}</GeoPlanContext.Provider>
}

export function useGeoPlan() {
  const ctx = useContext(GeoPlanContext)
  if (!ctx) throw new Error('GeoPlanContext not initialized')
  return ctx
}

