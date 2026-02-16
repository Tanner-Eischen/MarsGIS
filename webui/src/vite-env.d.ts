/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string
  readonly VITE_GLOBAL_BASEMAP_STYLE?: string
  readonly VITE_USE_LEGACY_IMAGE_OVERLAY?: string
  readonly VITE_WS_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

