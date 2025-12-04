# Mars Geospatial Datasets - Implementation Status

## ✅ Fully Implemented (Backend API)

### 1. **MOLA (Mars Orbiter Laser Altimeter)**
- **Type**: Raster DEM
- **Resolution**: 463m per pixel
- **Source**: `https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`
- **Status**: ✅ Working via DataManager
- **Overlays**: elevation, solar, hillshade, slope, aspect, roughness, TRI

### 2. **MOLA 200m**
- **Type**: Raster DEM  
- **Resolution**: 200m per pixel
- **Source**: `https://planetarymaps.usgs.gov/mosaic/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif`
- **Status**: ✅ Working via DataManager
- **Overlays**: elevation, solar, hillshade, slope, aspect, roughness, TRI

### 3. **HiRISE (High Resolution Imaging Science Experiment)**
- **Type**: Raster DEM
- **Resolution**: 1m per pixel
- **Source**: `https://s3.amazonaws.com/mars-hirise-pds/HiRISE_DEMs/`
- **Status**: ✅ Working via DataManager
- **Overlays**: elevation, solar, hillshade, slope, aspect, roughness, TRI

### 4. **CTX (Context Camera)**
- **Type**: Raster DEM
- **Resolution**: 18m per pixel
- **Source**: `https://ode.rsl.wustl.edu/mars/data/ctx/`
- **Status**: ✅ Working via DataManager
- **Overlays**: elevation, solar, hillshade, slope, aspect, roughness, TRI

## ⚠️ Configured but NOT Fully Implemented

### 5. **TES Dust Cover Index (DCI)**
- **Type**: Raster
- **Resolution**: ~3.5 km per pixel
- **Source**: `https://tes.asu.edu/~ruff/DCI/`
- **Status**: ⚠️ **CONFIGURED BUT NOT IMPLEMENTED**
- **Location**: 
  - Config: `marshab_config.yaml` (line 26-31)
  - Frontend: `webui/src/config/marsDataSources.ts` (line 80-97)
  - Backend: `marshab/web/routes/visualization.py` - has `_generate_dust_overlay()` but uses synthetic data
- **Issue**: DataManager doesn't have `get_dci_data()` method
- **Current**: Uses synthetic dust pattern
- **Needs**: Implementation of DCI data loading in `marshab/core/data_manager.py`

### 6. **Mars Nomenclature (OpenPlanetaryMap)**
- **Type**: Vector (GeoJSON)
- **Source**: `https://openplanetary.carto.com/u/opmbuilder/dataset/opm_499_mars_nomenclature_polygons`
- **Status**: ⚠️ **CONFIGURED BUT NOT IMPLEMENTED**
- **Location**: `webui/src/config/marsDataSources.ts` (line 157-168)
- **Issue**: No fetch logic for external CARTO API
- **Needs**: CARTO API client or proxy endpoint

### 7. **Mars Topography Contours (OpenPlanetaryMap)**
- **Type**: Vector (GeoJSON lines)
- **Resolution**: 200m contour intervals
- **Source**: `https://openplanetary.carto.com/u/opmbuilder/dataset/opm_499_mars_contours_200m_lines`
- **Status**: ⚠️ **CONFIGURED BUT NOT IMPLEMENTED**
- **Location**: `webui/src/config/marsDataSources.ts` (line 169-180)
- **Issue**: No fetch logic for external CARTO API
- **Needs**: CARTO API client or proxy endpoint

### 8. **TES Albedo (OpenPlanetaryMap)**
- **Type**: Vector (GeoJSON polygons)
- **Source**: `https://openplanetary.carto.com/u/opmbuilder/dataset/opm_499_mars_albedo_tes_7classes`
- **Status**: ⚠️ **CONFIGURED BUT NOT IMPLEMENTED**
- **Location**: `webui/src/config/marsDataSources.ts` (line 181-192)
- **Issue**: No fetch logic for external CARTO API
- **Needs**: CARTO API client or proxy endpoint

## 📍 Where Datasets Are Defined

### Frontend Configuration
- **File**: `webui/src/config/marsDataSources.ts`
- **Defines**: All overlay types, metadata, external URLs
- **Lines**: 50-193

### Backend Configuration  
- **File**: `marshab_config.yaml`
- **Defines**: Data source URLs and metadata
- **Lines**: 7-31

### Backend Implementation
- **File**: `marshab/core/data_manager.py`
- **Implements**: MOLA, HiRISE, CTX loading
- **Missing**: TES DCI loading method

### Backend Routes
- **File**: `marshab/web/routes/visualization.py`
- **Implements**: Overlay generation endpoints
- **Lines**: 
  - Dust overlay: 1226-1323 (uses synthetic data)
  - Solar overlay: 1116-1223 (DCI integration stubbed)

## 🔧 What Needs to Be Implemented

1. **TES DCI Data Loading**
   - Add `get_dci_data()` method to `DataManager`
   - Download/load DCI GeoTIFF from ASU
   - Integrate with solar calculations

2. **OpenPlanetaryMap CARTO API Client**
   - Create proxy endpoint in backend: `/api/v1/visualization/external/{dataset}`
   - Fetch GeoJSON from CARTO datasets
   - Handle WMS/WFS requests if needed

3. **External Dataset Fetching**
   - Update `useOverlayLayerManager.ts` to handle external URLs
   - Add GeoJSON parsing for vector overlays
   - Display vector overlays on map

## 📊 Current Status Summary

- **Working**: 4 datasets (MOLA, MOLA 200m, HiRISE, CTX)
- **Configured but not working**: 4 datasets (TES DCI, Nomenclature, Contours, Albedo)
- **Total overlays available**: 11 types (elevation, solar, dust, hillshade, slope, aspect, roughness, TRI, nomenclature, contours, albedo)


