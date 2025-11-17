# Mars Data Sources

This document provides information about Mars DEM data sources and how to obtain them.

## MOLA (Mars Orbiter Laser Altimeter)

MOLA provides global Mars elevation data at various resolutions.

### Automatic Download
The application attempts to download MOLA data automatically from configured URLs. If automatic download fails, you can manually download data.

### Manual Download Options

1. **USGS Astrogeology Science Center**
   - Website: https://astrogeology.usgs.gov/search/map/Mars/GlobalSurveyor/MOLA
   - Search for "MOLA DEM mosaic"
   - Download the global 463m resolution GeoTIFF

2. **NASA Planetary Data System (PDS)**
   - Website: https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/
   - Navigate to the data directory
   - Download MEGDR files

3. **Planetary Maps**
   - Website: https://planetarymaps.usgs.gov/
   - Browse Mars datasets

### File Placement
After downloading, place the DEM file in the cache directory:
```
data/cache/mola_<hash>.tif
```

The hash is generated from the dataset name and ROI. You can also rename your downloaded file to match the expected cache filename, or the application will use it if it's the only MOLA file in the cache.

## HiRISE (High Resolution Imaging Science Experiment)

HiRISE provides very high resolution (1m) DEMs for specific regions.

### Sources
- **HiRISE PDS**: https://www.uahirise.org/hiwish/
- **AWS S3**: https://s3.amazonaws.com/mars-hirise-pds/

Note: HiRISE data is region-specific and requires selecting specific observation IDs.

## CTX (Context Camera)

CTX provides medium resolution (18m) images and DEMs.

### Sources
- **WUSTL ODE**: https://ode.rsl.wustl.edu/mars/
- Navigate to CTX data section

## Troubleshooting

### Download Failures

If automatic downloads fail:

1. **Check Internet Connection**: Ensure you have internet access
2. **Verify URL**: Data source URLs may change. Check the official sources above
3. **Manual Download**: Download the file manually and place it in the cache directory
4. **Update Config**: Update `marshab_config.yaml` with a working URL

### Cache Issues

To clear the cache and force re-download:
```bash
rm -rf data/cache/*
# Or on Windows:
Remove-Item -Recurse -Force data\cache\*
```

### File Format

All DEM files should be in GeoTIFF format (.tif) with proper georeferencing information.




