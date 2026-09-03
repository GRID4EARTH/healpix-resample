# Data attribution

## Copernicus Sentinel-2

Contains modified Copernicus Sentinel data 2026.

The archived extracts originate from the following Sentinel-2 Level-2A
products:

- Urban: `S2B_MSIL2A_20260527T105619_N0512_R094_T31UDQ_20260527T133135`
- Water: `S2A_MSIL2A_20260624T103041_N0512_R108_T32TLS_20260624T182910`
- Forest: `S2A_MSIL2A_20260704T102701_N0512_R108_T32UMU_20260704T170718`
- Agriculture: `S2B_MSIL2A_20260623T104619_N0512_R051_T31UCP_20260623T132606`

The extracts were spatially subset and converted to Zarr for the experiments.
Their checksums and repository paths are recorded in `data_manifest.csv`.

Source legal notice:
https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice

## ERA5

ERA5 data were obtained from the Copernicus Climate Change Service operated
by ECMWF.

Source dataset:

Complete ERA5 global atmospheric reanalysis  
DOI: https://doi.org/10.24381/cds.143582cf

The archived GRIB file contains an ensemble surface-flux extract for
2024-06-01 at 06:00 UTC, ensemble members 0–9, on the N256 grid, with forecast
steps of 3 and 6 hours.

## Esri World Imagery

The scene-level high-resolution textures and the 360 multi-region patches
are derived from Esri World Imagery at zoom level 17, read from the pinned
Wayback release 26334 (2026-08-05).

These derivatives are NOT redistributed: Esri's terms grant redistribution
for static map images, not for machine-readable derived datasets, and the
underlying imagery belongs to third-party providers. Each user regenerates
the derivatives locally by running the notebooks' acquisition cells; the
Wayback release pin makes the regeneration deterministic and
`data_manifest.csv` records the expected SHA-256 of every store.

Required attribution for any display of the imagery or derivatives:
"Sources: Esri, Maxar, Earthstar Geographics, and the GIS User Community."

Source service (versioned snapshots):
https://livingatlas.arcgis.com/wayback/
