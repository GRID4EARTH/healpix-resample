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

The scene-level high-resolution textures and the 360 multi-region patches were
derived from Esri World Imagery at zoom level 17.

Source service:

Esri World Imagery  
https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9

Imagery attribution:

Copyright Esri and the applicable World Imagery data providers.

Acquisition dates: [TO COMPLETE]

Provider-specific attribution notices: [TO COMPLETE]

Redistribution permission or licence reference: [TO COMPLETE]

Patch locations, tile coordinates, acquisition provenance, and checksums are
provided in:

- `notebooks/tables/multi_patch_sites.csv`
- `notebooks/tables/multi_patch_tile_provenance.csv`
- `data_manifest.csv`

Esri World Imagery can contain imagery supplied by third-party providers.
Inclusion of the derived Zarr stores in the public archive is subject to the
applicable Esri and provider-specific redistribution terms.

Esri terms of use:
https://www.esri.com/en-us/legal/terms/web-site-service

## Software

The processing and resampling software is available from:

- Repository: https://github.com/GRID4EARTH/healpix-resample
- Software concept DOI: https://doi.org/10.5281/zenodo.21723671
- Software version DOI: https://doi.org/10.5281/zenodo.21723672

The exact Git commit used for this dataset is recorded in `git_commit.txt`.

