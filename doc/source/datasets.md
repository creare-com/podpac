# Supported Datasets

POPDAC natively supports several third-party datasources and
continue to increase support each release. The following datasets are currently supported:

## SMAP

- **Source**: [NASA Soil Moisture Active Passive (SMAP) Satellites](https://smap.jpl.nasa.gov/data/)
- **Module**: `podpac.datalib.smap`

Global soil moisture measurements from NASA.

### Examples

- [Analyzing SMAP Data](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/basic_examples/analyzing-SMAP-data.ipynb)
- [Running SMAP Analysis on AWS Lambda](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/basic_examples/running-on-aws-lambda.ipynb)
- [SMAP Sentinel data access](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/demos/SMAP-Sentinel-data-access.ipynb)
- [SMAP downscaling example application](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/demos/SMAP-downscaling-example-application.ipynb)
- [SMAP level 4 data access](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/demos/SMAP-level4-data-access.ipynb)
- [SMAP data access widget](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/demos/SMAP-widget-data-access.ipynb)

## TerrainTiles

- **Source**: [Terrain Tiles](https://registry.opendata.aws/terrain-tiles/)
- **Module**: `podpac.datalib.terraintiles`

Global dataset providing bare-earth terrain heights, tiled for easy usage and provided on S3.

### Examples

- [Terrain Tiles Usage](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/demos/Terrain-Tiles.ipynb)
- [Terrain Tiles Pattern Match](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/demos/Terrain-Tiles-Pattern-Match.ipynb)

## GFS

- **Source**: [NOAA Global Forecast System (GFS) Model](https://registry.opendata.aws/noaa-gfs-pds/)
- **Module**: `podpac.datalib.gfs`

Weather forecast model produced by the National Centers for Environmental Prediction (NCEP)

### Examples

- [GFS Usage]()