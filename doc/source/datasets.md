# Supported Datasets

POPDAC natively supports several third-party datasources and will
continue to expand support each release. The following datasets are currently supported:

## SMAP

- **Source**: [NASA Soil Moisture Active Passive (SMAP) Satellites](https://smap.jpl.nasa.gov/data/)
- **Module**: `podpac.datalib.smap`, `podpac.datalib.smap_egi`

Global soil moisture measurements from NASA.

### Examples

- [Retrieving SMAP Data](https://github.com/creare-com/podpac-examples/blob/master/notebooks/5-datalib/smap/010-retrieving-SMAP-data.ipynb)
- [Analyzing SMAP Data](https://github.com/creare-com/podpac-examples/blob/master/notebooks/5-datalib/smap/100-analyzing-SMAP-data.ipynb)
- [Working with SMAP-Sentinel Data](https://github.com/creare-com/podpac-examples/blob/master/notebooks/5-datalib/smap/101-working-with-SMAP-Sentinel-data.ipynb)
- [SMAP-EGI](https://github.com/creare-com/podpac-examples/blob/master/notebooks/5-datalib/smap/SMAP-EGI.ipynb)
- [SMAP Data Access Without PODPAC](https://github.com/creare-com/podpac-examples/blob/master/notebooks/5-datalib/smap/SMAP-data-access-without-podpac.ipynb)
- [SMAP Downscaling Example Application](https://github.com/creare-com/podpac-examples/blob/master/notebooks/5-datalib/smap/SMAP-downscaling-example-application.ipynb)

## TerrainTiles

- **Source**: [Terrain Tiles](https://registry.opendata.aws/terrain-tiles/)
- **Module**: `podpac.datalib.terraintiles`

Global dataset providing bare-earth terrain heights, tiled for easy usage and provided on S3.

### Examples

- [Terrain Tiles Usage](https://github.com/creare-com/podpac-examples/blob/master/notebooks/5-datalib/terrtain-tiles.ipynb)
- [Terrain Tiles Pattern Match](https://github.com/creare-com/podpac-examples/blob/master/notebooks/scratch/demos/Terrain-Tiles-Pattern-Match.ipynb)

## GFS

- **Source**: [NOAA Global Forecast System (GFS) Model](https://registry.opendata.aws/noaa-gfs-pds/)
- **Module**: `podpac.datalib.gfs`

Weather forecast model produced by the National Centers for Environmental Prediction (NCEP)

### Examples

- [GFS Usage](https://github.com/creare-com/podpac-examples/blob/master/notebooks/5-datalib/gfs.ipynb)