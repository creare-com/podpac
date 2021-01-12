import podpac

node = podpac.data.Rasterio(source=r"D:\soilmap\MOD13Q1.A2013033.h08v05.006.2015256072248.hdf")
node.subdatasets
node2 = podpac.data.Rasterio(source=node.subdatasets[0])

node = podpac.data.Rasterio(source="s3://podpac-internal-test/MOD13Q1.A2013033.h08v05.006.2015256072248.hdf")


import warnings

warnings.filterwarnings("ignore")
import podpac
import podpac.datalib
from podpac import datalib
import ipywidgets as widgets
import logging
import rasterio

logger = logging.getLogger("podpac")
logger.setLevel(logging.DEBUG)

podpac.settings["AWS_REQUESTER_PAYS"] = True

from podpac.datalib import cosmos_stations
from podpac.datalib.modis_pds import MODIS
from podpac.datalib.satutils import Landsat8, Sentinel2

terrain2 = datalib.terraintiles.TerrainTiles(zoom=2)
terrain10 = datalib.terraintiles.TerrainTiles(zoom=10)
modis = MODIS(product="MCD43A4.006", data_key="B01")
landsat_b1 = Landsat8(asset="B1", min_bounds_span={"time": "4,D"})
cosmos = cosmos_stations.COSMOSStations()

from podpac.datalib.satutils import Landsat8, Sentinel2

sentinel2_b2 = Sentinel2(asset="B02", min_bounds_span={"time": "4,D"})

lat = podpac.crange(60, 10, -2.0)  # (start, stop, step)
lon = podpac.crange(-130, -60, 2.0)  # (start, stop, step)

# Specify date and time
time = "2019-04-23"

# Create the PODPAC Coordinates
# lat_lon_time_US = podpac.Coordinates([lat, lon, time], dims=['lat', 'lon', 'time'])
lat_lon_time_DHMC = podpac.Coordinates(
    [podpac.clinspace(43.7125, 43.6, 256), podpac.clinspace(-72.3125, -72.3, 256), time], dims=["lat", "lon", "time"]
)

o = sentinel2_b2.eval(lat_lon_time_DHMC)

source = sentinel2_b2.sources[0]
a = source.dataset.read(1, window=((5654, 6906), (1655, 1796)))
b = source.dataset.read(1, window=((0, 256), (0, 256)))

source.force_eval = True
o2 = source.eval(lat_lon_time_DHMC)

# with source.s3.open(source.source) as fobj:
# with rasterio.MemoryFile(fobj) as mf:
# ds = mf.open()

print("done")
