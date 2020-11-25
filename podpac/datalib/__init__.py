"""
Datalib Public API

This module gets imported in the root __init__.py
and exposed its contents to podpac.datalib
"""

import sys

from podpac.datalib.cosmos_stations import COSMOSStations
from podpac.datalib.drought_monitor import DroughtCategory, DroughtMonitorCategory
from podpac.datalib.egi import EGI
from podpac.datalib.gfs import GFS, GFSLatest
from podpac.datalib.modis_pds import MODIS
from podpac.datalib.satutils import Landsat8, Sentinel2
from podpac.datalib.smap import SMAP as SMAPOpenDAP
from podpac.datalib.smap_egi import SMAP
from podpac.datalib.terraintiles import TerrainTiles
from podpac.datalib.weathercitizen import WeatherCitizen
from podpac.datalib.soilscape import SoilSCAPE20min
from podpac.datalib import soilgrids

# intake requires python >= 3.6
if sys.version >= "3.6":
    from podpac.datalib.intake_catalog import IntakeCatalog
