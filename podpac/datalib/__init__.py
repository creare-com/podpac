"""
Datalib Public API

This module gets imported in the root __init__.py
and exposed its contents to podpac.datalib
"""

import sys

from podpac.datalib import cosmos_stations
from podpac.datalib import drought_monitor
from podpac.datalib import egi
from podpac.datalib import gfs
from podpac.datalib import modis_pds
from podpac.datalib import satutils
from podpac.datalib import smap
from podpac.datalib import smap_egi
from podpac.datalib import terraintiles
from podpac.datalib import weathercitizen


# intake requires python >= 3.6
if sys.version >= "3.6":
    from podpac.datalib.intake_catalog import IntakeCatalog
