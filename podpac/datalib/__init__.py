"""
Datalib Public API

This module gets imported in the root __init__.py
and exposed its contents to podpac.datalib
"""

from podpac.datalib import smap
from podpac.datalib.smap import (
    SMAP, SMAPBestAvailable, SMAPSource, SMAPPorosity, SMAPProperties,
    SMAPWilt, SMAP_PRODUCT_MAP
)
