"""
Landsat 8 on AWS OpenData

https://registry.opendata.aws/landsat-8/

Leverages sat-utils (https://github.com/sat-utils) developed by Development Seed
"""

import logging
import datetime

import numpy as np
import traitlets as tl

from podpac.datalib.satutils import SatUtils

_logger = logging.getLogger(__name__)

COLLECTION = "landsat-8-l1"


class LandsatSource(SatUtils):
    """
    Landsat 8 on AWS OpenData
    https://registry.opendata.aws/landsat-8/

    Leverages sat-utils (https://github.com/sat-utils) developed by Development Seed

    Attributes
    ----------
    query : dict, optional
        Dictionary of properties to query on, supports eq, lt, gt, lte, gte
        Passed through to the sat-search module.
        See https://github.com/sat-utils/sat-search/blob/master/tutorial-1.ipynb
        Defaults to None
    asset : str, optional
        Asset to download from the satellite image.
        The asset must be a band name or a common extension name, see https://github.com/radiantearth/stac-spec/tree/master/extensions/eo
        See also the Assets section of this tutorial: https://github.com/sat-utils/sat-stac/blob/master/tutorial-2.ipynb 
        Defaults to "MTL"
    """

    collection = "landsat-8-l1"
