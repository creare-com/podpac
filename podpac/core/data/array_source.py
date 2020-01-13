"""
Array Datasource
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
from six import string_types

import numpy as np
import traitlets as tl
import pandas as pd  # Core dependency of xarray

from podpac.core.utils import common_doc, ArrayTrait
from podpac.core.data.datasource import COMMON_DATA_DOC, DataSource
from podpac.core.cache import CacheCtrl
from podpac.core.coordinates import Coordinates


class Array(DataSource):
    """Create a DataSource from an array -- this node is mostly meant for small experiments
    
    Attributes
    ----------
    source : np.ndarray
        Numpy array containing the source data
    native_coordinates : podpac.Coordinates
        The coordinates of the source data
        
    Notes
    ------
    `native_coordinates` need to supplied by the user when instantiating this node.
    
    This Node is not meant for large arrays, and cause issues with caching. As such, this Node override the default 
    cache behavior as having no cache -- its data is in RAM already and caching is not helpful.
    
    Example
    ---------
    >>> # Create a time series of 10 32x34 images with R-G-B channels
    >>> import numpy as np
    >>> import podpac
    >>> data = np.random.rand(10, 32, 34, 3)
    >>> coords = podpac.Coordinates([podpac.clinspace(1, 10, 10, 'time'), 
                                     podpac.clinspace(1, 32, 32, 'lat'),
                                     podpac.clinspace(1, 34, 34, 'lon')])
    >>> node = podpac.data.Array(source=data, native_coordinates=coords, outputs=['R', 'G', 'B'])
    >>> output = node.eval(coords)
    """

    source = ArrayTrait().tag(readonly=True)
    native_coordinates = tl.Instance(Coordinates, allow_none=False).tag(attr=True)

    @tl.default("cache_ctrl")
    def _cache_ctrl_default(self):
        return CacheCtrl([])

    @tl.validate("source")
    def _validate_source(self, d):
        a = d["value"]
        try:
            a.astype(float)
        except:
            raise ValueError("Array source must be numerical")
        return a

    def _first_init(self, **kwargs):
        # If Array is being created from Node.from_definition or Node.from_json, then we have to handle the
        # native coordinates specifically. This is special. No other DataSource node needs to deserialize
        # native_coordinates in this way because it is implemented specifically in the node through get_coordinates
        if isinstance(kwargs.get("native_coordinates"), OrderedDict):
            kwargs["native_coordinates"] = Coordinates.from_definition(kwargs["native_coordinates"])
        elif isinstance(kwargs.get("native_coordinates"), string_types):
            kwargs["native_coordinates"] = Coordinates.from_json(kwargs["native_coordinates"])

        return kwargs

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        s = coordinates_index
        d = self.create_output_array(coordinates, data=self.source[s])
        return d
