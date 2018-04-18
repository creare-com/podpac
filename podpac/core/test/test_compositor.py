import warnings
warnings.filterwarnings('ignore')

import unittest
import pytest
import podpac

from podpac import datalib
from podpac.datalib import smap

class TestCompositor(object):
    @pytest.mark.skip(reason="unfinished")
    def test_threaded(self):
        datalib.smap.SMAP_PRODUCT_MAP.product.data.tolist()

        product = 'SPL4SMAU.003'
        smap = datalib.smap.SMAP(product=product, interpolation='nearest_preview')
        # The available coordinates are built from all available SMAP sources for this product
        # This requires multiple http get requests from the DAAC OpenDAP server, and may take a while
        # Because of this, the results are cached after the first run
        smap.native_coordinates 

        coordinates_point = \
            podpac.Coordinate(lat=39., lon=-77, 
                              time=('2017-09-01', '2017-10-31', '1,D'), 
                              order=['lat', 'lon', 'time'])
        smap.threaded = True
        output = smap.execute(coordinates_point)
        smap.threaded = False

class TestOrderedCompositor(object):
    pass