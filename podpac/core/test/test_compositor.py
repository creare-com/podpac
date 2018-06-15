import warnings
warnings.filterwarnings('ignore')

import unittest
import pytest
import podpac
from podpac import Coordinate
from podpac import OrderedCompositor
import numpy as np
from podpac.core.data.type import NumpyArray

class TestCompositor(object):

    # Mock some data to pass to compositor tests
    def setup_method(self, method):
        self.coord_src = podpac.Coordinate(
            lat=(45, 0, 16),
            lon=(-70., -65., 16),
            time=(0, 1, 2),
            order=['lat', 'lon', 'time'])

        LON, LAT, TIME = np.meshgrid(
            self.coord_src['lon'].coordinates,
            self.coord_src['lat'].coordinates,
            self.coord_src['time'].coordinates)

        self.latSource = LAT
        self.lonSource = LON
        self.timeSource = TIME

        self.nasLat = NumpyArray(
            source=LAT.astype(float),
            native_coordinates=self.coord_src,
            interpolation='bilinear')

        self.nasLon = NumpyArray(
            source=LON.astype(float),
            native_coordinates=self.coord_src,
            interpolation='bilinear')

        self.nasTime = NumpyArray(source=TIME.astype(float),
            native_coordinates=self.coord_src,
            interpolation='bilinear')

        self.sources = np.array([self.nasLat, self.nasLon, self.nasTime])

    # These test that unimplemented or interface methods remain as such.
    # Should those methods  be implemented in Compositor, new tests should be written
    # in the place of these.
    def testCompositorInterfaceFunctions(self):
        self.compositor = podpac.Compositor(sources=self.sources)
        with pytest.raises(NotImplementedError):
            self.compositor._shared_coordinates_default()
        with pytest.raises(NotImplementedError):
            self.compositor.composite(outputs=None)
        assert None == self.compositor.get_source_coordinates()
        assert self.compositor._source_coordinates_default() == self.compositor.get_source_coordinates()

    # TODO Test non None (basic cases) just to call unstable methods.
    # These functions are volatile, and may be difficult to test until their
    # spec is complete.
    @pytest.mark.skip(reason="unifinished")
    def testCompositorImplementedFunctions(self):
        pass

    # Simple test of creating and executing an OrderedCompositor
    def testOrderedCompositor(self):
        self.orderedCompositor = podpac.OrderedCompositor(sources=self.sources, shared_coordinates=self.coord_src, cache_native_coordinates=False, threaded=True)
        result = self.orderedCompositor.execute(coordinates=self.coord_src)
        assert True == self.orderedCompositor.evaluated
        assert self.orderedCompositor._native_coordinates_default().dims == self.coord_src.dims

    @pytest.mark.skip("not-yet-working")
    def testSourceCoordinatesOrderedCompositor(self):
        self.orderedCompositor = podpac.OrderedCompositor(sources=self.sources, shared_coordinates=self.coord_src, cache_native_coordinates=False, threaded=True, source_coordinates=self.source_coords)

    @pytest.mark.skip("not-yet-working")
    def testCachingOrderedCompositor(self):
        self.orderedCompositor = podpac.OrderedCompositor(sources=self.sources, shared_coordinates=self.coord_src, threaded=True)
