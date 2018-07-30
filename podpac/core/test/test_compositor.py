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
    def test_compositor_interface_functions(self):
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
    def test_compositor_implemented_functions(self):
        acoords = Coordinate(lat=(0, 1, 11), lon=(0, 1, 11), order=['lat', 'lon'])
        bcoords = Coordinate(lat=(2, 3, 10), lon=(2, 3, 10), order=['lat', 'lon'])
        scoords = Coordinate(lat_lon=[[0.5, 2.5], [0.5, 2.5]])
        
        a = NumpyArray(source=np.random.random(acoords.shape), native_coordinates=acoords)
        b = NumpyArray(source=-np.random.random(bcoords.shape), native_coordinates=bcoords)
        composited = OrderedCompositor(sources=np.array([a, b]), cache_native_coordinates=True, 
                                                                 source_coordinates=scoords, 
                                                                 interpolation='nearest')
        c = Coordinate(lat=0.5, lon=0.5, order=['lat', 'lon'])
        o = composited.execute(c)
        np.testing.assert_array_equal(o.data, a.source[5, 5])

    # Simple test of creating and executing an OrderedCompositor
    def test_ordered_compositor(self):
        self.orderedCompositor = podpac.OrderedCompositor(sources=self.sources, shared_coordinates=self.coord_src,
                                                          cache_native_coordinates=False, threaded=True)
        result = self.orderedCompositor.execute(coordinates=self.coord_src)
        assert True == self.orderedCompositor.evaluated
        assert self.orderedCompositor._native_coordinates_default().dims == self.coord_src.dims

    def test_source_coordinates_ordered_compositor(self):
        self.orderedCompositor = podpac.OrderedCompositor(sources=self.sources, shared_coordinates=self.coord_src,
                                                          cache_native_coordinates=False, threaded=True)

    def test_caching_ordered_compositor(self):
        self.orderedCompositor = podpac.OrderedCompositor(sources=self.sources, shared_coordinates=self.coord_src,
                                                          threaded=True)
        
    def test_heterogeous_sources_composited(self):
        a = NumpyArray(source=np.random.rand(3), 
                       native_coordinates=Coordinate(lat_lon=((0, 1), (1, 2), 3)))
        b = NumpyArray(source=np.random.rand(3, 3) + 2,
                       native_coordinates=Coordinate(lat=(-2, 3, 3), lon=(-1, 4, 3), order=['lat', 'lon'])))
        c = podpac.OrderedCompositor(sources=np.array([a, b]), interpolation='bilinear')
        coords = Coordinate(lat=(-3, 4, 32), lon=(-2, 5, 32), order=['lat', 'lon'])
        o = c.execute(coords)
        # Check that both data sources are being used in the interpolation
        mask = o.data >= 2
        assert mask.sum() > 0 
        mask = o.data <= 1
        assert mask.sum() > 0
        
