import warnings
warnings.filterwarnings('ignore')

import unittest
import pytest

import podpac
from podpac.compositor import OrderedCompositor, Compositor
import numpy as np
from podpac.core.data.types import Array

class TestCompositor(object):

    # Mock some data to pass to compositor tests
    def setup_method(self, method):
        self.coord_src = podpac.Coordinates(
            [podpac.clinspace(45, 0, 16), podpac.clinspace(-70, -65, 16), podpac.clinspace(0, 1, 2)],
            dims=['lat', 'lon', 'time'])

        LON, LAT, TIME = np.meshgrid(
            self.coord_src['lon'].coordinates,
            self.coord_src['lat'].coordinates,
            self.coord_src['time'].coordinates)

        self.latSource = LAT
        self.lonSource = LON
        self.timeSource = TIME

        self.nasLat = Array(
            source=LAT.astype(float),
            native_coordinates=self.coord_src,
            interpolation='bilinear')

        self.nasLon = Array(
            source=LON.astype(float),
            native_coordinates=self.coord_src,
            interpolation='bilinear')

        self.nasTime = Array(source=TIME.astype(float),
            native_coordinates=self.coord_src,
            interpolation='bilinear')

        self.sources = np.array([self.nasLat, self.nasLon, self.nasTime])

    # These test that unimplemented or interface methods remain as such.
    # Should those methods  be implemented in Compositor, new tests should be written
    # in the place of these.
    def test_compositor_interface_functions(self):
        self.compositor = Compositor(sources=self.sources)
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
        acoords = podpac.Coordinates([podpac.clinspace(0, 1, 11), podpac.clinspace(0, 1, 11)], dims=['lat', 'lon'])
        bcoords = podpac.Coordinates([podpac.clinspace(2, 3, 10), podpac.clinspace(2, 3, 10)], dims=['lat', 'lon'])
        scoords = podpac.Coordinates([[(0.5, 0.5), (2.5, 2.5)]], dims=['lat_lon'])
        
        a = Array(source=np.random.random(acoords.shape), native_coordinates=acoords)
        b = Array(source=-np.random.random(bcoords.shape), native_coordinates=bcoords)
        composited = OrderedCompositor(
            sources=np.array([a, b]),
            cache_native_coordinates=True, 
            source_coordinates=scoords,
            interpolation='nearest')
        c = podpac.Coordinates([0.5, 0.5], dims=['lat', 'lon'])
        o = composited.eval(c)
        np.testing.assert_array_equal(o.data, a.source[5, 5])

    def test_ordered_compositor(self):
        self.orderedCompositor = OrderedCompositor(
            sources=self.sources, shared_coordinates=self.coord_src, cache_native_coordinates=False, threaded=True)
        result = self.orderedCompositor.eval(coordinates=self.coord_src)
        assert True == self.orderedCompositor.evaluated
        # assert self.orderedCompositor._native_coordinates_default().dims == self.coord_src.dims

    def test_source_coordinates_ordered_compositor(self):
        self.orderedCompositor = OrderedCompositor(sources=self.sources, shared_coordinates=self.coord_src,
                                                          cache_native_coordinates=False, threaded=True)

    def test_caching_ordered_compositor(self):
        self.orderedCompositor = OrderedCompositor(sources=self.sources, shared_coordinates=self.coord_src,
                                                          threaded=True)
        
    def test_heterogeous_sources_composited(self):
        anative = podpac.Coordinates([podpac.clinspace((0, 1), (1, 2), size=3)], dims=['lat_lon'])
        bnative = podpac.Coordinates([podpac.clinspace(-2, 3, 3), podpac.clinspace(-1, 4, 3)], dims=['lat', 'lon'])
        a = Array(source=np.random.rand(3), native_coordinates=anative)
        b = Array(source=np.random.rand(3, 3) + 2, native_coordinates=bnative)
        c = OrderedCompositor(sources=np.array([a, b]), interpolation='bilinear')
        coords = podpac.Coordinates([podpac.clinspace(-3, 4, 32), podpac.clinspace(-2, 5, 32)], dims=['lat', 'lon'])
        o = c.eval(coords)
        # Check that both data sources are being used in the interpolation
        mask = o.data >= 2
        assert mask.sum() > 0 
        mask = o.data <= 1
        assert mask.sum() > 0
        
