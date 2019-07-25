from __future__ import division, unicode_literals, print_function, absolute_import

import pytest

import podpac.core.common_test_utils as ctu
from podpac.core.coordinates import UniformCoordinates1d, ArrayCoordinates1d


class TestMakeCoordinates(object):
    def test_default_creation(self):
        # Just make sure it runs
        coords = ctu.make_coordinate_combinations()
        assert len(coords) > 0
        assert len(coords) == 168

    def test_custom_creation_no_stack(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([2, 3, 4, 5, 6], name="lon")
        alt = ArrayCoordinates1d([6, 7, 8, 9, 10, 11, 12], name="alt")
        time = ArrayCoordinates1d(["2018-01-01", "2018-02-01"], name="time")
        coords = ctu.make_coordinate_combinations(lat=lat, lon=lon, alt=alt, time=time)
        assert len(coords) > 0
        assert len(coords) == 48

    def test_custom_creation_latlon_stack(self):
        alt = ArrayCoordinates1d([6, 7, 8, 9, 10, 11, 12], name="alt")
        time = ArrayCoordinates1d(["2018-01-01", "2018-02-01"], name="time")
        coords = ctu.make_coordinate_combinations(alt=alt, time=time)
        assert len(coords) > 0
        assert len(coords) == 70

    def test_custom_creation_mixed_type_1d(self):
        lat = ArrayCoordinates1d([0.0, 1.0, 2.0, 4.0], name="lat")
        coords = ctu.make_coordinate_combinations(lat=lat)
        assert len(coords) > 0
        assert len(coords) == 84
