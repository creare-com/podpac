import pytest
import numpy as np

import podpac
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac import clinspace


class TestCoordinates1d(object):
    """
    See test_array_coordinates1d.py for additional Coordinates1d coverage
    """

    def test_common_api(self):
        c = Coordinates1d(name="lat")

        attrs = [
            "name",
            "is_monotonic",
            "is_descending",
            "is_uniform",
            "start",
            "stop",
            "step",
            "dims",
            "xdims",
            "udims",
            "shape",
            "size",
            "dtype",
            "deltatype",
            "bounds",
            "xcoords",
            "definition",
            "full_definition",
        ]

        for attr in attrs:
            try:
                getattr(c, attr)
            except NotImplementedError:
                pass

        try:
            c.from_definition({})
        except NotImplementedError:
            pass

        try:
            c.copy()
        except NotImplementedError:
            pass

        try:
            c.select([0, 1])
        except NotImplementedError:
            pass

        try:
            c.select([0, 1], outer=True, return_index=True)
        except NotImplementedError:
            pass

        try:
            c._select([0, 1], False, False)
        except NotImplementedError:
            pass

        try:
            c.simplify()
        except NotImplementedError:
            pass

        try:
            c.flatten()
        except NotImplementedError:
            pass

        try:
            c.reshape((10, 10))
        except NotImplementedError:
            pass

        try:
            c.issubset(c)
        except NotImplementedError:
            pass

    def test_horizontal_resolution(self):
        """Test horizontal resolution implentation for Coordinates1d. Edge cases are handled in Coordinates.py"""
        # Latitude
        lat = clinspace(-80, 80, 5)
        lat.name = "lat"  # normally assigned when creating Coords object
        assert type(lat) == podpac.core.coordinates.uniform_coordinates1d.UniformCoordinates1d

        # Longitude
        lon = podpac.clinspace(-180, 180, 5)
        lon.name = "lon"
        assert type(lon) == podpac.core.coordinates.uniform_coordinates1d.UniformCoordinates1d

        # Sample Ellipsoid Tuple
        ell_tuple = (6378.137, 6356.752314245179, 0.0033528106647474805)

        # Sample Coordinate name:
        coord_name = "ellipsoidal"

        # Resolution: nominal
        assert lat.horizontal_resolution(lat, ell_tuple, coord_name) == 3554055.948774749 * podpac.units("meter")
        assert lon.horizontal_resolution(lat, ell_tuple, coord_name) == 0.0 * podpac.units("meter")

        # Resolution: summary
        assert lat.horizontal_resolution(lat, ell_tuple, coord_name, restype="summary") == (
            4442569.935968436 * podpac.units("meter"),
            13040.905617921147 * podpac.units("meter"),
        )
        assert lon.horizontal_resolution(lat, ell_tuple, coord_name, restype="summary") == (
            5558704.3695234 * podpac.units("meter"),
            3399219.0171971265 * podpac.units("meter"),
        )

        # Resolution: full
        lat_answer = [4455610.84158636, 4429529.03035052, 4429529.03035052, 4455610.84158636]

        lon_answer = [
            [1575399.99090356, 1575399.99090356, 1575399.99090356, 1575399.99090356],
            [7311983.84720763, 7311983.84720763, 7311983.84720763, 7311983.84720763],
            [10018754.17139462, 10018754.17139462, 10018754.17139462, 10018754.17139462],
            [7311983.84720763, 7311983.84720763, 7311983.84720763, 7311983.84720763],
            [1575399.99090356, 1575399.99090356, 1575399.99090356, 1575399.99090356],
        ]

        np.testing.assert_array_almost_equal(
            lat.horizontal_resolution(lat, ell_tuple, coord_name, restype="full").magnitude, lat_answer
        )
        np.testing.assert_array_almost_equal(
            lon.horizontal_resolution(lat, ell_tuple, coord_name, restype="full").magnitude, lon_answer
        )

        # Different Units
        np.testing.assert_almost_equal(
            lat.horizontal_resolution(lat, ell_tuple, coord_name).to(podpac.units("feet")).magnitude,
            lat.horizontal_resolution(lat, ell_tuple, coord_name, units="feet").magnitude,
        )
