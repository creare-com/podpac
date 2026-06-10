import pytest
import numpy as np

from podpac.core.coordinates.affine_coordinates import AffineCoordinates
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d

# origin [10, 20], pixel size [3, 2], north up
GEOTRANSFORM_NORTHUP = (10.0, 2.0, 0.0, 20.0, 0.0, -3.0)

# origin [10, 20], step [2, 3], rotated 20 degrees
GEOTRANSFORM_ROTATED = (10.0, 1.879, -1.026, 20.0, 0.684, 2.819)

from podpac import Coordinates

UNIFORM = Coordinates.from_geotransform(geotransform=GEOTRANSFORM_NORTHUP, shape=(3, 4))


class TestAffineCoordinatesCreation(object):
    def test_init(self):
        c = AffineCoordinates(geotransform=GEOTRANSFORM_NORTHUP, shape=(3, 4))

        assert c.geotransform == GEOTRANSFORM_NORTHUP
        assert c.shape == (3, 4)
        assert c.is_affine
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert len(set(c.xdims)) == 2
        assert c.name == "lat_lon"
        _ = repr(c)

    def test_rotated(self):
        c = AffineCoordinates(geotransform=GEOTRANSFORM_ROTATED, shape=(3, 4))

        assert c.geotransform == GEOTRANSFORM_ROTATED
        assert c.shape == (3, 4)
        assert c.is_affine
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert len(set(c.xdims)) == 2
        assert c.name == "lat_lon"
        _ = repr(c)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid shape"):
            AffineCoordinates(geotransform=GEOTRANSFORM_NORTHUP, shape=(-3, 4))

        with pytest.raises(ValueError, match="Invalid shape"):
            AffineCoordinates(geotransform=GEOTRANSFORM_NORTHUP, shape=(3, 0))

    def test_size_one_simplify(self):
        c = AffineCoordinates(geotransform=GEOTRANSFORM_NORTHUP, shape=(1, 1))
        c2 = c.simplify()
        assert isinstance(c2[0], UniformCoordinates1d)
        assert isinstance(c2[1], UniformCoordinates1d)
        assert c2[0].shape == (1,)
        assert c2[1].shape == (1,)
        assert c2[0].step == GEOTRANSFORM_NORTHUP[-1]
        assert c2[1].step == GEOTRANSFORM_NORTHUP[1]
        assert c2[1].name == "lon"
        assert c2[0].name == "lat"


class TestAffineCoordinatesProperties(object):
    def test_origin(self):
        c = AffineCoordinates(geotransform=GEOTRANSFORM_NORTHUP, shape=(3, 4))
        np.testing.assert_array_equal(c.origin, [20.0, 10.0])

    def test_origin_rotated(self):
        # lat, lon
        c = AffineCoordinates(geotransform=GEOTRANSFORM_ROTATED, shape=(3, 4))
        np.testing.assert_array_equal(c.origin, [20.0, 10.0])

    def test_coordinates(self):
        c = AffineCoordinates(geotransform=GEOTRANSFORM_NORTHUP, shape=(3, 4))

        assert c.coordinates.shape == (3, 4, 2)

        lat = c.coordinates[:, :, 0]
        lon = c.coordinates[:, :, 1]

        np.testing.assert_allclose(
            lat,
            [
                [18.5, 18.5, 18.5, 18.5],
                [15.5, 15.5, 15.5, 15.5],
                [12.5, 12.5, 12.5, 12.5],
            ],
        )

        np.testing.assert_allclose(
            lon,
            [
                [11.0, 13.0, 15.0, 17.0],
                [11.0, 13.0, 15.0, 17.0],
                [11.0, 13.0, 15.0, 17.0],
            ],
        )
