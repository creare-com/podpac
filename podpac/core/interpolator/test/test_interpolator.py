"""
Test interpolation methods


"""
# pylint: disable=C0111,W0212,R0903

import traitlets as tl
import numpy as np

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.interpolator.interpolator import Interpolator


class TestInterpolator(object):
    """Test abstract interpolator class"""

    def test_can_select(self):
        class CanAlwaysSelect(Interpolator):
            def can_select(self, udims, reqcoords, srccoords):
                return udims

        class CanNeverSelect(Interpolator):
            def can_select(self, udims, reqcoords, srccoords):
                return tuple()

        interp = CanAlwaysSelect(method="method")
        can_select = interp.can_select(("time", "lat"), None, None)
        assert "lat" in can_select and "time" in can_select

        interp = CanNeverSelect(method="method")
        can_select = interp.can_select(("time", "lat"), None, None)
        assert not can_select

    def test_dim_in(self):
        interpolator = Interpolator(methods_supported=["test"], method="test")

        coords = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=["lat", "lon"])
        assert interpolator._dim_in("lat", coords)
        assert interpolator._dim_in("lat", coords, unstacked=True)
        assert not interpolator._dim_in("time", coords)

        coords_two = Coordinates([clinspace(0, 10, 5)], dims=["lat"])
        assert interpolator._dim_in("lat", coords, coords_two)
        assert not interpolator._dim_in("lon", coords, coords_two)

        coords_three = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lat_lon"])
        assert not interpolator._dim_in("lat", coords, coords_two, coords_three)
        assert interpolator._dim_in("lat", coords, coords_two, coords_three, unstacked=True)
