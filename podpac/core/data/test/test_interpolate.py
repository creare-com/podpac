"""
Test interpolation methods
"""

import pytest
from traitlets import TraitError

from podpac.core.coordinates import Coordinates
from podpac.core.data import interpolate
from podpac.core.data.interpolate import ( Interpolator, InterpolationException,
                                           InterpolationMethod, INTERPOLATION_METHODS, 
                                           INTERPOLATION_SHORTCUTS )


class TestInterpolate(object):

    """ Test interpolation methods
    """

    def test_interpolate_module(self):
        """smoke test interpolate module"""

        assert interpolate is not None

    def test_allow_missing_modules(self):
        """TODO: Allow user to be missing rasterio and scipy"""
        pass


    class TestInterpolator(object):

        """ Test Interpolator class
        """

        def test_interpolator_init(self):
            """test constructor
            """


            # should throw an error if string input is not one of the INTERPOLATION_SHORTCUTS
            Interpolator('nearest')
            with pytest.raises(InterpolationException):
                Interpolator('test')




