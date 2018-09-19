"""
Test interpolation methods
"""

import pytest
from traitlets import TraitError

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.data import interpolate
from podpac.core.data.interpolate import (Interpolator, InterpolationException,
                                          InterpolationMethod, INTERPOLATION_METHODS,
                                          INTERPOLATION_SHORTCUTS, NearestNeighbor, NearestPreview)

COORDINATES = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=['lat', 'lon'])
STACKED_COORDINATES = Coordinates([([1,2,3,4,5], [0,1,2,3,4])], dims=['lat_lon'])

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

        def test_interpolator_default_method(self):
            
            # should throw an error if default_method is not in INTERPOLATION_SHORTCUTS
            Interpolator('nearest', COORDINATES, default_method='nearest')
            with pytest.raises(InterpolationException):
                Interpolator('nearest', COORDINATES, default_method='test')

            # should fill in methods not included in definition dictionary
            coords = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=['lat', 'lon'])
            interp = Interpolator({'lat': 'nearest'}, coords, default_method='nearest_preview')
            assert interp._definition['lon']
            assert interp._definition['lon'] ==INTERPOLATION_METHODS['nearest_preview']

        
        def test_interpolator_init_type(self):
            """test constructor
            """

            # should throw an error if definition is not str, dict, or InterpolationMethod
            with pytest.raises(ValueError):
                Interpolator(5, COORDINATES)

        
        def test_str_definition(self):
            # should throw an error if string input is not one of the INTERPOLATION_SHORTCUTS
            with pytest.raises(InterpolationException):
                Interpolator('test', COORDINATES)

            interp = Interpolator('nearest', COORDINATES)
            assert interp._definition['lat']
            assert interp._definition['lat'] == INTERPOLATION_METHODS['nearest']

        def test_dict_definition(self):

            # should throw an error on _parse_interpolation_method(definition) 
            # if definition is not in INTERPOLATION_SHORTCUTS
            with pytest.raises(InterpolationException):
                Interpolator({'lat': 'test'}, COORDINATES)

            # handle string methods
            interp = Interpolator({'lat': 'nearest'}, COORDINATES)
            assert interp._definition['lat'] == INTERPOLATION_METHODS['nearest']






