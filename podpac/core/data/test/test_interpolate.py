"""
Test interpolation methods
"""

import pytest
from traitlets import TraitError

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.data import interpolate
from podpac.core.data.interpolate import (Interpolator, InterpolationException,
                                          InterpolationMethod, INTERPOLATION_METHODS, TOLERANCE_DEFAULTS,
                                          INTERPOLATION_SHORTCUTS, NearestNeighbor, NearestPreview,
                                          Rasterio)

COORDINATES = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=['lat', 'lon'])
STACKED_COORDINATES = Coordinates([([1, 2, 3, 4, 5],  [0, 1, 2, 3, 4])], dims=['lat_lon'])

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
            

        
        def test_interpolator_init_type(self):
            """test constructor
            """

            # should throw an error if definition is not str, dict, or InterpolationMethod
            with pytest.raises(TypeError):
                Interpolator(5, COORDINATES)

        
        def test_str_definition(self):
            # should throw an error if string input is not one of the INTERPOLATION_SHORTCUTS
            with pytest.raises(InterpolationException):
                Interpolator('test', COORDINATES)

            interp = Interpolator('nearest', COORDINATES)
            assert interp._definition['lat']
            assert isinstance(interp._definition['lat'], list)
            assert interp._definition['lat'] == INTERPOLATION_METHODS['nearest']
            assert interp._definition['lon'] == INTERPOLATION_METHODS['nearest']

        def test_dict_definition(self):

            # should throw an error on _parse_interpolation_method(definition) 
            # if definition is not in INTERPOLATION_SHORTCUTS
            with pytest.raises(InterpolationException):
                Interpolator({'lat': 'test'}, COORDINATES)

            # handle string methods
            interp = Interpolator({'lat': 'nearest'}, COORDINATES)
            assert isinstance(interp._definition['lat'], list)
            assert interp._definition['lat'] == INTERPOLATION_METHODS['nearest']

            
            # should throw an error if default_method is not in INTERPOLATION_SHORTCUTS
            Interpolator({'lat': 'nearest'}, COORDINATES, default_method='nearest')
            with pytest.raises(InterpolationException):
                Interpolator({'lat': 'nearest'}, COORDINATES, default_method='test')

            # should fill in methods not included in definition dictionary
            interp = Interpolator({'lat': 'nearest'}, COORDINATES, default_method='bilinear')
            assert interp._definition['lon']
            assert isinstance(interp._definition['lon'], list)
            assert interp._definition['lon'] == INTERPOLATION_METHODS['bilinear']


        def test_interpolation_method_definition(self):

            interp = Interpolator(NearestNeighbor, COORDINATES)
            assert interp._definition['lat']
            assert isinstance(interp._definition['lat'], list)
            assert interp._definition['lat'] == [NearestNeighbor]


            # should throw an error if items are not InterpolationMethods
            with pytest.raises(TypeError):
                Interpolator(['nearest'], COORDINATES)

        def test_list_definition(self):

            interp_list = [NearestNeighbor, Rasterio, NearestPreview]
            interp = Interpolator(interp_list, COORDINATES)
            assert interp._definition['lat']
            assert isinstance(interp._definition['lat'], list)
            assert interp._definition['lat'] == interp_list


            # should throw an error if items are not InterpolationMethods
            with pytest.raises(TypeError):
                Interpolator(['nearest'], COORDINATES)


        def test_wrong_tolerance_type(self):

            with pytest.raises(TypeError):
                Interpolator('nearest', COORDINATES, tolerance='test')
 

        def test_number_tolerance(self):
            interp = Interpolator('nearest', COORDINATES, tolerance=5)
            assert interp._tolerance['lat']
            assert interp._tolerance['lat'] == 5
            assert interp._tolerance['lon'] == 5


        def test_dict_tolerance(self):

            # test defaults
            interp = Interpolator('nearest', COORDINATES, tolerance={'lat': 5})
            assert interp._tolerance['lat'] == 5
            assert interp._tolerance['lon'] == TOLERANCE_DEFAULTS['lon']

            # coords don't allow this yet
            # coords = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=['lat', 'other'])
            # interp = Interpolator('nearest', coords, tolerance={'lat': 5})
            # assert interp._tolerance['lat'] == 5
            # assert interp._tolerance['other'] is None

        def test_none_tolerance(self):

            # test defaults
            interp = Interpolator('nearest', COORDINATES)
            assert interp._tolerance['lat'] == TOLERANCE_DEFAULTS['lat']
            assert interp._tolerance['lon'] == TOLERANCE_DEFAULTS['lon']

            # coords = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=['lat', 'other'])
            # interp = Interpolator('nearest', coords)
            # assert interp._tolerance['lat'] == TOLERANCE_DEFAULTS['lat']
            # assert interp._tolerance['other'] is None
