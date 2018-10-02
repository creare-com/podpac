"""
Test interpolation methods
"""

import pytest
from traitlets import TraitError

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.data import interpolate
from podpac.core.data.interpolate import (Interpolation, InterpolationException,
                                          Interpolator, INTERPOLATION_METHODS, INTERPOLATION_DEFAULT,
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


    class TestInterpolation(object):

        """ Test Interpolation class
        """
            

        def test_interpolator_init_type(self):
            """test constructor
            """

            # should throw an error if definition is not str, dict, or Interpolator
            with pytest.raises(TypeError):
                Interpolation(5)

        
        def test_str_definition(self):
            # should throw an error if string input is not one of the INTERPOLATION_SHORTCUTS
            with pytest.raises(InterpolationException):
                Interpolation('test')

            interp = Interpolation('nearest')
            assert interp._config[()]
            assert isinstance(interp._config[()], tuple)
            assert interp._config[()][0] == 'nearest'
            assert isinstance(interp._config[()][1][0], Interpolator)

        def test_dict_definition(self):

            # should throw an error on _parse_interpolation_method(definition)
            # if definition is not in INTERPOLATION_SHORTCUTS
            with pytest.raises(InterpolationException):
                Interpolation({'lat': 'test'})

            # handle string methods
            interp = Interpolation({('lat', 'lon'): 'nearest'})
            assert isinstance(interp._config[('lat', 'lon')], tuple)
            assert interp._config[('lat', 'lon')][0] == 'nearest'
            assert isinstance(interp._config[('lat', 'lon')][1][0], Interpolator)

            # handle tuple methods
            interp = Interpolation({('lat', 'lon'): ('nearest', [NearestNeighbor])})
            assert isinstance(interp._config[('lat', 'lon')], tuple)
            assert interp._config[('lat', 'lon')][0] == 'nearest'
            assert isinstance(interp._config[('lat', 'lon')][1][0], Interpolator)

            
            # use default if not all dimensions are supplied
            interp = Interpolation({'lat': 'bilinear'})
            assert interp._config[()][0] == INTERPOLATION_DEFAULT


            # use default with override if not all dimensions are supplied
            interp = Interpolation({'lat': 'bilinear'}, default='optimal')
            assert interp._config[()][0] == 'optimal'


        def test_tuple_definition(self):

            interp_tuple = ('myinterp', [NearestNeighbor, Rasterio, NearestPreview])
            interp = Interpolation(interp_tuple)
            assert interp._config[()]
            assert isinstance(interp._config[()], tuple)
            assert isinstance(interp._config[()][1][1], Rasterio)


            # should throw an error if items are not in a list
            with pytest.raises(TypeError):
                Interpolation(('myinter', 'test'))


            # should throw an error if items are not Interpolators
            with pytest.raises(TypeError):
                Interpolation(('myinter', ['test']))

            # should throw an error if method is not a string are not Interpolators
            with pytest.raises(TypeError):
                Interpolation((5, [NearestNeighbor, Rasterio, NearestPreview]))

        def test_interpolator_init(self):

            interp = Interpolation('nearest')
            assert interp._config[()][1][0].method == 'nearest'

        def test_init_kwargs(self):
            
            interp = Interpolation('nearest', tolerance=1)
            assert interp._config[()][1][0].tolerance == 1

            # should throw TraitErrors defined by Interpolator
            with pytest.raises(TraitError):
                Interpolation('nearest', tolerance='tol')

            # should allow other properties, but it won't put them on the 
            interp = Interpolation('nearest', myarg='tol')
            with pytest.raises(AttributeError):
                assert interp._config[()][1][0].myarg == 'tol'


