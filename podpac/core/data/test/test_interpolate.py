"""
Test interpolation methods
"""
# pylint: disable=C0111,W0212

import pytest
from traitlets import TraitError

from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.data import interpolate
from podpac.core.data.interpolate import (Interpolation, InterpolationException,
                                          Interpolator, INTERPOLATION_METHODS, INTERPOLATION_DEFAULT,
                                          INTERPOLATION_SHORTCUTS, NearestNeighbor, NearestPreview,
                                          Rasterio)

COORDINATES = Coordinates([clinspace(-25, 25, 11), clinspace(-25, 25, 11)], dims=['lat', 'lon'])
STACKED_COORDINATES = Coordinates([([1, 2, 3, 4, 5], [0, 1, 2, 3, 4])], dims=['lat_lon'])

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
            assert interp._config[('default',)]
            assert isinstance(interp._config[('default',)], dict)
            assert interp._config[('default',)]['method'] == 'nearest'
            assert isinstance(interp._config[('default',)]['interpolators'][0], Interpolator)

        def test_dict_definition(self):

            # should throw an error on _parse_interpolation_method(definition)
            # if definition is not in INTERPOLATION_SHORTCUTS
            with pytest.raises(InterpolationException):
                Interpolation({('lat', 'lon'): 'test'})

            # handle string methods
            interp = Interpolation({('lat', 'lon'): 'nearest'})
            assert isinstance(interp._config[('lat', 'lon')], dict)
            assert interp._config[('lat', 'lon')]['method'] == 'nearest'
            assert isinstance(interp._config[('default',)]['interpolators'][0], Interpolator)
            assert interp._config[('default',)]['params'] == {}

            # handle dict methods
            

            # should throw an error if method is not in dict
            with pytest.raises(InterpolationException):
                Interpolation({
                    ('lat', 'lon'): {
                        'test': 'test'
                    }
                })

            # should throw an error if method is not a string
            with pytest.raises(InterpolationException):
                Interpolation({
                    ('lat', 'lon'): {
                        'method': 5
                    }
                })

            # should throw an error if method is not one of the INTERPOLATION_SHORTCUTS and no interpolators defined
            with pytest.raises(InterpolationException):
                Interpolation({
                    ('lat', 'lon'): {
                        'method': 'myinter'
                    }
                })

            # should throw an error if params is not a dict
            with pytest.raises(TypeError):
                Interpolation({
                    ('lat', 'lon'): {
                        'method': 'nearest',
                        'params': 'test'
                    }
                })

            # should throw an error if interpolators is not a list
            with pytest.raises(TypeError):
                Interpolation({
                    ('lat', 'lon'): {
                        'method': 'nearest',
                        'interpolators': 'test'
                    }
                })

            # should throw an error if interpolators are not Interpolator classes
            with pytest.raises(TypeError):
                Interpolation({
                    ('lat', 'lon'): {
                        'method': 'nearest',
                        'interpolators':  [NearestNeighbor, 'test']
                    }
                })

            # should throw an error if dimension is defined twice
            with pytest.raises(InterpolationException):
                Interpolation({
                    ('lat', 'lon'): 'nearest',
                    'lat': 'bilinear'
                })


            # should handle standard INTEPROLATION_SHORTCUTS
            interp = Interpolation({
                ('lat', 'lon'): {
                    'method': 'nearest'
                }
            })
            assert isinstance(interp._config[('lat', 'lon')], dict)
            assert interp._config[('lat', 'lon')]['method'] == 'nearest'
            assert isinstance(interp._config[('lat', 'lon')]['interpolators'][0], Interpolator)
            assert interp._config[('lat', 'lon')]['params'] == {}
            

        
            # should allow custom methods if interpolators are defined
            interp = Interpolation({
                ('lat', 'lon'): {
                    'method': 'myinter',
                    'interpolators': [NearestNeighbor, NearestPreview]
                }
            })
            assert interp._config[('lat', 'lon')]['method'] == 'myinter'
            assert isinstance(interp._config[('lat', 'lon')]['interpolators'][0], NearestNeighbor)

            # should allow params to be set
            interp = Interpolation({
                ('lat', 'lon'): {
                    'method': 'myinter',
                    'interpolators': [NearestNeighbor, NearestPreview],
                    'params': {
                        'tolerance': 5
                    }
                }
            })
            assert interp._config[('lat', 'lon')]['params'] == {'tolerance': 5}

            # set default equal to empty tuple
            interp = Interpolation({'lat': 'bilinear'})
            assert interp._config[('default',)]['method'] == INTERPOLATION_DEFAULT


            # use default with override if not all dimensions are supplied
            interp = Interpolation({'lat': 'bilinear', 'default': 'optimal'})
            assert interp._config[('default',)]['method'] == 'optimal'


        def test_interpolator_init(self):

            interp = Interpolation('nearest')
            assert interp._config[('default',)]['interpolators'][0].method == 'nearest'


        def test_init_interpolators(self):

            # Interpolation init should init all interpolators in the list
            interp = Interpolation({
                'default': {
                    'method': 'nearest',
                    'params': {
                        'tolerance':1
                    }
                }
            })
            assert interp._config[('default',)]['interpolators'][0].tolerance == 1

            # should throw TraitErrors defined by Interpolator
            with pytest.raises(TraitError):
                Interpolation({
                    'default': {
                        'method': 'nearest',
                        'params': {
                            'tolerance':'tol'
                        }
                    }
                })

            # should not allow undefined params
            interp = Interpolation({
                'default': {
                    'method': 'nearest',
                    'params': {
                        'myarg':1
                    }
                }
            })
            with pytest.raises(AttributeError):
                assert interp._config[('default',)]['interpolators'][0].myarg == 'tol'
