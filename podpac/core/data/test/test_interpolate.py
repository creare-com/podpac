"""
Test interpolation methods


"""
# pylint: disable=C0111,W0212,R0903

from collections import OrderedDict

import pytest
from traitlets import TraitError
import numpy as np

from podpac.core.units import UnitsDataArray
from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.data.types import rasterio
from podpac.core.data import datasource
from podpac.core.data import interpolate
from podpac.core.data.interpolate import (Interpolation, InterpolationException,
                                          Interpolator, INTERPOLATION_METHODS, INTERPOLATION_DEFAULT,
                                          INTERPOLATION_SHORTCUTS, NearestNeighbor, NearestPreview,
                                          Rasterio)


# test fixtures
from podpac.core.data.test.test_datasource import MockArrayDataSource


class TestInterpolation(object):
    """ Test interpolation class and support methods"""

    def test_interpolate_module(self):
        """smoke test interpolate module"""

        assert interpolate is not None

    def test_allow_missing_modules(self):
        """TODO: Allow user to be missing rasterio and scipy"""
        pass

    def test_interpolation_methods(self):
        assert INTERPOLATION_SHORTCUTS == INTERPOLATION_METHODS.keys()

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

        # should handle a default definition without any dimensions specified as keys
        interp = Interpolation({
            'method': 'nearest',
            'params': {
                'space_tolerance': 1
            }
        })
        assert isinstance(interp._config[('default',)], dict)
        assert interp._config[('default',)]['method'] == 'nearest'
        assert isinstance(interp._config[('default',)]['interpolators'][0], Interpolator)
        assert interp._config[('default',)]['params'] == {'space_tolerance': 1}

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
                    'space_tolerance': 5
                }
            }
        })
        assert interp._config[('lat', 'lon')]['params'] == {'space_tolerance': 5}

        # set default equal to empty tuple
        interp = Interpolation({'lat': 'bilinear'})
        assert interp._config[('default',)]['method'] == INTERPOLATION_DEFAULT


        # use default with override if not all dimensions are supplied
        interp = Interpolation({'lat': 'bilinear', 'default': 'optimal'})
        assert interp._config[('default',)]['method'] == 'optimal'

        # make sure default is always the last key in the ordered config dict
        interp = Interpolation({'default': 'optimal', 'lat': 'bilinear'})
        assert list(interp._config.keys())[-1] == ('default',)

    def test_init_interpolators(self):

        # should set method
        interp = Interpolation('nearest')
        assert interp._config[('default',)]['interpolators'][0].method == 'nearest'

        # Interpolation init should init all interpolators in the list
        interp = Interpolation({
            'default': {
                'method': 'nearest',
                'params': {
                    'space_tolerance':1
                }
            }
        })
        assert interp._config[('default',)]['interpolators'][0].space_tolerance == 1

        # should throw TraitErrors defined by Interpolator
        with pytest.raises(TraitError):
            Interpolation({
                'default': {
                    'method': 'nearest',
                    'params': {
                        'space_tolerance':'tol'
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

    def test_select_interpolator_queue(self):

        reqcoords = Coordinates([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dims=['lat', 'lon', 'time', 'alt'])
        srccoords = Coordinates([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dims=['lat', 'lon', 'time', 'alt'])

        # create a few dummy interpolators that handle certain dimensions
        # (can_select is defined by default to look at dims_supported)
        class TimeLat(Interpolator):
            dims_supported = ['time', 'lat']

        class LatLon(Interpolator):
            dims_supported = ['lat', 'lon']
            
        class Lon(Interpolator):
            dims_supported = ['lon']


        # set up a strange interpolation definition
        # we want to interpolate (lat, lon) first, then after (time, alt)
        interp = Interpolation({
            ('lat', 'lon'): {
                'method': 'myinterp',
                'interpolators': [LatLon, TimeLat]
            },
            ('time', 'alt'): {
                'method': 'myinterp',
                'interpolators': [TimeLat, Lon]
            }
        })

        # default = Nearest, which always returns () for can_select
        interpolator_queue = interp._select_interpolator_queue(srccoords, reqcoords, 'can_select')
        assert isinstance(interpolator_queue, OrderedDict)
        assert isinstance(interpolator_queue[('lat', 'lon')], LatLon)
        assert ('time', 'alt') not in interpolator_queue

        # should throw an error if strict is set and not all dimensions can be handled
        with pytest.raises(InterpolationException):
            interpolator_queue = interp._select_interpolator_queue(srccoords, reqcoords, 'can_select', strict=True)

        # default = Nearest, which can handle all dims for can_interpolate
        interpolator_queue = interp._select_interpolator_queue(srccoords, reqcoords, 'can_interpolate')
        assert isinstance(interpolator_queue, OrderedDict)
        assert isinstance(interpolator_queue[('lat', 'lon')], LatLon)
        if ('alt', 'time') in interpolator_queue:
            assert isinstance(interpolator_queue[('alt', 'time')], NearestNeighbor)
        else:
            assert isinstance(interpolator_queue[('time', 'alt')], NearestNeighbor)

    def test_select_coordinates(self):

        reqcoords = Coordinates([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dims=['lat', 'lon', 'time', 'alt'])
        srccoords = Coordinates([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dims=['lat', 'lon', 'time', 'alt'])

        # create a few dummy interpolators that handle certain dimensions
        # (can_select is defined by default to look at dims_supported)
        class TimeLat(Interpolator):
            dims_supported = ['time', 'lat']

            def select_coordinates(self, udims, srccoords, srccoords_idx, reqcoords):
                return srccoords, srccoords_idx

        class LatLon(Interpolator):
            dims_supported = ['lat', 'lon']

            def select_coordinates(self, udims, srccoords, srccoords_idx, reqcoords):
                return srccoords, srccoords_idx

        class Lon(Interpolator):
            dims_supported = ['lon']

            def select_coordinates(self, udims, srccoords, srccoords_idx, reqcoords):
                return srccoords, srccoords_idx


        # set up a strange interpolation definition
        # we want to interpolate (lat, lon) first, then after (time, alt)
        interp = Interpolation({
            ('lat', 'lon'): {
                'method': 'myinterp',
                'interpolators': [LatLon, TimeLat]
            },
            ('time', 'alt'): {
                'method': 'myinterp',
                'interpolators': [TimeLat, Lon]
            }
        })

        coords, cidx = interp.select_coordinates(srccoords, [], reqcoords)

        assert len(coords) == len(srccoords)
        assert len(coords['lat']) == len(srccoords['lat'])
        assert cidx == []

    def test_interpolate(self):

        class TestInterp(Interpolator):
            dims_supported = ['lat', 'lon']
            def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
                output_data = source_data
                return source_coordinates, source_data, output_data

        # test basic functionality
        reqcoords = Coordinates([[-.5, 1.5, 3.5], [.5, 2.5, 4.5]], dims=['lat', 'lon'])
        srccoords = Coordinates([[0, 2, 4], [0, 3, 4]], dims=['lat', 'lon'])
        srcdata = UnitsDataArray(np.random.rand(3, 3),
                                 coords=[srccoords[c].coordinates for c in srccoords],
                                 dims=srccoords.dims)
        outdata = UnitsDataArray(np.zeros(srcdata.shape),
                                 coords=[reqcoords[c].coordinates for c in reqcoords],
                                 dims=reqcoords.dims)

        interp = Interpolation({('lat', 'lon'): {'method': 'test', 'interpolators': [TestInterp]}})
        srccoords, srcdata, outdata = interp.interpolate(srccoords, srcdata, reqcoords, outdata)

        assert np.all(outdata == srcdata)

        # test if data is size 1
        class TestFakeInterp(Interpolator):
            dims_supported = ['lat']
            def interpolate(self, udims, source_coordinates, source_data, eval_coordinates, output_data):
                return None
                
        reqcoords = Coordinates([[1]], dims=['lat'])
        srccoords = Coordinates([[1]], dims=['lat'])
        srcdata = UnitsDataArray(np.random.rand(1),
                                 coords=[srccoords[c].coordinates for c in srccoords],
                                 dims=srccoords.dims)
        outdata = UnitsDataArray(np.zeros(srcdata.shape),
                                 coords=[reqcoords[c].coordinates for c in reqcoords],
                                 dims=reqcoords.dims)

        interp = Interpolation({('lat', 'lon'): {'method': 'test', 'interpolators': [TestFakeInterp]}})
        srccoords, srcdata, outdata = interp.interpolate(srccoords, srcdata, reqcoords, outdata)

        assert np.all(outdata == srcdata)



class TestInterpolators(object):
    
    class TestInterpolator(object):
        """Test abstract interpolator class"""
        
        def test_can_select(self):

            class NoDimsSupported(Interpolator):
                method = 'mymthod'


            class DimsSupported(Interpolator):
                dims_supported = ['time', 'lat']

            class CanAlwaysSelect(Interpolator):

                def can_select(self, udims, reqcoords, srccoords):
                    return udims
            
            class CanNeverSelect(Interpolator):

                def can_select(self, udims, reqcoords, srccoords):
                    return tuple()

            interp = DimsSupported(method='method')
            can_select = interp.can_select(('time', 'lat'),  None, None)
            assert 'lat' in can_select and 'time' in can_select

            with pytest.raises(NotImplementedError):
                interp = NoDimsSupported(method='method')
                can_select = interp.can_select(('lat'), None, None)

            interp = CanAlwaysSelect(method='method')
            can_select = interp.can_select(('time', 'lat'), None, None)
            assert 'lat' in can_select and 'time' in can_select

            interp = CanNeverSelect(method='method')
            can_select = interp.can_select(('time', 'lat'), None, None)
            assert not can_select

    class TestNearest(object):

        def test_nearest_preview_select(self):
            
            # test straight ahead functionality
            reqcoords = Coordinates([[-.5, 1.5, 3.5], [.5, 2.5, 4.5]], dims=['lat', 'lon'])
            srccoords = Coordinates([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dims=['lat', 'lon'])

            interp = Interpolation('nearest_preview')

            srccoords, srccoords_index = srccoords.intersect(reqcoords, outer=True, return_indices=True)
            coords, cidx = interp.select_coordinates(srccoords, srccoords_index, reqcoords)

            assert len(coords) == len(srccoords) == len(cidx)
            assert len(coords['lat']) == len(reqcoords['lat'])
            assert len(coords['lon']) == len(reqcoords['lon'])
            assert np.all(coords['lat'].coordinates == np.array([0, 2, 4]))


            # test when selection is applied serially
            # this is equivalent to above
            reqcoords = Coordinates([[-.5, 1.5, 3.5], [.5, 2.5, 4.5]], dims=['lat', 'lon'])
            srccoords = Coordinates([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dims=['lat', 'lon'])

            interp = Interpolation({
                'lat': 'nearest_preview',
                'lon': 'nearest_preview'
            })

            srccoords, srccoords_index = srccoords.intersect(reqcoords, outer=True, return_indices=True)
            coords, cidx = interp.select_coordinates(srccoords, srccoords_index, reqcoords)

            assert len(coords) == len(srccoords) == len(cidx)
            assert len(coords['lat']) == len(reqcoords['lat'])
            assert len(coords['lon']) == len(reqcoords['lon'])
            assert np.all(coords['lat'].coordinates == np.array([0, 2, 4]))


            # test when coordinates are stacked and unstacked
            # TODO: how to handle stacked/unstacked coordinate asynchrony?
            # reqcoords = Coordinates([[-.5, 1.5, 3.5], [.5, 2.5, 4.5]], dims=['lat', 'lon'])
            # srccoords = Coordinates([([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])], dims=['lat_lon'])

            # interp = Interpolation('nearest_preview')

            # srccoords, srccoords_index = srccoords.intersect(reqcoords, outer=True, return_indices=True)
            # coords, cidx = interp.select_coordinates(reqcoords, srccoords, srccoords_index)

            # assert len(coords) == len(srcoords) == len(cidx)
            # assert len(coords['lat']) == len(reqcoords['lat'])
            # assert len(coords['lon']) == len(reqcoords['lon'])
            # assert np.all(coords['lat'].coordinates == np.array([0, 2, 4]))

        def test_interpolation(self):

            for interpolation in ['nearest', 'nearest_preview']:

                # unstacked 1D
                source = np.random.rand(5)
                coords_src = Coordinates([np.linspace(0, 10, 5,)], dims=['lat'])
                node = MockArrayDataSource(source=source, native_coordinates=coords_src, interpolation=interpolation)

                coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9]], dims=['lat'])
                output = node.eval(coords_dst)

                assert isinstance(output, UnitsDataArray)
                assert np.all(output.lat.values == coords_dst.coords['lat'])
                assert output.values[0] == source[0] and \
                       output.values[1] == source[0] and \
                       output.values[2] == source[1]

                # unstacked N-D
                source = np.random.rand(5, 5)
                coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
                coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=['lat', 'lon'])

                node = MockArrayDataSource(source=source, native_coordinates=coords_src, interpolation=interpolation)
                output = node.eval(coords_dst)

                assert isinstance(output, UnitsDataArray)
                assert np.all(output.lat.values == coords_dst.coords['lat'])
                assert output.values[0, 0] == source[1, 1]

                # stacked
                source = np.random.rand(5)
                coords_src = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=['lat_lon'])
                node = MockArrayDataSource(source=source, native_coordinates=coords_src, interpolation=interpolation)

                coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=['lat_lon'])
                
                with pytest.raises(InterpolationException):
                    node.eval(coords_dst)

                # source = stacked, dest = unstacked
                source = np.random.rand(5)
                coords_src = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=['lat_lon'])
                node = MockArrayDataSource(source=source, native_coordinates=coords_src, interpolation=interpolation)

                coords_dst = Coordinates([np.linspace(1, 9, 3), np.linspace(1, 9, 3)], dims=['lat', 'lon'])

                with pytest.raises(InterpolationException):
                    node.eval(coords_dst)

                # source = unstacked, dest = stacked
                source = np.random.rand(5, 5)
                coords_src = Coordinates([np.linspace(0, 10, 5), np.linspace(0, 10, 5)], dims=['lat', 'lon'])
                node = MockArrayDataSource(source=source, native_coordinates=coords_src, interpolation=interpolation)

                coords_dst = Coordinates([(np.linspace(1, 9, 3), np.linspace(1, 9, 3))], dims=['lat_lon'])

                with pytest.raises(InterpolationException):
                    node.eval(coords_dst)

        def test_space_tolerance(self):

            # unstacked 1D
            source = np.random.rand(5)
            coords_src = Coordinates([np.linspace(0, 10, 5)], dims=['lat'])
            node = MockArrayDataSource(source=source, native_coordinates=coords_src, interpolation={
                'default': {
                    'method': 'nearest',
                    'params': {
                        'space_tolerance':1.1
                    }
                }
            })

            coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9]], dims=['lat'])
            output = node.eval(coords_dst)

            print(output)
            print(source)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert output.values[0] == source[0] and \
                   np.isnan(output.values[1]) and \
                   output.values[2] == source[1]

        def test_time_tolerance(self):

            # unstacked 1D
            source = np.random.rand(5, 5)
            coords_src = Coordinates([np.linspace(0, 10, 5), clinspace('2018-01-01', '2018-01-09', 5)], 
                                     dims=['lat', 'time'])
            node = MockArrayDataSource(source=source, native_coordinates=coords_src, interpolation={
                'default': {
                    'method': 'nearest',
                    'params': {
                        'space_tolerance': 1.1,
                        'time_tolerance': np.timedelta64(1, 'D')
                    }
                }
            })

            coords_dst = Coordinates([[1, 1.2, 1.5, 5, 9], clinspace('2018-01-01', '2018-01-09', 3)], 
                                     dims=['lat', 'time'])
            output = node.eval(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert output.values[0, 0] == source[0, 0] and output.values[0, 1] == source[0, 2] and \
                   np.isnan(output.values[1, 0]) and np.isnan(output.values[1, 1]) and \
                   output.values[2, 0] == source[1, 0] and  output.values[2, 1] == source[1, 2]

    @pytest.mark.skip('for now')
    class TestInterpolateRasterio(object):
        """test interpolation functions"""

        def test_interpolate_rasterio(self):
            """ regular interpolation using rasterio"""

            assert rasterio is not None

            rasterio_interps = ['nearest', 'bilinear', 'cubic', 'cubic_spline',
                                'lanczos', 'average', 'mode', 'max', 'min',
                                'med', 'q1', 'q3']
            source = np.random.rand(5, 5)
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=['lat', 'lon'])

            node = MockArrayDataSource(source=source, native_coordinates=coords_src)

            # make sure it raises trait error
            with pytest.raises(TraitError):
                node.interpolation = 'myowninterp'
                output = node.eval(coords_dst)

            # make sure rasterio_interpolation method requires lat and lon
            # with pytest.raises(ValueError):
            #     coords_not_lon = Coordinates([clinspace(0, 10, 5)], dims=['lat'])
            #     node = MockArrayDataSource(source=source, native_coordinates=coords_not_lon)
            #     node.rasterio_interpolation(node, coords_src, coords_dst)

            # try all other interp methods
            for interp in rasterio_interps:
                node.interpolation = interp
                print(interp)
                output = node.eval(coords_dst)

                assert isinstance(output, UnitsDataArray)
                assert np.all(output.lat.values == coords_dst.coords['lat'])


        def test_interpolate_rasterio_descending(self):
            """should handle descending"""

            source = np.random.rand(5, 5)
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=['lat', 'lon'])
            
            node = MockArrayDataSource(source=source, native_coordinates=coords_src)
            output = node.eval(coords_dst)
            
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert np.all(output.lon.values == coords_dst.coords['lon'])


    @pytest.mark.skip('for now')
    class TestInterpolateNoRasterio(object):
        """test interpolation functions"""


        def test_interpolate_irregular_arbitrary(self):
            """ irregular interpolation """

            # suppress module to force statement
            datasource.rasterio = None

            rasterio_interps = ['nearest', 'bilinear', 'cubic', 'cubic_spline',
                                'lanczos', 'average', 'mode', 'max', 'min',
                                'med', 'q1', 'q3']
            source = np.random.rand(5, 5)
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=['lat', 'lon'])

            node = MockArrayDataSource(source=source, native_coordinates=coords_src)

            for interp in rasterio_interps:
                node.interpolation = interp
                print(interp)
                output = node.eval(coords_dst)

                assert isinstance(output, UnitsDataArray)
                assert np.all(output.lat.values == coords_dst.coords['lat'])

        def test_interpolate_irregular_arbitrary_2dims(self):
            """ irregular interpolation """

            datasource.rasterio = None

            # try >2 dims
            source = np.random.rand(5, 5, 3)
            coords_src = Coordinates(
                [clinspace(0, 10, 5), clinspace(0, 10, 5), [2, 3, 5]], dims=['lat', 'lon', 'time'])
            coords_dst = Coordinates(
                [clinspace(2, 12, 5), clinspace(2, 12, 5), [2, 3, 5]], dims=['lat', 'lon', 'time'])
            
            node = MockArrayDataSource(source=source, native_coordinates=coords_src)
            node.interpolation = 'nearest'
            output = node.eval(coords_dst)
            
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert np.all(output.lon.values == coords_dst.coords['lon'])
            assert np.all(output.time.values == coords_dst.coords['time'])

        def test_interpolate_irregular_arbitrary_descending(self):
            """should handle descending"""

            datasource.rasterio = None

            source = np.random.rand(5, 5)
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=['lat', 'lon'])
            
            node = MockArrayDataSource(source=source, native_coordinates=coords_src)
            node.interpolation = 'nearest'
            output = node.eval(coords_dst)
            
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert np.all(output.lon.values == coords_dst.coords['lon'])

        def test_interpolate_irregular_arbitrary_swap(self):
            """should handle descending"""

            datasource.rasterio = None

            source = np.random.rand(5, 5)
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
            coords_dst = Coordinates([clinspace(2, 12, 5), clinspace(2, 12, 5)], dims=['lat', 'lon'])
            
            node = MockArrayDataSource(source=source, native_coordinates=coords_src)
            node.interpolation = 'nearest'
            output = node.eval(coords_dst)
            
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert np.all(output.lon.values == coords_dst.coords['lon'])

        def test_interpolate_irregular_lat_lon(self):
            """ irregular interpolation """

            datasource.rasterio = None

            source = np.random.rand(5, 5)
            coords_src = Coordinates([clinspace(0, 10, 5), clinspace(0, 10, 5)], dims=['lat', 'lon'])
            coords_dst = Coordinates([[[0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]]], dims=['lat_lon'])

            node = MockArrayDataSource(source=source, native_coordinates=coords_src)
            node.interpolation = 'nearest'
            output = node.eval(coords_dst)

            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat_lon.values == coords_dst.coords['lat_lon'])
            assert output.values[0] == source[0, 0]
            assert output.values[1] == source[1, 1]
            assert output.values[-1] == source[-1, -1]

        def test_interpolate_point(self):
            """ interpolate point data to nearest neighbor with various coords_dst"""

            datasource.rasterio = None

            source = np.random.rand(6)
            coords_src = Coordinates([[[0, 2, 4, 6, 8, 10], [0, 2, 4, 5, 6, 10]]], dims=['lat_lon'])
            coords_dst = Coordinates([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]], dims=['lat_lon'])
            node = MockArrayDataSource(source=source, native_coordinates=coords_src)

            output = node.eval(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat_lon.values == coords_dst.coords['lat_lon'])
            assert output.values[0] == source[0]
            assert output.values[-1] == source[3]


            coords_dst = Coordinates([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dims=['lat', 'lon'])
            output = node.eval(coords_dst)
            assert isinstance(output, UnitsDataArray)
            assert np.all(output.lat.values == coords_dst.coords['lat'])
            assert output.values[0, 0] == source[0]
            assert output.values[-1, -1] == source[3]
