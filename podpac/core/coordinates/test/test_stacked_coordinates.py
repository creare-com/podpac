
from datetime import datetime

import pytest
import traitlets as tl
import numpy as np
from numpy.testing import assert_equal

from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates

class TestStackedCoordinatesCreation(object):
    def test_init_Coordinates1d(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03'], name='time')
        c = StackedCoordinates([lat, lon, time])
        assert c.dims == ('lat', 'lon', 'time')
        assert c.udims == ('lat', 'lon', 'time')
        assert c.name == 'lat_lon_time'

        # un-named
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([10, 20, 30])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03'])
        c = StackedCoordinates([lat, lon, time])
        assert c.dims == (None, None, None)
        assert c.udims == (None, None, None)
        assert c.name == '?_?_?'

    def test_ctype(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03'], name='time', ctype='right')

        c1 = StackedCoordinates([lat, lon, time], ctype='point')

        # lat and lon ctype set by StackedCoordinates
        assert c1['lat'].ctype == 'point'
        assert c1['lon'].ctype == 'point'

        # but time is left by StackedCoordinates because it was already explicitly set
        assert c1['time'].ctype == 'right'

        # same for the original objects (they are not copied)
        assert lat.ctype == 'point'
        assert lon.ctype == 'point'
        assert time.ctype == 'right'

    def test_StackedCoordinates(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        c = StackedCoordinates([lat, lon])
        c2 = StackedCoordinates(c)

    def test_invalid_coords(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([0, 1, 2, 3], name='lon')
        c = ArrayCoordinates1d([0, 1, 2])

        with pytest.raises(TypeError, match="Unrecognized coords type"):
            StackedCoordinates({})

        with pytest.raises(ValueError, match="stacked coords must have at least 2 coords"):
            StackedCoordinates([lat])

        with pytest.raises(ValueError, match="mismatch size in stacked coords"):
            StackedCoordinates([lat, lon])

        with pytest.raises(ValueError, match="duplicate dimension name"):
            StackedCoordinates([lat, lat])

        # but duplicate None name is okay
        StackedCoordinates([c, c])

        # TODO
        # with pytest.raises(tl.TraitError):
        #     StackedCoordinates([[0, 1, 2], [10, 20, 30]])

    def test_from_xarray(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03'], name='time')
        xcoords = StackedCoordinates([lat, lon, time]).coords

        c2 = StackedCoordinates.from_xarray(xcoords)
        assert c2.dims == ('lat', 'lon', 'time')
        assert_equal(c2['lat'].coordinates, lat.coordinates)
        assert_equal(c2['lon'].coordinates, lon.coordinates)
        assert_equal(c2['time'].coordinates, time.coordinates)

    def test_copy(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03'], name='time')
        c = StackedCoordinates([lat, lon, time])

        c2 = c.copy()
        assert c2.dims == c.dims
        assert_equal(c2['lat'].coordinates, c['lat'].coordinates)
        assert_equal(c2['lon'].coordinates, c['lon'].coordinates)
        assert_equal(c2['time'].coordinates, c['time'].coordinates)

        # set name
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([10, 20, 30])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03'])
        c = StackedCoordinates([lat, lon, time])

        c2 = c.copy(name='lat_lon_time')
        assert c2.dims == ('lat', 'lon', 'time')

        # set properties
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03'], name='time')
        c = StackedCoordinates([lat, lon, time])

        c2 = c.copy(ctype='point')
        c['lat'].ctype == 'midpoint'
        c['lon'].ctype == 'midpoint'
        c['time'].ctype == 'midpoint'
        c2['lat'].ctype == 'point'
        c2['lon'].ctype == 'point'
        c2['time'].ctype == 'point'

class TestStackedCoordinatesDefinition(object):
    def test_definition(self):
        lat = ArrayCoordinates1d([0, 1, 2], name='lat')
        lon = ArrayCoordinates1d([10, 20, 30], name='lon')
        time = UniformCoordinates1d('2018-01-01', '2018-01-03', '1,D', name='time')
        c = StackedCoordinates([lat, lon, time])
        d = c.definition
        
        assert isinstance(d, list)
        assert isinstance(d[0], dict)
        assert isinstance(d[1], dict)
        assert isinstance(d[2], dict)
        ArrayCoordinates1d.from_definition(d[0])
        ArrayCoordinates1d.from_definition(d[1])
        UniformCoordinates1d.from_definition(d[2])

        c2 = StackedCoordinates.from_definition(c.definition)
        assert c2.dims == c.dims
        assert_equal(c2['lat'].coordinates, c['lat'].coordinates)
        assert_equal(c2['lon'].coordinates, c['lon'].coordinates)
        assert_equal(c2['time'].coordinates, c['time'].coordinates)

    def test_from_definition(self):
        with pytest.raises(ValueError, match="Could not parse coordinates definition with keys"):
            StackedCoordinates.from_definition([{'apple': 10}, {}])

class TestStackedCoordinatesProperties(object):
    def test_set_name(self):
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([10, 20, 30])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03'])
        c = StackedCoordinates([lat, lon, time])
        c.name = 'lat_lon_time'

        assert c.dims == ('lat', 'lon', 'time')
        assert c.udims == ('lat', 'lon', 'time')
        assert c.name == 'lat_lon_time'
        
        # also sets the Coordinates1d objects:
        assert lat.name == 'lat'
        assert lon.name == 'lon'
        assert time.name == 'time'

    def test_size(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        c = StackedCoordinates([lat, lon, time])

        assert c.size == 4

    def test_coordinates(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        c = StackedCoordinates([lat, lon, time])

        # TODO

    def test_coords(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        c = StackedCoordinates([lat, lon, time])

        # TODO

class TestStackedCoordinatesIndexing(object):
    def test_get_dim(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        c = StackedCoordinates([lat, lon, time])

        # TODO

    def test_get_index(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        c = StackedCoordinates([lat, lon, time])

        # TODO

    def test_iter(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        c = StackedCoordinates([lat, lon, time])

        for item in c:
            assert isinstance(item, Coordinates1d)

    def test_len(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'])
        c = StackedCoordinates([lat, lon, time])

        assert len(c) == 3

class TestStackedCoordinatesSelection(object):
    def test_intersect(self):
        pass

    @pytest.mark.skip(reason="not yet implemented")
    def test_select(self):
        pass