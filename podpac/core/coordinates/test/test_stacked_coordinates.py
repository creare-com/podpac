from datetime import datetime
import json

import pytest
import traitlets as tl
import numpy as np
import pandas as pd
import xarray as xr
from numpy.testing import assert_equal

import podpac
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates


class TestStackedCoordinatesCreation(object):
    def test_init_explicit(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"], name="time")
        c = StackedCoordinates([lat, lon, time])
        assert c.dims == ("lat", "lon", "time")
        assert c.udims == ("lat", "lon", "time")
        assert c.name == "lat_lon_time"
        repr(c)

        # un-named
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([10, 20, 30])
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"])
        c = StackedCoordinates([lat, lon, time])
        assert c.dims == (None, None, None)
        assert c.udims == (None, None, None)
        assert c.name is None

        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        c = StackedCoordinates([lat, lon, time])
        assert c.dims == ("lat", None, None)
        assert c.udims == ("lat", None, None)
        assert c.name == "lat_?_?"

        repr(c)

    def test_init_explicit_shaped(self):
        lat = ArrayCoordinates1d([[0, 1, 2], [10, 11, 12]], name="lat")
        lon = ArrayCoordinates1d([[10, 20, 30], [11, 21, 31]], name="lon")
        c = StackedCoordinates([lat, lon])
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert c.name == "lat_lon"
        repr(c)

    def test_coercion_with_dims(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        c = StackedCoordinates([lat, lon], dims=["lat", "lon"])
        assert c.dims == ("lat", "lon")
        assert_equal(c["lat"].coordinates, lat)
        assert_equal(c["lon"].coordinates, lon)

    def test_coercion_with_name(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        c = StackedCoordinates([lat, lon], name="lat_lon")
        assert c.dims == ("lat", "lon")
        assert_equal(c["lat"].coordinates, lat)
        assert_equal(c["lon"].coordinates, lon)

    def test_coercion_shaped_with_dims(self):
        lat = [[0, 1, 2], [10, 11, 12]]
        lon = [[10, 20, 30], [11, 21, 31]]
        c = StackedCoordinates([lat, lon], dims=["lat", "lon"])
        assert c.dims == ("lat", "lon")
        assert_equal(c["lat"].coordinates, lat)
        assert_equal(c["lon"].coordinates, lon)

    def test_coercion_shaped_with_name(self):
        lat = [[0, 1, 2], [10, 11, 12]]
        lon = [[10, 20, 30], [11, 21, 31]]
        c = StackedCoordinates([lat, lon], name="lat_lon")
        assert c.dims == ("lat", "lon")
        assert_equal(c["lat"].coordinates, lat)
        assert_equal(c["lon"].coordinates, lon)

    def test_invalid_coords_type(self):
        with pytest.raises(TypeError, match="Unrecognized coords type"):
            StackedCoordinates({})

    def test_invalid_init_dims_and_name(self):
        with pytest.raises(TypeError):
            StackedCoordinates([[0, 1, 2], [10, 20, 30]], dims=["lat", "lon"], name="lat_lon")

    def test_duplicate_dims(self):
        with pytest.raises(ValueError, match="Duplicate dimension"):
            StackedCoordinates([[0, 1, 2], [10, 20, 30]], dims=["lat", "lat"])

        with pytest.raises(ValueError, match="Duplicate dimension"):
            StackedCoordinates([[0, 1, 2], [10, 20, 30]], name="lat_lat")

    def test_invalid_coords(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([0, 1, 2, 3], name="lon")
        c = ArrayCoordinates1d([0, 1, 2])

        with pytest.raises(ValueError, match="Stacked coords must have at least 2 coords"):
            StackedCoordinates([lat])

        with pytest.raises(ValueError, match="Shape mismatch in stacked coords"):
            StackedCoordinates([lat, lon])

        with pytest.raises(ValueError, match="Duplicate dimension"):
            StackedCoordinates([lat, lat])

        # (but duplicate None name is okay)
        StackedCoordinates([c, c])

    def test_invalid_coords_shaped(self):
        # same size, different shape
        lat = ArrayCoordinates1d(np.arange(12).reshape((3, 4)), name="lat")
        lon = ArrayCoordinates1d(np.arange(12).reshape((4, 3)), name="lon")

        with pytest.raises(ValueError, match="Shape mismatch in stacked coords"):
            StackedCoordinates([lat, lon])

    def test_from_xarray(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"], name="time")
        c = StackedCoordinates([lat, lon, time])
        x = xr.DataArray(np.empty(c.shape), coords=c.xcoords, dims=c.xdims)

        c2 = StackedCoordinates.from_xarray(x.coords)
        assert c2.dims == ("lat", "lon", "time")
        assert_equal(c2["lat"].coordinates, lat.coordinates)
        assert_equal(c2["lon"].coordinates, lon.coordinates)
        assert_equal(c2["time"].coordinates, time.coordinates)

    def test_copy(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"], name="time")
        c = StackedCoordinates([lat, lon, time])

        c2 = c.copy()
        assert c2 is not c
        assert c2 == c


class TestStackedCoordinatesEq(object):
    def test_eq_type(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        c = StackedCoordinates([lat, lon])
        assert c != [[0, 1, 2], [10, 20, 30]]

    def test_eq_size_shortcut(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        c1 = StackedCoordinates([lat, lon])
        c2 = StackedCoordinates([lat[:2], lon[:2]])
        assert c1 != c2

    def test_eq_dims_shortcut(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        c1 = StackedCoordinates([lat, lon])
        c2 = StackedCoordinates([lon, lat])
        assert c1 != c2

    def test_eq_coordinates(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        c1 = StackedCoordinates([lat, lon])
        c2 = StackedCoordinates([lat, lon])
        c3 = StackedCoordinates([lat[::-1], lon])
        c4 = StackedCoordinates([lat, lon[::-1]])

        assert c1 == c2
        assert c1 != c3
        assert c1 != c4

    def test_eq_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c1 = StackedCoordinates([lat, lon])
        c2 = StackedCoordinates([lat, lon])
        c3 = StackedCoordinates([lat[::-1], lon])
        c4 = StackedCoordinates([lat, lon[::-1]])

        assert c1 == c2
        assert c1 != c3
        assert c1 != c4


class TestStackedCoordinatesSerialization(object):
    def test_definition(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        time = UniformCoordinates1d("2018-01-01", "2018-01-03", "1,D", name="time")
        c = StackedCoordinates([lat, lon, time])
        d = c.definition

        assert isinstance(d, list)
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)  # test serializable
        c2 = StackedCoordinates.from_definition(d)
        assert c2 == c

    def test_invalid_definition(self):
        with pytest.raises(ValueError, match="Could not parse coordinates definition with keys"):
            StackedCoordinates.from_definition([{"apple": 10}, {}])

    def test_definition_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = StackedCoordinates([lat, lon], dims=["lat", "lon"])

        d = c.definition
        assert isinstance(d, list)
        assert len(d) == 2

        # serializable
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)

        # from definition
        c2 = StackedCoordinates.from_definition(d)
        assert c2 == c

    def test_full_definition_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = StackedCoordinates([lat, lon])
        d = c.full_definition
        assert isinstance(d, list)
        assert len(d) == 2

        # serializable
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)


class TestStackedCoordinatesProperties(object):
    def test_set_dims(self):
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([10, 20, 30])
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"])
        c = StackedCoordinates([lat, lon, time])
        c._set_dims(["lat", "lon", "time"])

        assert c.dims == ("lat", "lon", "time")
        assert lat.name == "lat"
        assert lon.name == "lon"
        assert time.name == "time"

        # some can already be set
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([10, 20, 30])
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"], name="time")
        c = StackedCoordinates([lat, lon, time])
        c._set_dims(["lat", "lon", "time"])

        assert c.dims == ("lat", "lon", "time")
        assert lat.name == "lat"
        assert lon.name == "lon"
        assert time.name == "time"

        # but they have to match
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"])
        c = StackedCoordinates([lat, lon, time])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            c._set_dims(["lon", "lat", "time"])

        # invalid dims
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([10, 20, 30])
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"])
        c = StackedCoordinates([lat, lon, time])
        with pytest.raises(ValueError, match="Invalid dims"):
            c._set_dims(["lat", "lon"])

    def test_set_name(self):
        # note: mostly tested by test_set_dims
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([10, 20, 30])
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"])
        c = StackedCoordinates([lat, lon, time])

        c._set_name("lat_lon_time")
        assert c.name == "lat_lon_time"

        # invalid
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([10, 20, 30])
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"])
        c = StackedCoordinates([lat, lon, time])
        with pytest.raises(ValueError, match="Invalid name"):
            c._set_name("lat_lon")

    def test_size(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"])
        c = StackedCoordinates([lat, lon, time])

        assert c.size == 4

    def test_shape(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"])
        c = StackedCoordinates([lat, lon, time])

        assert c.shape == (4,)

    def test_size_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = StackedCoordinates([lat, lon])
        assert c.size == 12

    def test_shape_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = StackedCoordinates([lat, lon])
        assert c.shape == (3, 4)

    def test_coordinates(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], name="time")
        c = StackedCoordinates([lat, lon, time])

        assert_equal(c.coordinates, np.array([lat.coordinates, lon.coordinates, time.coordinates]).T)
        assert c.coordinates.dtype == object

        # single dtype
        lat = ArrayCoordinates1d([0, 1, 2, 3], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        c = StackedCoordinates([lat, lon])

        assert_equal(c.coordinates, np.array([lat.coordinates, lon.coordinates]).T)
        assert c.coordinates.dtype == float

    def test_coordinates_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = StackedCoordinates([lat, lon])
        assert_equal(c.coordinates, np.array([lat.T, lon.T]).T)

    def test_xdims(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], name="time")
        c = StackedCoordinates([lat, lon, time])
        assert c.xdims == ("lat_lon_time",)

    def test_xdims_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = StackedCoordinates([lat, lon], dims=["lat", "lon"])
        assert len(set(c.xdims)) == 2

    def test_xcoords(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], name="time")
        c = StackedCoordinates([lat, lon, time])

        assert isinstance(c.xcoords, dict)
        x = xr.DataArray(np.empty(c.shape), dims=c.xdims, coords=c.xcoords)
        assert x.dims == ("lat_lon_time",)
        assert_equal(x.coords["lat"], c["lat"].coordinates)
        assert_equal(x.coords["lon"], c["lon"].coordinates)
        assert_equal(x.coords["time"], c["time"].coordinates)

        # unnamed
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"])
        c = StackedCoordinates([lat, lon, time])
        with pytest.raises(ValueError, match="Cannot get xcoords"):
            c.xcoords

    def test_xcoords_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = StackedCoordinates([lat, lon], dims=["lat", "lon"])

        assert isinstance(c.xcoords, dict)
        x = xr.DataArray(np.empty(c.shape), dims=c.xdims, coords=c.xcoords)
        assert_equal(x.coords["lat"], c["lat"].coordinates)
        assert_equal(x.coords["lon"], c["lon"].coordinates)

        c = StackedCoordinates([lat, lon])
        with pytest.raises(ValueError, match="Cannot get xcoords"):
            c.xcoords

    def test_bounds(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]

        c = StackedCoordinates([lat, lon], dims=["lat", "lon"])
        bounds = c.bounds
        assert isinstance(bounds, dict)
        assert set(bounds.keys()) == set(c.udims)
        assert_equal(bounds["lat"], c["lat"].bounds)
        assert_equal(bounds["lon"], c["lon"].bounds)

        c = StackedCoordinates([lat, lon])
        with pytest.raises(ValueError, match="Cannot get bounds"):
            c.bounds

    def test_bounds_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = StackedCoordinates([lat, lon], dims=["lat", "lon"])
        bounds = c.bounds
        assert isinstance(bounds, dict)
        assert set(bounds.keys()) == set(c.udims)
        assert_equal(bounds["lat"], c["lat"].bounds)
        assert_equal(bounds["lon"], c["lon"].bounds)

        c = StackedCoordinates([lat, lon])
        with pytest.raises(ValueError, match="Cannot get bounds"):
            c.bounds


class TestStackedCoordinatesIndexing(object):
    def test_get_dim(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], name="time")
        c = StackedCoordinates([lat, lon, time])

        assert c["lat"] is lat
        assert c["lon"] is lon
        assert c["time"] is time
        with pytest.raises(KeyError, match="Dimension 'other' not found in dims"):
            c["other"]

    def test_get_index(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], name="time")
        c = StackedCoordinates([lat, lon, time])

        # integer index
        I = 0
        cI = c[I]
        assert isinstance(cI, StackedCoordinates)
        assert cI.size == 1
        assert cI.dims == c.dims
        assert_equal(cI["lat"].coordinates, c["lat"].coordinates[I])

        # index array
        I = [1, 2]
        cI = c[I]
        assert isinstance(cI, StackedCoordinates)
        assert cI.size == 2
        assert cI.dims == c.dims
        assert_equal(cI["lat"].coordinates, c["lat"].coordinates[I])

        # boolean array
        I = [False, True, True, False]
        cI = c[I]
        assert isinstance(cI, StackedCoordinates)
        assert cI.size == 2
        assert cI.dims == c.dims
        assert_equal(cI["lat"].coordinates, c["lat"].coordinates[I])

        # slice
        cI = c[1:3]
        assert isinstance(cI, StackedCoordinates)
        assert cI.size == 2
        assert cI.dims == c.dims
        assert_equal(cI["lat"].coordinates, c["lat"].coordinates[1:3])

    def test_get_index_shaped(self):
        lat = np.linspace(0, 1, 60).reshape((5, 4, 3))
        lon = np.linspace(1, 2, 60).reshape((5, 4, 3))
        c = StackedCoordinates([lat, lon], dims=["lat", "lon"])

        I = [3, 1, 2]
        J = slice(1, 3)
        K = 1
        B = lat > 0.5

        # full
        c2 = c[I, J, K]
        assert isinstance(c2, StackedCoordinates)
        assert c2.shape == (3, 2)
        assert c2.dims == c.dims
        assert c2["lat"] == c["lat"][I, J, K]
        assert c2["lon"] == c["lon"][I, J, K]
        assert_equal(c2["lat"].coordinates, lat[I, J, K])
        assert_equal(c2["lon"].coordinates, lon[I, J, K])

        # partial/implicit
        c2 = c[I, J]
        assert isinstance(c2, StackedCoordinates)
        assert c2.shape == (3, 2, 3)
        assert c2.dims == c.dims
        assert c2["lat"] == c["lat"][I, J]
        assert c2["lon"] == c["lon"][I, J]
        assert_equal(c2["lat"].coordinates, lat[I, J])
        assert_equal(c2["lon"].coordinates, lon[I, J])

        # boolean
        c2 = c[B]
        assert isinstance(c2, StackedCoordinates)
        assert c2.shape == (30,)
        assert c2.dims == c.dims
        assert c2["lat"] == c["lat"][B]
        assert c2["lon"] == c["lon"][B]
        assert_equal(c2["lat"].coordinates, lat[B])
        assert_equal(c2["lon"].coordinates, lon[B])

    def test_iter(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"])
        c = StackedCoordinates([lat, lon, time])

        for item in c:
            assert isinstance(item, Coordinates1d)

    def test_len(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"])
        c = StackedCoordinates([lat, lon, time])

        assert len(c) == 3

    def test_in(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"])
        c = StackedCoordinates([lat, lon, time])

        assert (0, 10, "2018-01-01") in c
        assert (1, 10, "2018-01-01") not in c
        assert ("2018-01-01", 10, 0) not in c
        assert (0,) not in c
        assert "test" not in c

    def test_in_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = StackedCoordinates([lat, lon], dims=["lat", "lon"])

        assert (lat[0, 0], lon[0, 0]) in c
        assert (lat[0, 0], lon[0, 1]) not in c
        assert (lon[0, 0], lat[0, 0]) not in c
        assert lat[0, 0] not in c


class TestStackedCoordinatesSelection(object):
    def test_select_single(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], name="time")
        c = StackedCoordinates([lat, lon, time])

        # single dimension
        s = c.select({"lat": [0.5, 2.5]})
        assert s == c[1:3]

        s, I = c.select({"lat": [0.5, 2.5]}, return_index=True)
        assert s == c[I]
        assert s == c[1:3]

        # a different single dimension
        s = c.select({"lon": [5, 25]})
        assert s == c[0:2]

        s, I = c.select({"lon": [5, 25]}, return_index=True)
        assert s == c[I]
        assert s == c[0:2]

        # outer
        s = c.select({"lat": [0.5, 2.5]}, outer=True)
        assert s == c[0:4]

        s, I = c.select({"lat": [0.5, 2.5]}, outer=True, return_index=True)
        assert s == c[I]
        assert s == c[0:4]

        # no matching dimension
        s = c.select({"alt": [0, 10]})
        assert s == c

        s, I = c.select({"alt": [0, 10]}, return_index=True)
        assert s == c[I]
        assert s == c

    def test_select_multiple(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3, 4, 5], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40, 50, 60], name="lon")
        c = StackedCoordinates([lat, lon])

        # this should be the AND of both intersections
        slat = c.select({"lat": [0.5, 3.5]})
        slon = c.select({"lon": [25, 55]})
        s = c.select({"lat": [0.5, 3.5], "lon": [25, 55]})
        assert slat == c[1:4]
        assert slon == c[2:5]
        assert s == c[2:4]

        s, I = c.select({"lat": [0.5, 3.5], "lon": [25, 55]}, return_index=True)
        assert s == c[2:4]
        assert s == c[I]

    def test_select_single_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = StackedCoordinates([lat, lon], dims=["lat", "lon"])

        # single dimension
        bounds = {"lat": [0.25, 0.55]}
        E0, E1 = [0, 1, 1, 1], [3, 0, 1, 2]  # expected

        s = c.select(bounds)
        assert isinstance(s, StackedCoordinates)
        assert s == c[E0, E1]

        s, I = c.select(bounds, return_index=True)
        assert isinstance(s, StackedCoordinates)
        assert s == c[I]
        assert s == c[E0, E1]

        # a different single dimension
        bounds = {"lon": [12.5, 17.5]}
        E0, E1 = [0, 1, 1, 1, 1, 2], [3, 0, 1, 2, 3, 0]

        s = c.select(bounds)
        assert isinstance(s, StackedCoordinates)
        assert s == c[E0, E1]

        s, I = c.select(bounds, return_index=True)
        assert isinstance(s, StackedCoordinates)
        assert s == c[I]
        assert s == c[E0, E1]

        # outer
        bounds = {"lat": [0.25, 0.75]}
        E0, E1 = [0, 0, 1, 1, 1, 1, 2, 2], [2, 3, 0, 1, 2, 3, 0, 1]

        s = c.select(bounds, outer=True)
        assert isinstance(s, StackedCoordinates)
        assert s == c[E0, E1]

        s, I = c.select(bounds, outer=True, return_index=True)
        assert isinstance(s, StackedCoordinates)
        assert s == c[I]
        assert s == c[E0, E1]

        # no matching dimension
        bounds = {"alt": [0, 10]}
        s = c.select(bounds)
        assert s == c

        s, I = c.select(bounds, return_index=True)
        assert s == c[I]
        assert s == c

    def test_select_multiple_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = StackedCoordinates([lat, lon], dims=["lat", "lon"])

        # this should be the AND of both intersections
        bounds = {"lat": [0.25, 0.95], "lon": [10.5, 17.5]}
        E0, E1 = [0, 1, 1, 1, 1, 2], [3, 0, 1, 2, 3, 0]
        s = c.select(bounds)
        assert s == c[E0, E1]

        s, I = c.select(bounds, return_index=True)
        assert s == c[I]
        assert s == c[E0, E1]


class TestStackedCoordinatesMethods(object):
    def test_transpose(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"], name="time")
        c = StackedCoordinates([lat, lon, time])

        t = c.transpose("lon", "lat", "time")
        assert c.dims == ("lat", "lon", "time")
        assert t.dims == ("lon", "lat", "time")
        assert t["lat"] == lat
        assert t["lon"] == lon
        assert t["time"] == time

        # default transpose
        t = c.transpose()
        assert c.dims == ("lat", "lon", "time")
        assert t.dims == ("time", "lon", "lat")

    def test_transpose_invalid(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"], name="time")
        c = StackedCoordinates([lat, lon, time])

        with pytest.raises(ValueError, match="Invalid transpose dimensions"):
            c.transpose("lon", "lat")

    def test_transpose_in_place(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"], name="time")
        c = StackedCoordinates([lat, lon, time])

        t = c.transpose("lon", "lat", "time", in_place=False)
        assert c.dims == ("lat", "lon", "time")
        assert t.dims == ("lon", "lat", "time")

        c.transpose("lon", "lat", "time", in_place=True)
        assert c.dims == ("lon", "lat", "time")
        assert t["lat"] == lat
        assert t["lon"] == lon
        assert t["time"] == time

    def test_unique(self):
        lat = ArrayCoordinates1d([0, 1, 2, 1, 0, 5], name="lat")
        lon = ArrayCoordinates1d([10, 20, 20, 20, 10, 60], name="lon")
        c = StackedCoordinates([lat, lon])

        c2 = c.unique()
        assert_equal(c2["lat"].coordinates, [0, 1, 2, 5])
        assert_equal(c2["lon"].coordinates, [10, 20, 20, 60])

        c2, I = c.unique(return_index=True)
        assert_equal(c2["lat"].coordinates, [0, 1, 2, 5])
        assert_equal(c2["lon"].coordinates, [10, 20, 20, 60])
        assert c[I] == c2

    def test_unque_shaped(self):
        lat = ArrayCoordinates1d([[0, 1, 2], [1, 0, 5]], name="lat")
        lon = ArrayCoordinates1d([[10, 20, 20], [20, 10, 60]], name="lon")
        c = StackedCoordinates([lat, lon])

        # flattens
        c2 = c.unique()
        assert_equal(c2["lat"].coordinates, [0, 1, 2, 5])
        assert_equal(c2["lon"].coordinates, [10, 20, 20, 60])

        c2, I = c.unique(return_index=True)
        assert_equal(c2["lat"].coordinates, [0, 1, 2, 5])
        assert_equal(c2["lon"].coordinates, [10, 20, 20, 60])
        assert c.flatten()[I] == c2

    def test_get_area_bounds(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        c = StackedCoordinates([lat, lon])
        d = c.get_area_bounds({"lat": 0.5, "lon": 1})
        # this is just a pass through
        assert d["lat"] == lat.get_area_bounds(0.5)
        assert d["lon"] == lon.get_area_bounds(1)

        # has to be named
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([10, 20, 30])
        c = StackedCoordinates([lat, lon])
        with pytest.raises(ValueError, match="Cannot get area_bounds"):
            c.get_area_bounds({"lat": 0.5, "lon": 1})

    def test_issubset(self):
        lat = np.arange(4)
        lon = 10 * np.arange(4)
        time = 100 * np.arange(4)

        sc = StackedCoordinates([lat, lon], name="lat_lon")
        sc_2 = StackedCoordinates([lat + 100, lon], name="lat_lon")  # different coordinates
        sc_3 = StackedCoordinates([lat[::-1], lon], name="lat_lon")  # same coordinates, paired differently
        sc_t = sc.transpose("lon", "lat")
        sc_time = StackedCoordinates([lat, lon, time], name="lat_lon_time")

        assert sc.issubset(sc)
        assert sc[:2].issubset(sc)
        assert not sc.issubset(sc[:2])
        assert not sc_2.issubset(sc)
        assert not sc_3.issubset(sc)

        assert sc_t.issubset(sc)

        # extra/missing dimension
        assert not sc.issubset(sc_time)
        assert not sc_time.issubset(sc)

    def test_issubset_coordinates(self):
        lat = np.arange(4)
        lon = 10 * np.arange(4)
        time = 100 * np.arange(4)

        sc = StackedCoordinates([lat, lon], name="lat_lon")
        sc_2 = StackedCoordinates([lat + 100, lon], name="lat_lon")
        sc_3 = StackedCoordinates([lat[::-1], lon], name="lat_lon")
        sc_t = sc.transpose("lon", "lat")
        sc_time = StackedCoordinates([lat, lon, time], name="lat_lon_time")

        # coordinates with stacked lat_lon
        cs = podpac.Coordinates([[lat, lon]], dims=["lat_lon"])
        assert sc.issubset(cs)
        assert sc[:2].issubset(cs)
        assert sc[::-1].issubset(cs)
        assert not sc_2.issubset(cs)
        assert not sc_3.issubset(cs)
        assert sc_t.issubset(cs)
        assert not sc_time.issubset(cs)

        # coordinates with shaped stacked lat_lon
        cd = podpac.Coordinates([[lat.reshape((2, 2)), lon.reshape((2, 2))]], dims=["lat_lon"])
        assert sc.issubset(cd)
        assert sc[:2].issubset(cd)
        assert sc[::-1].issubset(cd)
        assert not sc_2.issubset(cd)
        assert not sc_3.issubset(cd)
        assert sc_t.issubset(cd)
        assert not sc_time.issubset(cd)

        # coordinates with unstacked lat, lon
        cu = podpac.Coordinates([lat, lon[::-1]], dims=["lat", "lon"])
        assert sc.issubset(cu)
        assert sc[:2].issubset(cu)
        assert sc[::-1].issubset(cu)
        assert not sc_2.issubset(cu)
        assert sc_3.issubset(cu)  # this is an important case!
        assert sc_t.issubset(cu)
        assert not sc_time.issubset(cu)

        # coordinates with unstacked lat, lon, time
        cu_time = podpac.Coordinates([lat, lon, time], dims=["lat", "lon", "time"])
        assert sc.issubset(cu_time)
        assert sc[:2].issubset(cu_time)
        assert sc[::-1].issubset(cu_time)
        assert not sc_2.issubset(cu_time)
        assert sc_3.issubset(cu_time)
        assert sc_t.issubset(cu_time)
        assert sc_time.issubset(cu_time)

        assert not sc.issubset(cu_time[:2, :, :])

        # mixed coordinates
        cmixed = podpac.Coordinates([[lat, lon], time], dims=["lat_lon", "time"])
        assert sc.issubset(cmixed)
        assert sc[:2].issubset(cmixed)
        assert sc[::-1].issubset(cmixed)
        assert not sc_2.issubset(cmixed)
        assert not sc_3.issubset(cmixed)
        assert sc_t.issubset(cmixed)
        assert sc_time.issubset(cmixed)  # this is the most general case

        assert not sc.issubset(cmixed[:2, :])
        assert not sc_time.issubset(cmixed[:, :1])

    def test_issubset_other(self):
        sc = StackedCoordinates([[1, 2, 3], [10, 20, 30]], name="lat_lon")

        with pytest.raises(TypeError, match="StackedCoordinates issubset expected Coordinates or StackedCoordinates"):
            sc.issubset([])

    def test_issubset_shaped(self):
        lat = np.arange(12).reshape(3, 4)
        lon = 10 * np.arange(12).reshape(3, 4)
        time = 100 * np.arange(12).reshape(3, 4)

        dc = StackedCoordinates([lat, lon], dims=["lat", "lon"])
        dc_2 = StackedCoordinates([lat + 100, lon], dims=["lat", "lon"])  # different coordinates
        dc_3 = StackedCoordinates([lat[::-1], lon], dims=["lat", "lon"])  # same coordinates, but paired differently
        dc_t = dc.transpose("lon", "lat")
        dc_shape = StackedCoordinates([lat.reshape(6, 2), lon.reshape(6, 2)], dims=["lat", "lon"])
        dc_time = StackedCoordinates([lat, lon, time], dims=["lat", "lon", "time"])

        assert dc.issubset(dc)
        assert dc[:2, :2].issubset(dc)
        assert not dc.issubset(dc[:2, :2])
        assert not dc_2.issubset(dc)
        assert not dc_3.issubset(dc)

        assert dc_t.issubset(dc)
        assert dc_shape.issubset(dc)

        # extra/missing dimension
        assert not dc_time.issubset(dc)
        assert not dc.issubset(dc_time)

    def test_issubset_coordinates_shaped(self):
        ulat = np.arange(12)
        ulon = 10 * np.arange(12)
        utime = 100 * np.arange(12)

        lat = ulat.reshape(3, 4)
        lon = ulon.reshape(3, 4)
        time = utime.reshape(3, 4)

        dc = StackedCoordinates([lat, lon], dims=["lat", "lon"])
        dc_2 = StackedCoordinates([lat + 100, lon], dims=["lat", "lon"])  # different coordinates
        dc_3 = StackedCoordinates([lat[::-1], lon], dims=["lat", "lon"])  # same coordinates, but paired differently
        dc_t = dc.transpose("lon", "lat")
        dc_shape = StackedCoordinates([lat.reshape(6, 2), lon.reshape(6, 2)], dims=["lat", "lon"])
        dc_time = StackedCoordinates([lat, lon, time], dims=["lat", "lon", "time"])

        # coordinates with stacked lat_lon
        cs = podpac.Coordinates([[ulat, ulon]], dims=["lat_lon"])
        assert dc.issubset(cs)
        assert dc[:2, :3].issubset(cs)
        assert dc[::-1].issubset(cs)
        assert not dc_2.issubset(cs)
        assert not dc_3.issubset(cs)
        assert dc_t.issubset(cs)
        assert dc_shape.issubset(cs)
        assert not dc_time.issubset(cs)

        # coordinates with dependent lat,lon
        cd = podpac.Coordinates([[lat, lon]], dims=["lat_lon"])
        assert dc.issubset(cd)
        assert dc[:2, :3].issubset(cd)
        assert dc[::-1].issubset(cd)
        assert not dc_2.issubset(cd)
        assert not dc_3.issubset(cd)
        assert dc_t.issubset(cd)
        assert dc_shape.issubset(cd)
        assert not dc_time.issubset(cd)

        # coordinates with unstacked lat, lon
        cu = podpac.Coordinates([ulat, ulon[::-1]], dims=["lat", "lon"])
        assert dc.issubset(cu)
        assert dc[:2, :3].issubset(cu)
        assert dc[::-1].issubset(cu)
        assert not dc_2.issubset(cu)
        assert dc_3.issubset(cu)  # this is an important case!
        assert dc_t.issubset(cu)
        assert dc_shape.issubset(cu)
        assert not dc_time.issubset(cu)

        # coordinates with unstacked lat, lon, time
        cu_time = podpac.Coordinates([ulat, ulon, utime], dims=["lat", "lon", "time"])
        assert dc.issubset(cu_time)
        assert dc[:2, :3].issubset(cu_time)
        assert dc[::-1].issubset(cu_time)
        assert not dc_2.issubset(cu_time)
        assert dc_3.issubset(cu_time)
        assert dc_t.issubset(cu_time)
        assert dc_shape.issubset(cu_time)
        assert dc_time.issubset(cu_time)

        assert not dc.issubset(cu_time[:2, :, :])

        # mixed coordinates
        cmixed = podpac.Coordinates([[ulat, ulon], utime], dims=["lat_lon", "time"])
        assert dc.issubset(cmixed)
        assert dc[:2, :3].issubset(cmixed)
        assert dc[::-1].issubset(cmixed)
        assert not dc_2.issubset(cmixed)
        assert not dc_3.issubset(cmixed)
        assert dc_t.issubset(cmixed)
        assert dc_shape.issubset(cmixed)
        assert dc_time.issubset(cmixed)  # this is the most general case

        assert not dc.issubset(cmixed[:2, :])
        assert not dc_time.issubset(cmixed[:, :1])

    def test_flatten(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = StackedCoordinates([lat, lon])

        assert c.flatten() == StackedCoordinates([lat.flatten(), lon.flatten()])

    def test_reshape(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = StackedCoordinates([lat, lon])

        assert c.reshape((4, 3)) == StackedCoordinates([lat.reshape((4, 3)), lon.reshape((4, 3))])
        assert c.flatten().reshape((3, 4)) == c

    def test_horizontal_resolution(self):
        """Test Horizontal Resolution of Stacked Coordinates. Edge cases are handled in Coordinates.py"""
        lat = podpac.clinspace(-80, 80, 5)
        lat.name = "lat"  # normally assigned when creating Coords object
        lon = podpac.clinspace(-180, 180, 5)
        lon.name = "lon"
        c = StackedCoordinates([lat, lon])

        # Sample Ellipsoid Tuple
        ell_tuple = (6378.137, 6356.752314245179, 0.0033528106647474805)

        # Sample Coordinate name:
        coord_name = "ellipsoidal"

        # Nominal resolution:
        np.testing.assert_almost_equal(
            c.horizontal_resolution(None, ell_tuple, coord_name, restype="nominal").magnitude, 7397047.845631437
        )

        # Summary resolution
        np.testing.assert_almost_equal(
            c.horizontal_resolution(None, ell_tuple, coord_name, restype="summary")[0].magnitude, 7397047.845631437
        )
        np.testing.assert_almost_equal(
            c.horizontal_resolution(None, ell_tuple, coord_name, restype="summary")[1].magnitude, 2134971.4571846593
        )

        # Full resolution
        distance_matrix = [
            [0.0, 5653850.95046188, 11118791.58668857, 14351078.11393555, 17770279.74387375],
            [5653850.95046188, 0.0, 10011843.18838578, 20003931.45862544, 14351078.11393555],
            [11118791.58668857, 10011843.18838578, 0.0, 10011843.18838578, 11118791.58668857],
            [14351078.11393555, 20003931.45862544, 10011843.18838578, 0.0, 5653850.95046188],
            [17770279.74387375, 14351078.11393555, 11118791.58668857, 5653850.95046188, 0.0],
        ]

        np.testing.assert_array_almost_equal(
            c.horizontal_resolution(None, ell_tuple, coord_name, restype="full"), distance_matrix
        )

        # Test different order of lat/lon still works
        c2 = StackedCoordinates([lon, lat])
        np.testing.assert_equal(
            c.horizontal_resolution(None, ell_tuple, "lat").magnitude,
            c2.horizontal_resolution(None, ell_tuple, "lat").magnitude,
        )

        # Test multiple stacked coordinates
        alt = podpac.clinspace(0, 1, 5, "alt")

        c2 = StackedCoordinates([lon, alt, lat])
        np.testing.assert_equal(
            c.horizontal_resolution(None, ell_tuple, "lat").magnitude,
            c2.horizontal_resolution(None, ell_tuple, "lat").magnitude,
        )
        c2 = StackedCoordinates([alt, lon, lat])
        np.testing.assert_equal(
            c.horizontal_resolution(None, ell_tuple, "lat").magnitude,
            c2.horizontal_resolution(None, ell_tuple, "lat").magnitude,
        )
        c2 = StackedCoordinates([lat, alt, lon])
        np.testing.assert_equal(
            c.horizontal_resolution(None, ell_tuple, "lat").magnitude,
            c2.horizontal_resolution(None, ell_tuple, "lat").magnitude,
        )
