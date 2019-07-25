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
    def test_init_Coordinates1d(self):
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

    def test_ctype(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat", ctype="left")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        c = StackedCoordinates([lat, lon], ctype="right")

        # lon ctype set by StackedCoordinates
        assert c["lon"].ctype == "right"

        # but lat is left by StackedCoordinates because it was already explicitly set
        assert c["lat"].ctype == "left"

    def test_coercion_with_dims(self):
        c = StackedCoordinates([[0, 1, 2], [10, 20, 30]], dims=["lat", "lon"])
        assert c.dims == ("lat", "lon")
        assert_equal(c["lat"].coordinates, [0, 1, 2])
        assert_equal(c["lon"].coordinates, [10, 20, 30])

    def test_coercion_with_name(self):
        c = StackedCoordinates([[0, 1, 2], [10, 20, 30]], name="lat_lon")
        assert c.dims == ("lat", "lon")
        assert_equal(c["lat"].coordinates, [0, 1, 2])
        assert_equal(c["lon"].coordinates, [10, 20, 30])

    def test_invalid_coords(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([0, 1, 2, 3], name="lon")
        c = ArrayCoordinates1d([0, 1, 2])

        with pytest.raises(TypeError, match="Unrecognized coords type"):
            StackedCoordinates({})

        with pytest.raises(ValueError, match="Stacked coords must have at least 2 coords"):
            StackedCoordinates([lat])

        with pytest.raises(ValueError, match="Size mismatch in stacked coords"):
            StackedCoordinates([lat, lon])

        with pytest.raises(ValueError, match="Duplicate dimension"):
            StackedCoordinates([lat, lat])

        # but duplicate None name is okay
        StackedCoordinates([c, c])

        # dims and name
        with pytest.raises(TypeError):
            StackedCoordinates([[0, 1, 2], [10, 20, 30]], dims=["lat", "lon"], name="lat_lon")

    def test_from_xarray(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"], name="time")
        c = StackedCoordinates([lat, lon, time])
        x = xr.DataArray(np.empty(c.shape), coords=c.coords, dims=c.idims)

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

    def test_coordinates(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], name="time")
        c = StackedCoordinates([lat, lon, time])

        assert isinstance(c.coordinates, pd.MultiIndex)
        assert c.coordinates.size == 4
        assert c.coordinates.names == ["lat", "lon", "time"]
        assert c.coordinates[0] == (0.0, 10, np.datetime64("2018-01-01"))
        assert c.coordinates[1] == (1.0, 20, np.datetime64("2018-01-02"))
        assert c.coordinates[2] == (2.0, 30, np.datetime64("2018-01-03"))
        assert c.coordinates[3] == (3.0, 40, np.datetime64("2018-01-04"))

    def test_coords(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], name="time")
        c = StackedCoordinates([lat, lon, time])

        assert isinstance(c.coords, dict)
        x = xr.DataArray(np.empty(c.shape), dims=c.idims, coords=c.coords)
        assert x.dims == ("lat_lon_time",)
        assert_equal(x.coords["lat"], c["lat"].coordinates)
        assert_equal(x.coords["lon"], c["lon"].coordinates)
        assert_equal(x.coords["time"], c["time"].coordinates)

        lat = ArrayCoordinates1d([0, 1, 2, 3])
        lon = ArrayCoordinates1d([10, 20, 30, 40])
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"])
        c = StackedCoordinates([lat, lon, time])
        with pytest.raises(ValueError, match="Cannot get coords"):
            c.coords

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

    def test_area_bounds(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ["2018-01-01", "2018-01-02"]

        c = StackedCoordinates([lat, lon], dims=["lat", "lon"])
        area_bounds = c.area_bounds
        assert isinstance(area_bounds, dict)
        assert set(area_bounds.keys()) == set(c.udims)
        assert_equal(area_bounds["lat"], c["lat"].area_bounds)
        assert_equal(area_bounds["lon"], c["lon"].area_bounds)

        c = StackedCoordinates([lat, lon])
        with pytest.raises(ValueError, match="Cannot get area_bounds"):
            c.area_bounds


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


class TestStackedCoordinatesSelection(object):
    def test_select_single(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], name="time")
        c = StackedCoordinates([lat, lon, time])

        # single dimension
        s = c.select({"lat": [0.5, 2.5]})
        assert s == c[1:3]

        s, I = c.select({"lat": [0.5, 2.5]}, return_indices=True)
        assert s == c[I]
        assert s == c[1:3]

        # a different single dimension
        s = c.select({"lon": [5, 25]})
        assert s == c[0:2]

        s, I = c.select({"lon": [5, 25]}, return_indices=True)
        assert s == c[I]
        assert s == c[0:2]

        # outer
        s = c.select({"lat": [0.5, 2.5]}, outer=True)
        assert s == c[0:4]

        s, I = c.select({"lat": [0.5, 2.5]}, outer=True, return_indices=True)
        assert s == c[I]
        assert s == c[0:4]

        # no matching dimension
        s = c.select({"alt": [0, 10]})
        assert s == c

        s, I = c.select({"alt": [0, 10]}, return_indices=True)
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

        s, I = c.select({"lat": [0.5, 3.5], "lon": [25, 55]}, return_indices=True)
        assert s == c[2:4]
        assert s == c[I]
