from datetime import datetime
import json

import pytest
import traitlets as tl
import numpy as np
import xarray as xr
from numpy.testing import assert_equal

import podpac
from podpac.core.coordinates.utils import make_coord_array
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.coordinates import Coordinates


class TestArrayCoordinatesInit(object):
    def test_empty(self):
        c = ArrayCoordinates1d([])
        a = np.array([], dtype=float)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [np.nan, np.nan])
        with pytest.raises(RuntimeError):
            c.argbounds
        assert c.size == 0
        assert c.shape == (0,)
        assert c.dtype is None
        assert c.deltatype is None
        assert c.is_monotonic is None
        assert c.is_descending is None
        assert c.is_uniform is None
        assert c.start is None
        assert c.stop is None
        assert c.step is None
        repr(c)

    def test_numerical_singleton(self):
        a = np.array([10], dtype=float)
        c = ArrayCoordinates1d(10)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [10.0, 10.0])
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 1
        assert c.shape == (1,)
        assert c.dtype == float
        assert c.deltatype == float
        assert c.is_monotonic == True
        assert c.is_descending is None
        assert c.is_uniform is None
        assert c.start is None
        assert c.stop is None
        assert c.step is None
        repr(c)

    def test_numerical_array(self):
        # unsorted
        values = [1, 6, 0, 4.0]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(a)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0.0, 6.0])
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 4
        assert c.shape == (4,)
        assert c.dtype == float
        assert c.deltatype == float
        assert c.is_monotonic == False
        assert c.is_descending is False
        assert c.is_uniform == False
        assert c.start is None
        assert c.stop is None
        assert c.step is None
        repr(c)

        # sorted ascending
        values = [0, 1, 4, 6]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0.0, 6.0])
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 4
        assert c.shape == (4,)
        assert c.dtype == float
        assert c.deltatype == float
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == False
        assert c.start is None
        assert c.stop is None
        assert c.step is None
        repr(c)

        # sorted descending
        values = [6, 4, 1, 0]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0.0, 6.0])
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 4
        assert c.shape == (4,)
        assert c.dtype == float
        assert c.deltatype == float
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == False
        assert c.start is None
        assert c.stop is None
        assert c.step is None
        repr(c)

        # uniform ascending
        values = [0, 2, 4, 6]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0.0, 6.0])
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 4
        assert c.shape == (4,)
        assert c.dtype == float
        assert c.deltatype == float
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True
        assert c.start == 0.0
        assert c.stop == 6.0
        assert c.step == 2
        repr(c)

        # uniform descending
        values = [6, 4, 2, 0]
        a = np.array(values, dtype=float)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0.0, 6.0])
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 4
        assert c.shape == (4,)
        assert c.dtype == float
        assert c.deltatype == float
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True
        assert c.start == 6.0
        assert c.stop == 0.0
        assert c.step == -2
        repr(c)

    def test_datetime_singleton(self):
        a = np.array("2018-01-01").astype(np.datetime64)
        c = ArrayCoordinates1d("2018-01-01")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(["2018-01-01", "2018-01-01"]).astype(np.datetime64))
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 1
        assert c.shape == (1,)
        assert c.dtype == np.datetime64
        assert c.deltatype == np.timedelta64
        assert c.is_monotonic == True
        assert c.is_descending is None
        assert c.is_uniform is None
        assert c.start is None
        assert c.stop is None
        assert c.step is None
        repr(c)

    def test_datetime_array(self):
        # unsorted
        values = ["2018-01-01", "2019-01-01", "2017-01-01", "2018-01-02"]
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(["2017-01-01", "2019-01-01"]).astype(np.datetime64))
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 4
        assert c.shape == (4,)
        assert c.dtype == np.datetime64
        assert c.deltatype == np.timedelta64
        assert c.is_monotonic == False
        assert c.is_descending == False
        assert c.is_uniform == False
        assert c.start is None
        assert c.stop is None
        assert c.step is None
        repr(c)

        # sorted ascending
        values = ["2017-01-01", "2018-01-01", "2018-01-02", "2019-01-01"]
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(["2017-01-01", "2019-01-01"]).astype(np.datetime64))
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 4
        assert c.shape == (4,)
        assert c.dtype == np.datetime64
        assert c.deltatype == np.timedelta64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == False
        assert c.start is None
        assert c.stop is None
        assert c.step is None
        repr(c)

        # sorted descending
        values = ["2019-01-01", "2018-01-02", "2018-01-01", "2017-01-01"]
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(["2017-01-01", "2019-01-01"]).astype(np.datetime64))
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 4
        assert c.shape == (4,)
        assert c.dtype == np.datetime64
        assert c.deltatype == np.timedelta64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == False
        assert c.start is None
        assert c.stop is None
        assert c.step is None
        repr(c)

        # uniform ascending
        values = ["2017-01-01", "2018-01-01", "2019-01-01"]
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(["2017-01-01", "2019-01-01"]).astype(np.datetime64))
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 3
        assert c.shape == (3,)
        assert c.dtype == np.datetime64
        assert c.deltatype == np.timedelta64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True
        assert c.start == np.datetime64("2017-01-01")
        assert c.stop == np.datetime64("2019-01-01")
        assert c.step == np.timedelta64(365, "D")
        repr(c)

        # uniform descending
        values = ["2019-01-01", "2018-01-01", "2017-01-01"]
        a = np.array(values).astype(np.datetime64)
        c = ArrayCoordinates1d(values)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array(["2017-01-01", "2019-01-01"]).astype(np.datetime64))
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 3
        assert c.shape == (3,)
        assert c.dtype == np.datetime64
        assert c.deltatype == np.timedelta64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True
        assert c.start == np.datetime64("2019-01-01")
        assert c.stop == np.datetime64("2017-01-01")
        assert c.step == np.timedelta64(-365, "D")
        repr(c)

    def test_numerical_shaped(self):
        values = [[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]]
        c = ArrayCoordinates1d(values)
        a = np.array(values, dtype=float)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [1.0, 13.0])
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 6
        assert c.shape == (2, 3)
        assert c.dtype is float
        assert c.deltatype is float
        assert c.is_monotonic is None
        assert c.is_descending is None
        assert c.is_uniform is None
        assert c.start is None
        assert c.stop is None
        assert c.step is None
        repr(c)

    def test_datetime_shaped(self):
        values = [["2017-01-01", "2018-01-01"], ["2019-01-01", "2020-01-01"]]
        c = ArrayCoordinates1d(values)
        a = np.array(values, dtype=np.datetime64)
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [np.datetime64("2017-01-01"), np.datetime64("2020-01-01")])
        assert c.coordinates[c.argbounds[0]] == c.bounds[0]
        assert c.coordinates[c.argbounds[1]] == c.bounds[1]
        assert c.size == 4
        assert c.shape == (2, 2)
        assert c.dtype is np.datetime64
        assert c.deltatype is np.timedelta64
        assert c.is_monotonic is None
        assert c.is_descending is None
        assert c.is_uniform is None
        assert c.start is None
        assert c.stop is None
        assert c.step is None
        repr(c)

    def test_invalid_coords(self):
        with pytest.raises(ValueError, match="Invalid coordinate values"):
            ArrayCoordinates1d([1, 2, "2018-01"])

    def test_from_xarray(self):
        # numerical
        x = xr.DataArray([0, 1, 2], name="lat")
        c = ArrayCoordinates1d.from_xarray(x)
        assert c.name == "lat"
        assert_equal(c.coordinates, x.data)

        # datetime
        x = xr.DataArray([np.datetime64("2018-01-01"), np.datetime64("2018-01-02")], name="time")
        c = ArrayCoordinates1d.from_xarray(x)
        assert c.name == "time"
        assert_equal(c.coordinates, x.data)

        # unnamed
        x = xr.DataArray([0, 1, 2])
        c = ArrayCoordinates1d.from_xarray(x)
        assert c.name is None

    def test_copy(self):
        c = ArrayCoordinates1d([1, 2, 3], name="lat")
        c2 = c.copy()
        assert c is not c2
        assert c == c2

    def test_name(self):
        ArrayCoordinates1d([])
        ArrayCoordinates1d([], name="lat")
        ArrayCoordinates1d([], name="lon")
        ArrayCoordinates1d([], name="alt")
        ArrayCoordinates1d([], name="time")

        with pytest.raises(tl.TraitError):
            ArrayCoordinates1d([], name="depth")

        repr(ArrayCoordinates1d([], name="lat"))

    def test_set_name(self):
        # set if not already set
        c = ArrayCoordinates1d([])
        c._set_name("lat")
        assert c.name == "lat"

        # check if set already
        c = ArrayCoordinates1d([], name="lat")
        c._set_name("lat")
        assert c.name == "lat"

        with pytest.raises(ValueError, match="Dimension mismatch"):
            c._set_name("lon")

        # invalid name
        c = ArrayCoordinates1d([])
        with pytest.raises(tl.TraitError):
            c._set_name("depth")


class TestArrayCoordinatesEq(object):
    def test_eq_type(self):
        c1 = ArrayCoordinates1d([0, 1, 3])
        assert c1 != [0, 1, 3]

    def test_eq_coordinates(self):
        c1 = ArrayCoordinates1d([0, 1, 3])
        c2 = ArrayCoordinates1d([0, 1, 3])
        c3 = ArrayCoordinates1d([0, 1, 3, 4])
        c4 = ArrayCoordinates1d([0, 1, 4])
        c5 = ArrayCoordinates1d([0, 3, 1])

        assert c1 == c2
        assert not c1 == c3
        assert not c1 == c4
        assert not c1 == c5

        c1 = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-04"])
        c2 = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-04"])
        c3 = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-04", "2018-01-05"])
        c4 = ArrayCoordinates1d(["2018-01-01", "2018-01-04", "2018-01-02"])

        assert c1 == c2
        assert not c1 == c3
        assert not c1 == c4

    def test_eq_coordinates_shaped(self):
        c1 = ArrayCoordinates1d([0, 1, 3, 4])
        c2 = ArrayCoordinates1d([0, 1, 3, 4])
        c3 = ArrayCoordinates1d([[0, 1], [3, 4]])
        c4 = ArrayCoordinates1d([[0, 1], [3, 4]])
        c5 = ArrayCoordinates1d([[1, 0], [3, 4]])

        assert c1 == c2
        assert not c1 == c3
        assert not c1 == c4
        assert not c1 == c5

        assert c3 == c4
        assert not c3 == c5

    def test_ne(self):
        # this matters in python 2
        c1 = ArrayCoordinates1d([0, 1, 3])
        c2 = ArrayCoordinates1d([0, 1, 3])
        c3 = ArrayCoordinates1d([0, 1, 3, 4])
        c4 = ArrayCoordinates1d([0, 1, 4])
        c5 = ArrayCoordinates1d([0, 3, 1])

        assert not c1 != c2
        assert c1 != c3
        assert c1 != c4
        assert c1 != c5

        c1 = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-04"])
        c2 = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-04"])
        c3 = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-04", "2018-01-05"])
        c4 = ArrayCoordinates1d(["2018-01-01", "2018-01-04", "2018-01-02"])

        assert not c1 != c2
        assert c1 != c3
        assert c1 != c4

    def test_eq_name(self):
        c1 = ArrayCoordinates1d([0, 1, 3], name="lat")
        c2 = ArrayCoordinates1d([0, 1, 3], name="lat")
        c3 = ArrayCoordinates1d([0, 1, 3], name="lon")
        c4 = ArrayCoordinates1d([0, 1, 3])

        assert c1 == c2
        assert c1 != c3
        assert c1 != c4

        c4.name = "lat"
        assert c1 == c4


class TestArrayCoordinatesSerialization(object):
    def test_definition(self):
        # numerical
        c = ArrayCoordinates1d([0, 1, 2], name="lat")
        d = c.definition
        assert isinstance(d, dict)
        assert set(d.keys()) == {"values", "name"}
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)  # test serializable
        c2 = ArrayCoordinates1d.from_definition(d)  # test from_definition
        assert c2 == c

        # datetimes
        c = ArrayCoordinates1d(["2018-01-01", "2018-01-02"])
        d = c.definition
        assert isinstance(d, dict)
        assert set(d.keys()) == {"values"}
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)  # test serializable
        c2 = ArrayCoordinates1d.from_definition(d)  # test from_definition
        assert c2 == c

    def test_definition_shaped(self):
        # numerical
        c = ArrayCoordinates1d([[0, 1, 2], [3, 4, 5]], name="lat")
        d = c.definition
        assert isinstance(d, dict)
        assert set(d.keys()) == {"values", "name"}
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)  # test serializable
        c2 = ArrayCoordinates1d.from_definition(d)  # test from_definition
        assert c2 == c

        # datetimes
        c = ArrayCoordinates1d([["2018-01-01", "2018-01-02"], ["2018-01-03", "2018-01-04"]])
        d = c.definition
        assert isinstance(d, dict)
        assert set(d.keys()) == {"values"}
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)  # test serializable
        c2 = ArrayCoordinates1d.from_definition(d)  # test from_definition
        assert c2 == c


class TestArrayCoordinatesProperties(object):
    def test_dims(self):
        c = ArrayCoordinates1d([], name="lat")
        assert c.dims == ("lat",)
        assert c.udims == ("lat",)

        c = ArrayCoordinates1d([])
        with pytest.raises(TypeError, match="cannot access dims property of unnamed Coordinates1d"):
            c.dims
        with pytest.raises(TypeError, match="cannot access dims property of unnamed Coordinates1d"):
            c.udims

    def test_xdims(self):
        c = ArrayCoordinates1d([], name="lat")
        assert isinstance(c.xdims, tuple)
        assert c.xdims == ("lat",)

        c = ArrayCoordinates1d([0, 1, 2], name="lat")
        assert isinstance(c.xdims, tuple)
        assert c.xdims == ("lat",)

    def test_xdims_shaped(self):
        c = ArrayCoordinates1d([[0, 1, 2], [10, 11, 12]], name="lat")
        assert isinstance(c.xdims, tuple)
        assert len(set(c.xdims)) == 2

    def test_properties(self):
        c = ArrayCoordinates1d([])
        assert isinstance(c.properties, dict)
        assert set(c.properties) == set()

        c = ArrayCoordinates1d([], name="lat")
        assert isinstance(c.properties, dict)
        assert set(c.properties) == {"name"}

    def test_xcoords(self):
        c = ArrayCoordinates1d([1, 2], name="lat")
        x = xr.DataArray(np.empty(c.shape), dims=c.xdims, coords=c.xcoords)
        np.testing.assert_array_equal(x["lat"].data, c.coordinates)

    def test_xcoords_shaped(self):
        c = ArrayCoordinates1d([[0, 1, 2], [10, 11, 12]], name="lat")
        x = xr.DataArray(np.empty(c.shape), dims=c.xdims, coords=c.xcoords)
        np.testing.assert_array_equal(x["lat"].data, c.coordinates)

    def test_xcoords_unnamed(self):
        c = ArrayCoordinates1d([1, 2])
        with pytest.raises(ValueError, match="Cannot get xcoords"):
            c.xcoords


class TestArrayCoordinatesIndexing(object):
    def test_len(self):
        c = ArrayCoordinates1d([])
        assert len(c) == 0

        c = ArrayCoordinates1d([0, 1, 2])
        assert len(c) == 3

    def test_len_shaped(self):
        c = ArrayCoordinates1d([[0, 1, 2], [3, 4, 5]])
        assert len(c) == 2

    def test_index(self):
        c = ArrayCoordinates1d([20, 50, 60, 90, 40, 10], name="lat")

        # int
        c2 = c[2]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [60])

        c2 = c[-2]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [40])

        # slice
        c2 = c[:2]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [20, 50])

        c2 = c[::2]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [20, 60, 40])

        c2 = c[1:-1]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [50, 60, 90, 40])

        c2 = c[::-1]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [10, 40, 90, 60, 50, 20])

        # array
        c2 = c[[0, 3, 1]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [20, 90, 50])

        # boolean array
        c2 = c[[True, True, True, False, True, False]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [20, 50, 60, 40])

        # invalid
        with pytest.raises(IndexError):
            c[0.3]

        with pytest.raises(IndexError):
            c[10]

    def test_index_shaped(self):
        c = ArrayCoordinates1d([[20, 50, 60], [90, 40, 10]], name="lat")

        # multi-index
        c2 = c[0, 2]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.ndim == 1
        assert c2.shape == (1,)
        assert_equal(c2.coordinates, [60])

        # single-index
        c2 = c[0]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.ndim == 1
        assert c2.shape == (3,)
        assert_equal(c2.coordinates, [20, 50, 60])

        # boolean array
        c2 = c[np.array([[True, True, True], [False, True, False]])]  # has to be a numpy array
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.ndim == 1
        assert c2.shape == (4,)
        assert_equal(c2.coordinates, [20, 50, 60, 40])

    def test_in(self):
        c = ArrayCoordinates1d([20, 50, 60, 90, 40, 10], name="lat")
        assert 20.0 in c
        assert 50.0 in c
        assert 20 in c
        assert 5.0 not in c
        assert np.datetime64("2018") not in c
        assert "a" not in c

        c = ArrayCoordinates1d(["2020-01-01", "2020-01-05", "2020-01-04"], name="time")
        assert np.datetime64("2020-01-01") in c
        assert np.datetime64("2020-01-05") in c
        assert "2020-01-01" in c
        assert np.datetime64("2020-01-02") not in c
        assert 10 not in c
        assert "a" not in c

    def test_in_shaped(self):
        c = ArrayCoordinates1d([[20, 50, 60], [90, 40, 10]], name="lat")
        assert 20.0 in c
        assert 50.0 in c
        assert 20 in c
        assert 5.0 not in c
        assert np.datetime64("2018") not in c
        assert "a" not in c

        c = ArrayCoordinates1d([["2020-01-01", "2020-01-05"], ["2020-01-04", "2020-01-03"]], name="time")
        assert np.datetime64("2020-01-01") in c
        assert np.datetime64("2020-01-05") in c
        assert "2020-01-01" in c
        assert np.datetime64("2020-01-02") not in c
        assert 10 not in c
        assert "a" not in c


class TestArrayCoordinatesAreaBounds(object):
    def test_get_area_bounds_numerical(self):
        values = np.array([0.0, 1.0, 4.0, 6.0])
        c = ArrayCoordinates1d(values)

        # point
        area_bounds = c.get_area_bounds(None)
        assert_equal(area_bounds, [0.0, 6.0])

        # uniform
        area_bounds = c.get_area_bounds(0.5)
        assert_equal(area_bounds, [-0.5, 6.5])

        # segment
        area_bounds = c.get_area_bounds([-0.2, 0.7])
        assert_equal(area_bounds, [-0.2, 6.7])

        # polygon (i.e. there would be corresponding offets for another dimension)
        area_bounds = c.get_area_bounds([-0.2, -0.5, 0.7, 0.5])
        assert_equal(area_bounds, [-0.5, 6.7])

        # boundaries
        area_bounds = c.get_area_bounds([[-0.4, 0.1], [-0.3, 0.2], [-0.2, 0.3], [-0.1, 0.4]])
        assert_equal(area_bounds, [-0.4, 6.4])

    def test_get_area_bounds_datetime(self):
        values = make_coord_array(["2017-01-02", "2017-01-01", "2019-01-01", "2018-01-01"])
        c = ArrayCoordinates1d(values)

        # point
        area_bounds = c.get_area_bounds(None)
        assert_equal(area_bounds, make_coord_array(["2017-01-01", "2019-01-01"]))

        # uniform
        area_bounds = c.get_area_bounds("1,D")
        assert_equal(area_bounds, make_coord_array(["2016-12-31", "2019-01-02"]))

        area_bounds = c.get_area_bounds("1,M")
        assert_equal(area_bounds, make_coord_array(["2016-12-01", "2019-02-01"]))

        area_bounds = c.get_area_bounds("1,Y")
        assert_equal(area_bounds, make_coord_array(["2016-01-01", "2020-01-01"]))

        # segment
        area_bounds = c.get_area_bounds(["0,h", "12,h"])
        assert_equal(area_bounds, make_coord_array(["2017-01-01 00:00", "2019-01-01 12:00"]))

    def test_get_area_bounds_empty(self):
        c = ArrayCoordinates1d([])
        area_bounds = c.get_area_bounds(1.0)
        assert np.all(np.isnan(area_bounds))

    @pytest.mark.xfail(reason="spec uncertain")
    def test_get_area_bounds_overlapping(self):
        values = np.array([0.0, 1.0, 4.0, 6.0])
        c = ArrayCoordinates1d(values)

        area_bounds = c.get_area_bounds([[-0.1, 0.1], [-10.0, 10.0], [-0.1, 0.1], [-0.1, 0.1]])
        assert_equal(area_bounds, [-11.0, 11.0])


class TestArrayCoordinatesSelection(object):
    def test_select_empty_shortcut(self):
        c = ArrayCoordinates1d([])
        bounds = [0, 1]

        s = c.select(bounds)
        assert_equal(s.coordinates, [])

        s, I = c.select(bounds, return_index=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_all_shortcut(self):
        c = ArrayCoordinates1d([20.0, 50.0, 60.0, 90.0, 40.0, 10.0])
        bounds = [0, 100]

        s = c.select(bounds)
        assert_equal(s.coordinates, c.coordinates)

        s, I = c.select(bounds, return_index=True)
        assert_equal(s.coordinates, c.coordinates)
        assert_equal(c.coordinates[I], c.coordinates)

    def test_select_none_shortcut(self):
        c = ArrayCoordinates1d([20.0, 50.0, 60.0, 90.0, 40.0, 10.0])

        # above
        s = c.select([100, 200])
        assert_equal(s.coordinates, [])

        s, I = c.select([100, 200], return_index=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

        # below
        s = c.select([0, 5])
        assert_equal(s.coordinates, [])

        s, I = c.select([0, 5], return_index=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select(self):
        c = ArrayCoordinates1d([20.0, 50.0, 60.0, 90.0, 40.0, 10.0])

        # inner
        s = c.select([30.0, 55.0])
        assert_equal(s.coordinates, [50.0, 40.0])

        s, I = c.select([30.0, 55.0], return_index=True)
        assert_equal(s.coordinates, [50.0, 40.0])
        assert_equal(c.coordinates[I], [50.0, 40.0])

        # inner with aligned bounds
        s = c.select([40.0, 60.0])
        assert_equal(s.coordinates, [50.0, 60.0, 40.0])

        s, I = c.select([40.0, 60.0], return_index=True)
        assert_equal(s.coordinates, [50.0, 60.0, 40.0])
        assert_equal(c.coordinates[I], [50.0, 60.0, 40.0])

        # above
        s = c.select([50, 100])
        assert_equal(s.coordinates, [50.0, 60.0, 90.0])

        s, I = c.select([50, 100], return_index=True)
        assert_equal(s.coordinates, [50.0, 60.0, 90.0])
        assert_equal(c.coordinates[I], [50.0, 60.0, 90.0])

        # below
        s = c.select([0, 50])
        assert_equal(s.coordinates, [20.0, 50.0, 40.0, 10.0])

        s, I = c.select([0, 50], return_index=True)
        assert_equal(s.coordinates, [20.0, 50.0, 40.0, 10.0])
        assert_equal(c.coordinates[I], [20.0, 50.0, 40.0, 10.0])

        # between coordinates
        s = c.select([52, 55])
        assert_equal(s.coordinates, [])

        s, I = c.select([52, 55], return_index=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

        # backwards bounds
        s = c.select([70, 30])
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], return_index=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_one_between_coords(self):
        # Ascending
        c = ArrayCoordinates1d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        c1, inds1 = c.select([5.6, 6.1], return_index=True, outer=False)
        assert np.argwhere(inds1).squeeze() == 6

        c2, inds2 = c.select([5.6, 5.6], return_index=True, outer=False)
        assert np.argwhere(inds2).squeeze() == 6

        c3, inds3 = c.select([5.4, 5.4], return_index=True, outer=False)
        assert np.argwhere(inds3).squeeze() == 5

        c3b, inds3b = c.select([5.4, 5.6], return_index=True, outer=False)
        assert np.all(np.argwhere(inds3b).squeeze() == [5, 6])

        c4, inds4 = c.select([9.1, 9.1], return_index=True, outer=False)
        assert inds4 == []

        c5, inds5 = c.select([-0.1, -0.1], return_index=True, outer=False)
        assert inds5 == []

        # Decending
        c = ArrayCoordinates1d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9][::-1])
        c1, inds1 = c.select([5.6, 6.1], return_index=True, outer=False)
        assert np.argwhere(inds1).squeeze() == 3

        c2, inds2 = c.select([5.6, 5.6], return_index=True, outer=False)
        assert np.argwhere(inds2).squeeze() == 3

        c3, inds3 = c.select([5.4, 5.4], return_index=True, outer=False)
        assert np.argwhere(inds3).squeeze() == 4

        c3b, inds3b = c.select([5.4, 5.6], return_index=True, outer=False)
        assert np.all(np.argwhere(inds3b).squeeze() == [3, 4])

        c4, inds4 = c.select([9.1, 9.1], return_index=True, outer=False)
        assert inds4 == []

        c5, inds5 = c.select([-0.1, -0.1], return_index=True, outer=False)
        assert inds5 == []

    def test_select_outer_ascending(self):
        c = ArrayCoordinates1d([10.0, 20.0, 40.0, 50.0, 60.0, 90.0])

        # inner
        s = c.select([30.0, 55.0], outer=True)
        assert_equal(s.coordinates, [20, 40.0, 50.0, 60.0])

        s, I = c.select([30.0, 55.0], outer=True, return_index=True)
        assert_equal(s.coordinates, [20, 40.0, 50.0, 60.0])
        assert_equal(c.coordinates[I], [20, 40.0, 50.0, 60.0])

        # inner with aligned bounds
        s = c.select([40.0, 60.0], outer=True)
        assert_equal(s.coordinates, [40.0, 50.0, 60.0])

        s, I = c.select([40.0, 60.0], outer=True, return_index=True)
        assert_equal(s.coordinates, [40.0, 50.0, 60.0])
        assert_equal(c.coordinates[I], [40.0, 50.0, 60.0])

        # above
        s = c.select([50, 100], outer=True)
        assert_equal(s.coordinates, [50.0, 60.0, 90.0])

        s, I = c.select([50, 100], outer=True, return_index=True)
        assert_equal(s.coordinates, [50.0, 60.0, 90.0])
        assert_equal(c.coordinates[I], [50.0, 60.0, 90.0])

        # below
        s = c.select([0, 50], outer=True)
        assert_equal(s.coordinates, [10.0, 20.0, 40.0, 50.0])

        s, I = c.select([0, 50], outer=True, return_index=True)
        assert_equal(s.coordinates, [10.0, 20.0, 40.0, 50.0])
        assert_equal(c.coordinates[I], [10.0, 20.0, 40.0, 50.0])

        # between coordinates
        s = c.select([52, 55], outer=True)
        assert_equal(s.coordinates, [50, 60])

        s, I = c.select([52, 55], outer=True, return_index=True)
        assert_equal(s.coordinates, [50, 60])
        assert_equal(c.coordinates[I], [50, 60])

        # backwards bounds
        s = c.select([70, 30], outer=True)
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], outer=True, return_index=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_outer_descending(self):
        c = ArrayCoordinates1d([90.0, 60.0, 50.0, 40.0, 20.0, 10.0])

        # inner
        s = c.select([30.0, 55.0], outer=True)
        assert_equal(s.coordinates, [60.0, 50.0, 40.0, 20.0])

        s, I = c.select([30.0, 55.0], outer=True, return_index=True)
        assert_equal(s.coordinates, [60.0, 50.0, 40.0, 20.0])
        assert_equal(c.coordinates[I], [60.0, 50.0, 40.0, 20.0])

        # inner with aligned bounds
        s = c.select([40.0, 60.0], outer=True)
        assert_equal(s.coordinates, [60.0, 50.0, 40.0])

        s, I = c.select([40.0, 60.0], outer=True, return_index=True)
        assert_equal(s.coordinates, [60.0, 50.0, 40.0])
        assert_equal(c.coordinates[I], [60.0, 50.0, 40.0])

        # above
        s = c.select([50, 100], outer=True)
        assert_equal(s.coordinates, [90.0, 60.0, 50.0])

        s, I = c.select([50, 100], outer=True, return_index=True)
        assert_equal(s.coordinates, [90.0, 60.0, 50.0])
        assert_equal(c.coordinates[I], [90.0, 60.0, 50.0])

        # below
        s = c.select([0, 50], outer=True)
        assert_equal(s.coordinates, [50.0, 40.0, 20.0, 10.0])

        s, I = c.select([0, 50], outer=True, return_index=True)
        assert_equal(s.coordinates, [50.0, 40.0, 20.0, 10.0])
        assert_equal(c.coordinates[I], [50.0, 40.0, 20.0, 10.0])

        # between coordinates
        s = c.select([52, 55], outer=True)
        assert_equal(s.coordinates, [60, 50])

        s, I = c.select([52, 55], outer=True, return_index=True)
        assert_equal(s.coordinates, [60, 50])
        assert_equal(c.coordinates[I], [60, 50])

        # backwards bounds
        s = c.select([70, 30], outer=True)
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], outer=True, return_index=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_outer_nonmonotonic(self):
        c = ArrayCoordinates1d([20.0, 40.0, 60.0, 10.0, 90.0, 50.0])

        # inner
        s = c.select([30.0, 55.0], outer=True)
        assert_equal(s.coordinates, [20, 40.0, 60.0, 50.0])

        s, I = c.select([30.0, 55.0], outer=True, return_index=True)
        assert_equal(s.coordinates, [20, 40.0, 60.0, 50.0])
        assert_equal(c.coordinates[I], [20, 40.0, 60.0, 50.0])

        # inner with aligned bounds
        s = c.select([40.0, 60.0], outer=True)
        assert_equal(s.coordinates, [40.0, 60.0, 50.0])

        s, I = c.select([40.0, 60.0], outer=True, return_index=True)
        assert_equal(s.coordinates, [40.0, 60.0, 50.0])
        assert_equal(c.coordinates[I], [40.0, 60.0, 50.0])

        # above
        s = c.select([50, 100], outer=True)
        assert_equal(s.coordinates, [60.0, 90.0, 50.0])

        s, I = c.select([50, 100], outer=True, return_index=True)
        assert_equal(s.coordinates, [60.0, 90.0, 50.0])
        assert_equal(c.coordinates[I], [60.0, 90.0, 50.0])

        # below
        s = c.select([0, 50], outer=True)
        assert_equal(s.coordinates, [20.0, 40.0, 10.0, 50.0])

        s, I = c.select([0, 50], outer=True, return_index=True)
        assert_equal(s.coordinates, [20.0, 40.0, 10.0, 50.0])
        assert_equal(c.coordinates[I], [20.0, 40.0, 10.0, 50.0])

        # between coordinates
        s = c.select([52, 55], outer=True)
        assert_equal(s.coordinates, [60, 50])

        s, I = c.select([52, 55], outer=True, return_index=True)
        assert_equal(s.coordinates, [60, 50])
        assert_equal(c.coordinates[I], [60, 50])

        # backwards bounds
        s = c.select([70, 30], outer=True)
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], outer=True, return_index=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_dict(self):
        c = ArrayCoordinates1d([20.0, 40.0, 60.0, 10.0, 90.0, 50.0], name="lat")

        s = c.select({"lat": [30.0, 55.0]})
        assert_equal(s.coordinates, [40.0, 50.0])

        s = c.select({"lon": [30.0, 55]})
        assert s == c

    def test_select_time(self):
        c = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], name="time")
        s = c.select({"time": [np.datetime64("2018-01-03"), "2018-02-06"]})
        assert_equal(s.coordinates, np.array(["2018-01-03", "2018-01-04"]).astype(np.datetime64))

    def test_select_time_variable_precision(self):
        c = ArrayCoordinates1d(["2012-05-19"], name="time")
        c2 = ArrayCoordinates1d(["2012-05-19T12:00:00"], name="time")
        s = c.select(c2.bounds, outer=True)
        s1 = c.select(c2.bounds, outer=False)
        s2 = c2.select(c.bounds)
        assert s.size == 1
        assert s1.size == 0
        assert s2.size == 1

    def test_select_dtype(self):
        c = ArrayCoordinates1d([20.0, 40.0, 60.0, 10.0, 90.0, 50.0], name="lat")
        with pytest.raises(TypeError):
            c.select({"lat": [np.datetime64("2018-01-01"), "2018-02-01"]})

        c = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], name="time")
        with pytest.raises(TypeError):
            c.select({"time": [1, 10]})

    def test_select_shaped(self):
        c = ArrayCoordinates1d([[20.0, 50.0, 60.0], [90.0, 40.0, 10.0]])

        # inner
        s = c.select([30.0, 55.0])
        assert_equal(s.coordinates, [50.0, 40.0])

        s, I = c.select([30.0, 55.0], return_index=True)
        assert_equal(s.coordinates, [50.0, 40.0])
        assert_equal(c.coordinates[I], [50.0, 40.0])

        # inner with aligned bounds
        s = c.select([40.0, 60.0])
        assert_equal(s.coordinates, [50.0, 60.0, 40.0])

        s, I = c.select([40.0, 60.0], return_index=True)
        assert_equal(s.coordinates, [50.0, 60.0, 40.0])
        assert_equal(c.coordinates[I], [50.0, 60.0, 40.0])

        # above
        s = c.select([50, 100])
        assert_equal(s.coordinates, [50.0, 60.0, 90.0])

        s, I = c.select([50, 100], return_index=True)
        assert_equal(s.coordinates, [50.0, 60.0, 90.0])
        assert_equal(c.coordinates[I], [50.0, 60.0, 90.0])

        # below
        s = c.select([0, 50])
        assert_equal(s.coordinates, [20.0, 50.0, 40.0, 10.0])

        s, I = c.select([0, 50], return_index=True)
        assert_equal(s.coordinates, [20.0, 50.0, 40.0, 10.0])
        assert_equal(c.coordinates[I], [20.0, 50.0, 40.0, 10.0])

        # between coordinates
        s = c.select([52, 55])
        assert_equal(s.coordinates, [])

        s, I = c.select([52, 55], return_index=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

        # backwards bounds
        s = c.select([70, 30])
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], return_index=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_shaped_outer_nonmonotonic(self):
        c = ArrayCoordinates1d([[20.0, 40.0, 60.0], [10.0, 90.0, 50.0]])

        # inner
        s = c.select([30.0, 55.0], outer=True)
        assert_equal(s.coordinates, [20, 40.0, 60.0, 50.0])

        s, I = c.select([30.0, 55.0], outer=True, return_index=True)
        assert_equal(s.coordinates, [20, 40.0, 60.0, 50.0])
        assert_equal(c.coordinates[I], [20, 40.0, 60.0, 50.0])

        # inner with aligned bounds
        s = c.select([40.0, 60.0], outer=True)
        assert_equal(s.coordinates, [40.0, 60.0, 50.0])

        s, I = c.select([40.0, 60.0], outer=True, return_index=True)
        assert_equal(s.coordinates, [40.0, 60.0, 50.0])
        assert_equal(c.coordinates[I], [40.0, 60.0, 50.0])

        # above
        s = c.select([50, 100], outer=True)
        assert_equal(s.coordinates, [60.0, 90.0, 50.0])

        s, I = c.select([50, 100], outer=True, return_index=True)
        assert_equal(s.coordinates, [60.0, 90.0, 50.0])
        assert_equal(c.coordinates[I], [60.0, 90.0, 50.0])

        # below
        s = c.select([0, 50], outer=True)
        assert_equal(s.coordinates, [20.0, 40.0, 10.0, 50.0])

        s, I = c.select([0, 50], outer=True, return_index=True)
        assert_equal(s.coordinates, [20.0, 40.0, 10.0, 50.0])
        assert_equal(c.coordinates[I], [20.0, 40.0, 10.0, 50.0])

        # between coordinates
        s = c.select([52, 55], outer=True)
        assert_equal(s.coordinates, [60, 50])

        s, I = c.select([52, 55], outer=True, return_index=True)
        assert_equal(s.coordinates, [60, 50])
        assert_equal(c.coordinates[I], [60, 50])

        # backwards bounds
        s = c.select([70, 30], outer=True)
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], outer=True, return_index=True)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])


class TestArrayCoordinatesMethods(object):
    def test_unique(self):
        c = ArrayCoordinates1d([1, 2, 3, 2])

        u = c.unique()
        assert u.shape == (3,)
        assert_equal(u.coordinates, [1, 2, 3])

        u, I = c.unique(return_index=True)
        assert u == c[I]
        assert_equal(u.coordinates, [1, 2, 3])

    def test_unique_monotonic(self):
        c = ArrayCoordinates1d([1, 2, 3, 5])

        u = c.unique()
        assert u == c

        u, I = c.unique(return_index=True)
        assert u == c
        assert u == c[I]

    def test_unique_shaped(self):
        c = ArrayCoordinates1d([[1, 2], [3, 2]])

        # also flattens
        u = c.unique()
        assert u.shape == (3,)
        assert_equal(u.coordinates, [1, 2, 3])

        u, I = c.unique(return_index=True)
        assert u == c.flatten()[I]
        assert_equal(u.coordinates, [1, 2, 3])

    def test_simplify(self):
        # convert to UniformCoordinates
        c = ArrayCoordinates1d([1, 2, 3, 4])
        c2 = c.simplify()
        assert isinstance(c2, UniformCoordinates1d)
        assert c2 == c

        # reversed, step -2
        c = ArrayCoordinates1d([4, 2, 0])
        c2 = c.simplify()
        assert isinstance(c2, UniformCoordinates1d)
        assert c2 == c

        # don't simplify
        c = ArrayCoordinates1d([1, 2, 4])
        c2 = c.simplify()
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2 == c

        # time, convert to UniformCoordinates
        c = ArrayCoordinates1d(["2020-01-01", "2020-01-02", "2020-01-03"])
        c2 = c.simplify()
        assert isinstance(c2, UniformCoordinates1d)
        assert c2 == c

        # time, reverse -2,H
        c = ArrayCoordinates1d(["2020-01-01T12:00", "2020-01-01T10:00", "2020-01-01T08:00"])
        c2 = c.simplify()
        assert isinstance(c2, UniformCoordinates1d)
        assert c2 == c

        # time, don't simplify
        c = ArrayCoordinates1d(["2020-01-01", "2020-01-02", "2020-01-04"])
        c2 = c.simplify()
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2 == c

        # empty
        c = ArrayCoordinates1d([])
        c2 = c.simplify()
        assert c2 == c

    def test_simplify_shaped(self):
        # don't simplify
        c = ArrayCoordinates1d([[1, 2], [3, 4]])
        c2 = c.simplify()
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2 == c

    def test_issubset(self):
        c1 = ArrayCoordinates1d([2, 1])
        c2 = ArrayCoordinates1d([1, 2, 3])
        c3 = ArrayCoordinates1d([1, 2, 4])
        e = ArrayCoordinates1d([])

        # self
        assert c1.issubset(c1)
        assert e.issubset(e)

        # subsets
        assert c1.issubset(c2)
        assert c1.issubset(c3)
        assert e.issubset(c1)

        # not subsets
        assert not c2.issubset(c1)
        assert not c3.issubset(c1)
        assert not c2.issubset(c3)
        assert not c3.issubset(c2)
        assert not c1.issubset(e)

    def test_issubset_datetime(self):
        c1 = ArrayCoordinates1d(["2020-01-01", "2020-01-02"])
        c2 = ArrayCoordinates1d(["2020-01-01", "2020-01-02", "2020-01-03"])
        c3 = ArrayCoordinates1d(["2020-01-01T00:00", "2020-01-02T00:00"])
        c4 = ArrayCoordinates1d(["2020-01-01T12:00", "2020-01-02T12:00"])

        # same resolution
        assert c1.issubset(c1)
        assert c1.issubset(c2)
        assert not c2.issubset(c1)

        # different resolution
        assert c3.issubset(c2)
        assert c1.issubset(c3)
        assert not c1.issubset(c4)
        assert not c4.issubset(c1)

    def test_issubset_dtype(self):
        c1 = ArrayCoordinates1d([0, 1])
        c2 = ArrayCoordinates1d(["2018", "2019"])
        assert not c1.issubset(c2)
        assert not c2.issubset(c1)

    def test_issubset_uniform_coordinates(self):
        a = ArrayCoordinates1d([2, 1])
        u1 = UniformCoordinates1d(start=1, stop=3, step=1)
        u2 = UniformCoordinates1d(start=1, stop=3, step=0.5)
        u3 = UniformCoordinates1d(start=1, stop=4, step=2)

        # self
        assert a.issubset(u1)
        assert a.issubset(u2)
        assert not a.issubset(u3)

    def test_issubset_coordinates(self):
        a = ArrayCoordinates1d([3, 1], name="lat")
        c1 = Coordinates([[1, 2, 3], [10, 20, 30]], dims=["lat", "lon"])
        c2 = Coordinates([[1, 2, 4], [10, 20, 30]], dims=["lat", "lon"])
        c3 = Coordinates([[10, 20, 30]], dims=["alt"])

        assert a.issubset(c1)
        assert not a.issubset(c2)
        assert not a.issubset(c3)

    def test_issubset_shaped(self):
        c1 = ArrayCoordinates1d([2, 1])
        c2 = ArrayCoordinates1d([[1], [2]])
        c3 = ArrayCoordinates1d([[1, 2], [3, 4]])

        # self
        assert c1.issubset(c1)
        assert c2.issubset(c2)
        assert c3.issubset(c3)

        # subsets
        assert c1.issubset(c2)
        assert c1.issubset(c3)
        assert c2.issubset(c1)
        assert c2.issubset(c3)

        # not subsets
        assert not c3.issubset(c1)
        assert not c3.issubset(c2)

    def test_flatten(self):
        c = ArrayCoordinates1d([1, 2, 3, 2])
        c2 = ArrayCoordinates1d([[1, 2], [3, 2]])
        assert c != c2
        assert c2.flatten() == c
        assert c.flatten() == c

    def test_reshape(self):
        c = ArrayCoordinates1d([1, 2, 3, 2])
        c2 = ArrayCoordinates1d([[1, 2], [3, 2]])
        assert c.reshape((2, 2)) == c2
        assert c2.reshape((4,)) == c
        assert c.reshape((4, 1)) == c2.reshape((4, 1))

        with pytest.raises(ValueError, match="cannot reshape array"):
            c.reshape((5, 4))

        with pytest.raises(ValueError, match="cannot reshape array"):
            c2.reshape((2, 1))
