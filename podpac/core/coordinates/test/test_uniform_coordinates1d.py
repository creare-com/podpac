from datetime import datetime
import json

import pytest
import traitlets as tl
import numpy as np
from numpy.testing import assert_equal

import podpac
from podpac.core.coordinates.utils import make_coord_array
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.coordinates import Coordinates


class TestUniformCoordinatesCreation(object):
    def test_numerical(self):
        # ascending
        c = UniformCoordinates1d(0, 50, 10)
        a = np.array([0, 10, 20, 30, 40, 50], dtype=float)
        assert c.start == 0
        assert c.stop == 50
        assert c.step == 10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0, 50])
        assert c.size == 6
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d(50, 0, -10)
        a = np.array([50, 40, 30, 20, 10, 0], dtype=float)
        assert c.start == 50
        assert c.stop == 0
        assert c.step == -10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0, 50])
        assert c.size == 6
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_numerical_inexact(self):
        # ascending
        c = UniformCoordinates1d(0, 49, 10)
        a = np.array([0, 10, 20, 30, 40], dtype=float)
        assert c.start == 0
        assert c.stop == 49
        assert c.step == 10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [0, 40])
        assert c.size == 5
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d(50, 1, -10)
        a = np.array([50, 40, 30, 20, 10], dtype=float)
        assert c.start == 50
        assert c.stop == 1
        assert c.step == -10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [10, 50])
        assert c.dtype == float
        assert c.size == a.size
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime(self):
        # ascending
        c = UniformCoordinates1d("2018-01-01", "2018-01-04", "1,D")
        a = np.array(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"]).astype(np.datetime64)
        assert c.start == np.datetime64("2018-01-01")
        assert c.stop == np.datetime64("2018-01-04")
        assert c.step == np.timedelta64(1, "D")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d("2018-01-04", "2018-01-01", "-1,D")
        a = np.array(["2018-01-04", "2018-01-03", "2018-01-02", "2018-01-01"]).astype(np.datetime64)
        assert c.start == np.datetime64("2018-01-04")
        assert c.stop == np.datetime64("2018-01-01")
        assert c.step == np.timedelta64(-1, "D")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime_inexact(self):
        # ascending
        c = UniformCoordinates1d("2018-01-01", "2018-01-06", "2,D")
        a = np.array(["2018-01-01", "2018-01-03", "2018-01-05"]).astype(np.datetime64)
        assert c.start == np.datetime64("2018-01-01")
        assert c.stop == np.datetime64("2018-01-06")
        assert c.step == np.timedelta64(2, "D")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d("2018-01-06", "2018-01-01", "-2,D")
        a = np.array(["2018-01-06", "2018-01-04", "2018-01-02"]).astype(np.datetime64)
        assert c.start == np.datetime64("2018-01-06")
        assert c.stop == np.datetime64("2018-01-01")
        assert c.step == np.timedelta64(-2, "D")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime_month_step(self):
        # ascending
        c = UniformCoordinates1d("2018-01-01", "2018-04-01", "1,M")
        a = np.array(["2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01"]).astype(np.datetime64)
        assert c.start == np.datetime64("2018-01-01")
        assert c.stop == np.datetime64("2018-04-01")
        assert c.step == np.timedelta64(1, "M")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d("2018-04-01", "2018-01-01", "-1,M")
        a = np.array(["2018-04-01", "2018-03-01", "2018-02-01", "2018-01-01"]).astype(np.datetime64)
        assert c.start == np.datetime64("2018-04-01")
        assert c.stop == np.datetime64("2018-01-01")
        assert c.step == np.timedelta64(-1, "M")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime_year_step(self):
        # ascending, exact
        c = UniformCoordinates1d("2018-01-01", "2021-01-01", "1,Y")
        a = np.array(["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"]).astype(np.datetime64)
        assert c.start == np.datetime64("2018-01-01")
        assert c.stop == np.datetime64("2021-01-01")
        assert c.step == np.timedelta64(1, "Y")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending, exact
        c = UniformCoordinates1d("2021-01-01", "2018-01-01", "-1,Y")
        a = np.array(["2021-01-01", "2020-01-01", "2019-01-01", "2018-01-01"]).astype(np.datetime64)
        assert c.start == np.datetime64("2021-01-01")
        assert c.stop == np.datetime64("2018-01-01")
        assert c.step == np.timedelta64(-1, "Y")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

        # ascending, inexact (two cases)
        c = UniformCoordinates1d("2018-01-01", "2021-04-01", "1,Y")
        a = np.array(["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"]).astype(np.datetime64)
        assert c.start == np.datetime64("2018-01-01")
        assert c.stop == np.datetime64("2021-04-01")
        assert c.step == np.timedelta64(1, "Y")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        c = UniformCoordinates1d("2018-04-01", "2021-01-01", "1,Y")
        a = np.array(["2018-04-01", "2019-04-01", "2020-04-01"]).astype(np.datetime64)
        assert c.start == np.datetime64("2018-04-01")
        assert c.stop == np.datetime64("2021-01-01")
        assert c.step == np.timedelta64(1, "Y")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending, inexact (two cases)
        c = UniformCoordinates1d("2021-01-01", "2018-04-01", "-1,Y")
        a = np.array(["2021-01-01", "2020-01-01", "2019-01-01", "2018-01-01"]).astype(np.datetime64)
        assert c.start == np.datetime64("2021-01-01")
        assert c.stop == np.datetime64("2018-04-01")
        assert c.step == np.timedelta64(-1, "Y")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

        c = UniformCoordinates1d("2021-04-01", "2018-01-01", "-1,Y")
        a = np.array(["2021-04-01", "2020-04-01", "2019-04-01", "2018-04-01"]).astype(np.datetime64)
        assert c.start == np.datetime64("2021-04-01")
        assert c.stop == np.datetime64("2018-01-01")
        assert c.step == np.timedelta64(-1, "Y")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_numerical_size(self):
        # ascending
        c = UniformCoordinates1d(0, 10, size=20)
        assert c.start == 0
        assert c.stop == 10
        assert c.step == 10 / 19.0
        assert_equal(c.coordinates, np.linspace(0, 10, 20))
        assert_equal(c.bounds, [0, 10])
        assert c.size == 20
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d(10, 0, size=20)
        assert c.start == 10
        assert c.stop == 0
        assert c.step == -10 / 19.0
        assert_equal(c.coordinates, np.linspace(10, 0, 20))
        assert_equal(c.bounds, [0, 10])
        assert c.size == 20
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime_size(self):
        # ascending
        c = UniformCoordinates1d("2018-01-01", "2018-01-10", size=10)
        assert c.start == np.datetime64("2018-01-01")
        assert c.stop == np.datetime64("2018-01-10")
        assert_equal(c.bounds, [np.datetime64("2018-01-01"), np.datetime64("2018-01-10")])
        assert c.size == 10
        assert c.dtype == np.datetime64
        assert c.is_descending == False

        # descending
        c = UniformCoordinates1d("2018-01-10", "2018-01-01", size=10)
        assert c.start == np.datetime64("2018-01-10")
        assert c.stop == np.datetime64("2018-01-01")
        assert_equal(c.bounds, [np.datetime64("2018-01-01"), np.datetime64("2018-01-10")])
        assert c.size == 10
        assert c.dtype == np.datetime64
        assert c.is_descending == True

        # increase resolution
        c = UniformCoordinates1d("2018-01-01", "2018-01-10", size=21)
        assert c.start == np.datetime64("2018-01-01")
        assert c.stop == np.datetime64("2018-01-10")
        assert_equal(c.bounds, [np.datetime64("2018-01-01"), np.datetime64("2018-01-10")])
        assert c.size == 21
        assert c.dtype == np.datetime64
        assert c.is_descending == False

    def test_datetime_size_invalid(self):
        with pytest.raises(ValueError, match="Cannot divide timedelta"):
            c = UniformCoordinates1d("2018-01-01", "2018-01-10", size=20)

    def test_numerical_size_floating_point_error(self):
        c = UniformCoordinates1d(50.619, 50.62795, size=30)
        assert c.size == 30

    def test_numerical_singleton(self):
        # positive step
        c = UniformCoordinates1d(1, 1, 10)
        a = np.array([1], dtype=float)
        assert c.start == 1
        assert c.stop == 1
        assert c.step == 10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [1, 1])
        assert c.size == 1
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == None
        assert c.is_uniform == True

        # negative step
        c = UniformCoordinates1d(1, 1, -10)
        a = np.array([1], dtype=float)
        assert c.start == 1
        assert c.stop == 1
        assert c.step == -10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, [1, 1])
        assert c.size == 1
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == None
        assert c.is_uniform == True

    def test_datetime_singleton(self):
        # positive step
        c = UniformCoordinates1d("2018-01-01", "2018-01-01", "1,D")
        a = np.array(["2018-01-01"]).astype(np.datetime64)
        assert c.start == np.datetime64("2018-01-01")
        assert c.stop == np.datetime64("2018-01-01")
        assert c.step == np.timedelta64(1, "D")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == None
        assert c.is_uniform == True

        # negative step
        c = UniformCoordinates1d("2018-01-01", "2018-01-01", "-1,D")
        a = np.array(["2018-01-01"]).astype(np.datetime64)
        assert c.start == np.datetime64("2018-01-01")
        assert c.stop == np.datetime64("2018-01-01")
        assert c.step == np.timedelta64(-1, "D")
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == None
        assert c.is_uniform == True

    def test_from_tuple(self):
        # numerical, step
        c = UniformCoordinates1d.from_tuple((0, 10, 0.5))
        assert c.start == 0.0
        assert c.stop == 10.0
        assert c.step == 0.5

        # numerical, size
        c = UniformCoordinates1d.from_tuple((0, 10, 20))
        assert c.start == 0.0
        assert c.stop == 10.0
        assert c.size == 20

        # datetime, step
        c = UniformCoordinates1d.from_tuple(("2018-01-01", "2018-01-04", "1,D"))
        assert c.start == np.datetime64("2018-01-01")
        assert c.stop == np.datetime64("2018-01-04")
        assert c.step == np.timedelta64(1, "D")

        # invalid
        with pytest.raises(ValueError, match="UniformCoordinates1d.from_tuple expects a tuple"):
            UniformCoordinates1d.from_tuple((0, 10))

        with pytest.raises(ValueError, match="UniformCoordinates1d.from_tuple expects a tuple"):
            UniformCoordinates1d.from_tuple(np.array([0, 10, 0.5]))

    def test_copy(self):
        c = UniformCoordinates1d(0, 10, 50, name="lat")
        c2 = c.copy()
        assert c is not c2
        assert c == c2

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 0, 0)

        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, 0)

        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, -10)

        with pytest.raises(ValueError):
            UniformCoordinates1d(50, 0, 10)

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, "2018-01-01", 10)

        with pytest.raises(TypeError):
            UniformCoordinates1d("2018-01-01", 50, 10)

        with pytest.raises(TypeError):
            UniformCoordinates1d("2018-01-01", "2018-01-02", 10)

        with pytest.raises(TypeError):
            UniformCoordinates1d(0.0, "2018-01-01", "1,D")

        with pytest.raises(TypeError):
            UniformCoordinates1d("2018-01-01", 50, "1,D")

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50, "1,D")

        with pytest.raises(ValueError):
            UniformCoordinates1d("a", 50, 10)

        with pytest.raises(ValueError):
            UniformCoordinates1d(0, "b", 10)

        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, "a")

        with pytest.raises(TypeError):
            UniformCoordinates1d()

        with pytest.raises(TypeError):
            UniformCoordinates1d(0)

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50)

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50, 10, size=6)

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 10, size=20.0)

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 10, size="string")

        with pytest.raises(TypeError):
            UniformCoordinates1d("2018-01-10", "2018-01-01", size="1,D")


class TestUniformCoordinatesEq(object):
    def test_equal(self):
        c1 = UniformCoordinates1d(0, 50, 10)
        c2 = UniformCoordinates1d(0, 50, 10)
        c3 = UniformCoordinates1d(0, 50, 10)
        c4 = UniformCoordinates1d(5, 50, 10)
        c5 = UniformCoordinates1d(0, 60, 10)
        c6 = UniformCoordinates1d(0, 50, 5)
        c7 = UniformCoordinates1d(50, 0, -10)

        assert c1 == c2
        assert c1 == c3
        assert c1 != c4
        assert c1 != c5
        assert c1 != c6
        assert c1 != c7

    def test_equal_array_coordinates(self):
        c1 = UniformCoordinates1d(0, 50, 10)
        c2 = ArrayCoordinates1d([0, 10, 20, 30, 40, 50])
        c3 = ArrayCoordinates1d([10, 20, 30, 40, 50, 60])

        assert c1 == c2
        assert c1 != c3


class TestUniformCoordinatesSerialization(object):
    def test_definition(self):
        # numerical
        c = UniformCoordinates1d(0, 50, 10, name="lat")
        d = c.definition
        assert isinstance(d, dict)
        assert set(d.keys()) == set(["start", "stop", "step", "name"])
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)  # test serializable
        c2 = UniformCoordinates1d.from_definition(d)  # test from_definition
        assert c2 == c

        # datetimes
        c = UniformCoordinates1d("2018-01-01", "2018-01-03", "1,D")
        d = c.definition
        assert isinstance(d, dict)
        assert set(d.keys()) == set(["start", "stop", "step"])
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)  # test serializable
        c2 = UniformCoordinates1d.from_definition(d)  # test from_definition
        assert c2 == c

    def test_invalid_definition(self):
        # incorrect definition
        d = {"stop": 50}
        with pytest.raises(ValueError, match='UniformCoordinates1d definition requires "start"'):
            UniformCoordinates1d.from_definition(d)

        d = {"start": 0}
        with pytest.raises(ValueError, match='UniformCoordinates1d definition requires "stop"'):
            UniformCoordinates1d.from_definition(d)

    def test_from_definition_size(self):
        # numerical
        d = {"start": 0, "stop": 50, "size": 6}
        c = UniformCoordinates1d.from_definition(d)
        assert_equal(c.coordinates, [0, 10, 20, 30, 40, 50])

        # datetime, size
        d = {"start": "2018-01-01", "stop": "2018-01-03", "size": 3}
        c = UniformCoordinates1d.from_definition(d)
        assert_equal(c.coordinates, np.array(["2018-01-01", "2018-01-02", "2018-01-03"]).astype(np.datetime64))


class TestUniformCoordinatesIndexing(object):
    def test_len(self):
        c = UniformCoordinates1d(0, 50, 10)
        assert len(c) == 6

    def test_index(self):
        c = UniformCoordinates1d(0, 50, 10, name="lat")

        # int
        c2 = c[2]
        assert isinstance(c2, Coordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [20])

        c2 = c[-2]
        assert isinstance(c2, Coordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [40])

        # slice
        c2 = c[:2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 0
        assert c2.stop == 10
        assert c2.step == 10

        c2 = c[2:]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 20
        assert c2.stop == 50
        assert c2.step == 10

        c2 = c[::2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 0
        assert c2.stop == 50
        assert c2.step == 20

        c2 = c[1:-1]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 10
        assert c2.stop == 40
        assert c2.step == 10

        c2 = c[-3:5]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 30
        assert c2.stop == 40
        assert c2.step == 10

        c2 = c[::-1]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 50
        assert c2.stop == 0
        assert c2.step == -10

        # index array
        c2 = c[[0, 1, 3]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [0, 10, 30])

        c2 = c[[3, 1, 0]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [30, 10, 0])

        c2 = c[[0, 3, 1]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [0, 30, 10])

        c2 = c[[]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [])

        c2 = c[0:0]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [])

        c2 = c[[]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [])

        # boolean array
        c2 = c[[True, True, True, False, True, False]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [0, 10, 20, 40])

        # invalid
        with pytest.raises(IndexError):
            c[0.3]

        with pytest.raises(IndexError):
            c[10]

    def test_index_descending(self):
        c = UniformCoordinates1d(50, 0, -10, name="lat")

        # int
        c2 = c[2]
        assert isinstance(c2, Coordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [30])

        c2 = c[-2]
        assert isinstance(c2, Coordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [10])

        # slice
        c2 = c[:2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 50
        assert c2.stop == 40
        assert c2.step == -10

        c2 = c[2:]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 30
        assert c2.stop == 0
        assert c2.step == -10

        c2 = c[::2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 50
        assert c2.stop == 0
        assert c2.step == -20

        c2 = c[1:-1]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 40
        assert c2.stop == 10
        assert c2.step == -10

        c2 = c[-3:5]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 20
        assert c2.stop == 10
        assert c2.step == -10

        c2 = c[::-1]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 0
        assert c2.stop == 50
        assert c2.step == 10

        # index array
        c2 = c[[0, 1, 3]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [50, 40, 20])

        c2 = c[[3, 1, 0]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [20, 40, 50])

        c2 = c[[0, 3, 1]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [50, 20, 40])

        # boolean array
        c2 = c[[True, True, True, False, True, False]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, [50, 40, 30, 10])

    def test_in(self):
        c = UniformCoordinates1d(0, 50, 10, name="lat")
        assert 0 in c
        assert 10 in c
        assert 50 in c
        assert -10 not in c
        assert 60 not in c
        assert 5 not in c
        assert np.datetime64("2018") not in c
        assert "a" not in c

        c = UniformCoordinates1d(50, 0, -10, name="lat")
        assert 0 in c
        assert 10 in c
        assert 50 in c
        assert -10 not in c
        assert 60 not in c
        assert 5 not in c
        assert np.datetime64("2018") not in c
        assert "a" not in c

        c = UniformCoordinates1d("2020-01-01", "2020-01-09", "2,D", name="time")
        assert np.datetime64("2020-01-01") in c
        assert np.datetime64("2020-01-03") in c
        assert np.datetime64("2020-01-09") in c
        assert np.datetime64("2020-01-11") not in c
        assert np.datetime64("2020-01-02") not in c
        assert 10 not in c
        assert "a" not in c


class TestArrayCoordinatesAreaBounds(object):
    def test_get_area_bounds_numerical(self):
        c = UniformCoordinates1d(0, 50, 10)

        # point
        area_bounds = c.get_area_bounds(None)
        assert_equal(area_bounds, [0.0, 50.0])

        # uniform
        area_bounds = c.get_area_bounds(0.5)
        assert_equal(area_bounds, [-0.5, 50.5])

        # segment
        area_bounds = c.get_area_bounds([-0.2, 0.7])
        assert_equal(area_bounds, [-0.2, 50.7])

        # polygon (i.e. there would be corresponding offets for another dimension)
        area_bounds = c.get_area_bounds([-0.2, -0.5, 0.7, 0.5])
        assert_equal(area_bounds, [-0.5, 50.7])

    def test_get_area_bounds_datetime(self):
        c = UniformCoordinates1d("2018-01-01", "2018-01-04", "1,D")

        # point
        area_bounds = c.get_area_bounds(None)
        assert_equal(area_bounds, make_coord_array(["2018-01-01", "2018-01-04"]))

        # uniform
        area_bounds = c.get_area_bounds("1,D")
        assert_equal(area_bounds, make_coord_array(["2017-12-31", "2018-01-05"]))

        area_bounds = c.get_area_bounds("1,M")
        assert_equal(area_bounds, make_coord_array(["2017-12-01", "2018-02-04"]))

        area_bounds = c.get_area_bounds("1,Y")
        assert_equal(area_bounds, make_coord_array(["2017-01-01", "2019-01-04"]))

        # segment
        area_bounds = c.get_area_bounds(["0,h", "12,h"])
        assert_equal(area_bounds, make_coord_array(["2018-01-01 00:00", "2018-01-04 12:00"]))


class TestUniformCoordinatesSelection(object):
    def test_select_all_shortcut(self):
        c = UniformCoordinates1d(20.0, 70.0, 10.0)

        s = c.select([0, 100])
        assert s.start == 20.0
        assert s.stop == 70.0
        assert s.step == 10.0

        s, I = c.select([0, 100], return_indices=True)
        assert s.start == 20.0
        assert s.stop == 70.0
        assert s.step == 10.0
        assert_equal(c[I], s)

    def test_select_none_shortcut(self):
        c = UniformCoordinates1d(20.0, 70.0, 10.0)

        # above
        s = c.select([100, 200])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        s, I = c.select([100, 200], return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert c[I] == s

        # below
        s = c.select([0, 5])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        s, I = c.select([0, 5], return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert c[I] == s

    def test_select_ascending(self):
        c = UniformCoordinates1d(20.0, 70.0, 10.0)

        # inner
        s = c.select([35.0, 55.0])
        assert s.start == 40.0
        assert s.stop == 50.0
        assert s.step == 10.0

        s, I = c.select([35.0, 55.0], return_indices=True)
        assert s.start == 40.0
        assert s.stop == 50.0
        assert s.step == 10.0
        assert c[I] == s

        # inner with aligned bounds
        s = c.select([30.0, 60.0])
        assert s.start == 30.0
        assert s.stop == 60.0
        assert s.step == 10.0

        s, I = c.select([30.0, 60.0], return_indices=True)
        assert s.start == 30.0
        assert s.stop == 60.0
        assert s.step == 10.0
        assert c[I] == s

        # above
        s = c.select([45, 100])
        assert s.start == 50.0
        assert s.stop == 70.0
        assert s.step == 10.0

        s, I = c.select([45, 100], return_indices=True)
        assert s.start == 50.0
        assert s.stop == 70.0
        assert s.step == 10.0
        assert c[I] == s

        # below
        s = c.select([5, 55])
        assert s.start == 20.0
        assert s.stop == 50.0
        assert s.step == 10.0

        s, I = c.select([5, 55], return_indices=True)
        assert s.start == 20.0
        assert s.stop == 50.0
        assert s.step == 10.0
        assert c[I] == s

        # between coordinates
        s = c.select([52, 55])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        s, I = c.select([52, 55], return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

        # backwards bounds
        s = c.select([70, 30])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_descending(self):
        c = UniformCoordinates1d(70.0, 20.0, -10.0)

        # inner
        s = c.select([35.0, 55.0])
        assert s.start == 50.0
        assert s.stop == 40.0
        assert s.step == -10.0

        s, I = c.select([35.0, 55.0], return_indices=True)
        assert s.start == 50.0
        assert s.stop == 40.0
        assert s.step == -10.0
        assert c[I] == s

        # inner with aligned bounds
        s = c.select([30.0, 60.0])
        assert s.start == 60.0
        assert s.stop == 30.0
        assert s.step == -10.0

        s, I = c.select([30.0, 60.0], return_indices=True)
        assert s.start == 60.0
        assert s.stop == 30.0
        assert s.step == -10.0
        assert c[I] == s

        # above
        s = c.select([45, 100])
        assert s.start == 70.0
        assert s.stop == 50.0
        assert s.step == -10.0

        s, I = c.select([45, 100], return_indices=True)
        assert s.start == 70.0
        assert s.stop == 50.0
        assert s.step == -10.0
        assert c[I] == s

        # below
        s = c.select([5, 55])
        assert s.start == 50.0
        assert s.stop == 20.0
        assert s.step == -10.0

        s, I = c.select([5, 55], return_indices=True)
        assert s.start == 50.0
        assert s.stop == 20.0
        assert s.step == -10.0
        assert c[I] == s

        # between coordinates
        s = c.select([52, 55])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        s, I = c.select([52, 55], return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

        # backwards bounds
        s = c.select([70, 30])
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_outer(self):
        c = UniformCoordinates1d(20.0, 70.0, 10.0)

        # inner
        s = c.select([35.0, 55.0], outer=True)
        assert s.start == 30.0
        assert s.stop == 60.0
        assert s.step == 10.0

        s, I = c.select([35.0, 55.0], outer=True, return_indices=True)
        assert s.start == 30.0
        assert s.stop == 60.0
        assert s.step == 10.0
        assert c[I] == s

        # inner with aligned bounds
        s = c.select([30.0, 60.0], outer=True)
        assert s.start == 30.0
        assert s.stop == 60.0
        assert s.step == 10.0

        s, I = c.select([30.0, 60.0], outer=True, return_indices=True)
        assert s.start == 30.0
        assert s.stop == 60.0
        assert s.step == 10.0
        assert c[I] == s

        # above
        s = c.select([45, 100], outer=True)
        assert s.start == 40.0
        assert s.stop == 70.0
        assert s.step == 10.0

        s, I = c.select([45, 100], outer=True, return_indices=True)
        assert s.start == 40.0
        assert s.stop == 70.0
        assert s.step == 10.0
        assert c[I] == s

        # below
        s = c.select([5, 55], outer=True)
        assert s.start == 20.0
        assert s.stop == 60.0
        assert s.step == 10.0

        s, I = c.select([5, 55], outer=True, return_indices=True)
        assert s.start == 20.0
        assert s.stop == 60.0
        assert s.step == 10.0
        assert c[I] == s

        # between coordinates
        s = c.select([52, 55], outer=True)
        assert s.start == 50.0
        assert s.stop == 60.0
        assert s.step == 10.0

        s, I = c.select([52, 55], outer=True, return_indices=True)
        assert s.start == 50.0
        assert s.stop == 60.0
        assert s.step == 10.0
        assert c[I] == s

        # backwards bounds
        s = c.select([70, 30], outer=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        s, I = c.select([70, 30], outer=True, return_indices=True)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])
        assert_equal(c.coordinates[I], [])

    def test_select_time_variable_precision(self):
        c = UniformCoordinates1d("2012-05-19", "2012-05-20", "1,D", name="time")
        c2 = UniformCoordinates1d("2012-05-20T12:00:00", "2012-05-21T12:00:00", "1,D", name="time")
        s = c.select(c2.bounds, outer=True)
        s1 = c.select(c2.bounds, outer=False)
        s2 = c2.select(c.bounds)
        assert s.size == 1
        assert s1.size == 0
        assert s2.size == 1


class TestUniformCoordinatesMethods(object):
    def test_simplify(self):
        c = UniformCoordinates1d(1, 5, step=1)
        c2 = c.simplify()
        assert c2 == c

        # reversed, step -2
        c = UniformCoordinates1d(4, 0, step=-2)
        c2 = c.simplify()
        assert c2 == c

        # time, convert to UniformCoordinates
        c = UniformCoordinates1d("2020-01-01", "2020-01-05", step="1,D")
        c2 = c.simplify()
        assert c2 == c

        # time, reverse -2,h
        c = UniformCoordinates1d("2020-01-01T12:00", "2020-01-01T08:00", step="-3,h")
        c2 = c.simplify()
        assert c2 == c

    def test_issubset(self):
        c1 = UniformCoordinates1d(2, 1, step=-1)
        c2 = UniformCoordinates1d(1, 3, step=1)
        c3 = UniformCoordinates1d(0, 2, step=1)
        c4 = UniformCoordinates1d(1, 4, step=0.5)
        c5 = UniformCoordinates1d(1.5, 2.5, step=0.5)
        c6 = UniformCoordinates1d(1.4, 2.4, step=0.5)
        c7 = UniformCoordinates1d(1.4, 2.4, step=10)

        # self
        assert c1.issubset(c1)

        # subsets
        assert c1.issubset(c2)
        assert c1.issubset(c3)
        assert c1.issubset(c4)
        assert c5.issubset(c4)
        assert c7.issubset(c6)

        # not subsets
        assert not c2.issubset(c1)
        assert not c2.issubset(c3)
        assert not c3.issubset(c1)
        assert not c3.issubset(c2)
        assert not c4.issubset(c1)
        assert not c6.issubset(c4)

    def test_issubset_datetime(self):
        c1 = UniformCoordinates1d("2020-01-01", "2020-01-03", "1,D")
        c2 = UniformCoordinates1d("2020-01-01", "2020-01-03", "2,D")
        c3 = UniformCoordinates1d("2020-01-01", "2020-01-05", "1,D")
        c4 = UniformCoordinates1d("2020-01-05", "2020-01-01", "-2,D")

        # self
        assert c1.issubset(c1)

        # same resolution
        assert c1.issubset(c3)
        assert c2.issubset(c1)
        assert c2.issubset(c4)
        assert not c1.issubset(c2)
        assert not c1.issubset(c4)
        assert not c3.issubset(c1)

        # different resolution
        c5 = UniformCoordinates1d("2020-01-01T00:00", "2020-01-03T00:00", "1,D")
        c6 = UniformCoordinates1d("2020-01-01T00:00", "2020-01-03T00:00", "6,h")
        assert c1.issubset(c5)
        assert c5.issubset(c1)
        assert c1.issubset(c6)
        assert not c6.issubset(c1)

    def test_issubset_dtype(self):
        c1 = UniformCoordinates1d(0, 10, step=1)
        c2 = UniformCoordinates1d("2018", "2020", step="1,Y")
        assert not c1.issubset(c2)
        assert not c2.issubset(c1)

    def test_issubset_array_coordinates(self):
        u = UniformCoordinates1d(start=1, stop=3, step=1)
        a1 = ArrayCoordinates1d([1, 3, 2])
        a2 = ArrayCoordinates1d([1, 2, 3])
        a3 = ArrayCoordinates1d([1, 3, 4])
        e = ArrayCoordinates1d([])

        # self
        assert u.issubset(a1)
        assert u.issubset(a2)
        assert not u.issubset(a3)
        assert not u.issubset(e)

    def test_issubset_coordinates(self):
        u = UniformCoordinates1d(1, 3, 1, name="lat")
        c1 = Coordinates([[1, 2, 3], [10, 20, 30]], dims=["lat", "lon"])
        c2 = Coordinates([[1, 2, 4], [10, 20, 30]], dims=["lat", "lon"])
        c3 = Coordinates([[10, 20, 30]], dims=["alt"])

        assert u.issubset(c1)
        assert not u.issubset(c2)
        assert not u.issubset(c3)
