from datetime import datetime
import json

import pytest
import traitlets as tl
import numpy as np
import pandas as pd
import xarray as xr
from numpy.testing import assert_equal, assert_allclose

import podpac
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.polar_coordinates import PolarCoordinates
from podpac.core.coordinates.cfunctions import clinspace


class TestPolarCoordinatesCreation(object):
    def test_init(self):
        theta = np.linspace(0, 2 * np.pi, 9)[:-1]

        c = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta=theta, dims=["lat", "lon"])
        assert_equal(c.center, [1.5, 2.0])
        assert_equal(c.theta.coordinates, theta)
        assert_equal(c.radius.coordinates, [1, 2, 4, 5])
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert c.xdims == ("r", "t")
        assert c.name == "lat_lon"
        assert c.shape == (4, 8)
        repr(c)

        # uniform theta
        c = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta_size=8, dims=["lat", "lon"])
        assert c.theta.start == 0
        assert c.theta.size == 8
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert c.xdims == ("r", "t")
        assert c.name == "lat_lon"
        assert c.shape == (4, 8)
        repr(c)

    def test_invalid(self):
        with pytest.raises(TypeError, match="PolarCoordinates expected theta or theta_size, not both"):
            PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta=[0, 1, 2], theta_size=8, dims=["lat", "lon"])

        with pytest.raises(TypeError, match="PolarCoordinates requires theta or theta_size"):
            PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], dims=["lat", "lon"])

        with pytest.raises(ValueError, match="PolarCoordinates radius must all be positive"):
            PolarCoordinates(center=[1.5, 2.0], radius=[-1, 2, 4, 5], theta_size=8, dims=["lat", "lon"])

        with pytest.raises(ValueError, match="PolarCoordinates dims"):
            PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta_size=8, dims=["lat", "time"])

        with pytest.raises(ValueError, match="Duplicate dimension"):
            PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta_size=8, dims=["lat", "lat"])

    def test_copy(self):
        c = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta_size=8, dims=["lat", "lon"])
        c2 = c.copy()
        assert c2 is not c
        assert c2 == c


class TestDependentCoordinatesStandardMethods(object):
    def test_eq_type(self):
        c = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta_size=8, dims=["lat", "lon"])
        assert c != []

    def test_eq_center(self):
        c1 = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta_size=8, dims=["lat", "lon"])
        c2 = PolarCoordinates(center=[1.5, 2.5], radius=[1, 2, 4, 5], theta_size=8, dims=["lat", "lon"])
        assert c1 != c2

    def test_eq_radius(self):
        c1 = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta_size=8, dims=["lat", "lon"])
        c2 = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4], theta_size=8, dims=["lat", "lon"])
        assert c1 != c2

    def test_eq_theta(self):
        c1 = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta_size=8, dims=["lat", "lon"])
        c2 = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta_size=7, dims=["lat", "lon"])
        assert c1 != c2

    def test_eq(self):
        c1 = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta_size=8, dims=["lat", "lon"])
        c2 = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta_size=8, dims=["lat", "lon"])
        assert c1 == c2


class TestPolarCoordinatesSerialization(object):
    def test_definition(self):
        # array radius and theta, plus other checks
        c = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta=[0, 1, 2], dims=["lat", "lon"])
        d = c.definition

        assert isinstance(d, dict)
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)  # test serializable
        c2 = PolarCoordinates.from_definition(d)
        assert c2 == c

        # uniform radius and theta
        c = PolarCoordinates(
            center=[1.5, 2.0], radius=clinspace(1, 5, 4), theta=clinspace(0, np.pi, 5), dims=["lat", "lon"]
        )
        d = c.definition
        c2 = PolarCoordinates.from_definition(d)
        assert c2 == c

    def test_from_definition(self):
        # radius and theta lists
        d = {"center": [1.5, 2.0], "radius": [1, 2, 4, 5], "theta": [0, 1, 2], "dims": ["lat", "lon"]}
        c = PolarCoordinates.from_definition(d)
        assert_allclose(c.center, [1.5, 2.0])
        assert_allclose(c.radius.coordinates, [1, 2, 4, 5])
        assert_allclose(c.theta.coordinates, [0, 1, 2])
        assert c.dims == ("lat", "lon")

        # theta size
        d = {"center": [1.5, 2.0], "radius": [1, 2, 4, 5], "theta_size": 8, "dims": ["lat", "lon"]}
        c = PolarCoordinates.from_definition(d)
        assert_allclose(c.center, [1.5, 2.0])
        assert_allclose(c.radius.coordinates, [1, 2, 4, 5])
        assert_allclose(c.theta.coordinates, np.linspace(0, 2 * np.pi, 9)[:-1])
        assert c.dims == ("lat", "lon")

    def test_invalid_definition(self):
        d = {"radius": [1, 2, 4, 5], "theta": [0, 1, 2], "dims": ["lat", "lon"]}
        with pytest.raises(ValueError, match='PolarCoordinates definition requires "center"'):
            PolarCoordinates.from_definition(d)

        d = {"center": [1.5, 2.0], "theta": [0, 1, 2], "dims": ["lat", "lon"]}
        with pytest.raises(ValueError, match='PolarCoordinates definition requires "radius"'):
            PolarCoordinates.from_definition(d)

        d = {"center": [1.5, 2.0], "radius": [1, 2, 4, 5], "dims": ["lat", "lon"]}
        with pytest.raises(ValueError, match='PolarCoordinates definition requires "theta" or "theta_size"'):
            PolarCoordinates.from_definition(d)

        d = {"center": [1.5, 2.0], "radius": [1, 2, 4, 5], "theta": [0, 1, 2]}
        with pytest.raises(ValueError, match='PolarCoordinates definition requires "dims"'):
            PolarCoordinates.from_definition(d)

        d = {"center": [1.5, 2.0], "radius": {"a": 1}, "theta": [0, 1, 2], "dims": ["lat", "lon"]}
        with pytest.raises(ValueError, match="Could not parse radius coordinates"):
            PolarCoordinates.from_definition(d)

        d = {"center": [1.5, 2.0], "radius": [1, 2, 4, 5], "theta": {"a": 1}, "dims": ["lat", "lon"]}
        with pytest.raises(ValueError, match="Could not parse theta coordinates"):
            PolarCoordinates.from_definition(d)

    def test_full_definition(self):
        c = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta=[0, 1, 2], dims=["lat", "lon"])
        d = c.full_definition

        assert isinstance(d, dict)
        assert set(d.keys()) == {"dims", "radius", "center", "theta"}
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)  # test serializable


class TestPolarCoordinatesProperties(object):
    def test_coordinates(self):
        c = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4], theta_size=4, dims=["lat", "lon"])
        lat, lon = c.coordinates

        assert_allclose(lat, [[1.5, 2.5, 1.5, 0.5], [1.5, 3.5, 1.5, -0.5], [1.5, 5.5, 1.5, -2.5]])

        assert_allclose(lon, [[3.0, 2.0, 1.0, 2.0], [4.0, 2.0, 0.0, 2.0], [6.0, 2.0, -2.0, 2.0]])


class TestPolarCoordinatesIndexing(object):
    def test_get_dim(self):
        c = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta_size=8, dims=["lat", "lon"])

        lat = c["lat"]
        lon = c["lon"]
        assert isinstance(lat, ArrayCoordinates1d)
        assert isinstance(lon, ArrayCoordinates1d)
        assert lat.name == "lat"
        assert lon.name == "lon"
        assert_equal(lat.coordinates, c.coordinates[0])
        assert_equal(lon.coordinates, c.coordinates[1])

        with pytest.raises(KeyError, match="Dimension .* not found"):
            c["other"]

    def test_get_index_slices(self):
        c = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5, 6], theta_size=8, dims=["lat", "lon"])

        # full
        c2 = c[1:4, 2:4]
        assert isinstance(c2, PolarCoordinates)
        assert c2.shape == (3, 2)
        assert_allclose(c2.center, c.center)
        assert c2.radius == c.radius[1:4]
        assert c2.theta == c.theta[2:4]
        assert c2.dims == c.dims
        assert_allclose(c2.coordinates[0], c.coordinates[0][1:4, 2:4])
        assert_allclose(c2.coordinates[1], c.coordinates[1][1:4, 2:4])

        # partial/implicit
        c2 = c[1:4]
        assert isinstance(c2, PolarCoordinates)
        assert c2.shape == (3, 8)
        assert_allclose(c2.center, c.center)
        assert c2.radius == c.radius[1:4]
        assert c2.theta == c.theta
        assert c2.dims == c.dims
        assert_allclose(c2.coordinates[0], c.coordinates[0][1:4])
        assert_allclose(c2.coordinates[1], c.coordinates[1][1:4])

        # stepped
        c2 = c[1:4:2, 2:4]
        assert isinstance(c2, PolarCoordinates)
        assert c2.shape == (2, 2)
        assert_allclose(c2.center, c.center)
        assert c2.radius == c.radius[1:4:2]
        assert c2.theta == c.theta[2:4]
        assert c2.dims == c.dims
        assert_allclose(c2.coordinates[0], c.coordinates[0][1:4:2, 2:4])
        assert_allclose(c2.coordinates[1], c.coordinates[1][1:4:2, 2:4])

        # reversed
        c2 = c[4:1:-1, 2:4]
        assert isinstance(c2, PolarCoordinates)
        assert c2.shape == (3, 2)
        assert_allclose(c2.center, c.center)
        assert c2.radius == c.radius[4:1:-1]
        assert c2.theta == c.theta[2:4]
        assert c2.dims == c.dims
        assert_allclose(c2.coordinates[0], c.coordinates[0][4:1:-1, 2:4])
        assert_allclose(c2.coordinates[1], c.coordinates[1][4:1:-1, 2:4])

    def test_get_index_fallback(self):
        c = PolarCoordinates(center=[1.5, 2.0], radius=[1, 2, 4, 5], theta_size=8, dims=["lat", "lon"])
        lat, lon = c.coordinates

        Ra = [3, 1]
        Th = slice(1, 4)
        B = lat > 0.5

        # int/slice/indices
        c2 = c[Ra, Th]
        assert isinstance(c2, StackedCoordinates)
        assert c2.shape == (2, 3)
        assert c2.dims == c.dims
        assert_equal(c2["lat"].coordinates, lat[Ra, Th])
        assert_equal(c2["lon"].coordinates, lon[Ra, Th])

        # boolean
        c2 = c[B]
        assert isinstance(c2, StackedCoordinates)
        assert c2.shape == (22,)
        assert c2.dims == c.dims
        assert_equal(c2["lat"].coordinates, lat[B])
        assert_equal(c2["lon"].coordinates, lon[B])
