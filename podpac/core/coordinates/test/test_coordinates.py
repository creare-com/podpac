import sys
import json
from copy import deepcopy

import pytest
import numpy as np
from numpy.testing import assert_equal
import xarray as xr
import pyproj

import podpac
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.dependent_coordinates import DependentCoordinates
from podpac.core.coordinates.rotated_coordinates import RotatedCoordinates
from podpac.core.coordinates.cfunctions import crange, clinspace
from podpac.core.coordinates.coordinates import Coordinates
from podpac.core.coordinates.coordinates import concat, union, merge_dims


class TestCoordinateCreation(object):
    def test_empty(self):
        c = Coordinates([])
        assert c.dims == tuple()
        assert c.udims == tuple()
        assert c.idims == tuple()
        assert c.shape == tuple()
        assert c.ndim == 0
        assert c.size == 0

    def test_single_dim(self):
        # single value
        date = "2018-01-01"

        c = Coordinates([date], dims=["time"])
        assert c.dims == ("time",)
        assert c.udims == ("time",)
        assert c.idims == ("time",)
        assert c.shape == (1,)
        assert c.ndim == 1
        assert c.size == 1

        # array
        dates = ["2018-01-01", "2018-01-02"]

        c = Coordinates([dates], dims=["time"])
        assert c.dims == ("time",)
        assert c.udims == ("time",)
        assert c.idims == ("time",)
        assert c.shape == (2,)
        assert c.ndim == 1
        assert c.size == 2

        c = Coordinates([np.array(dates).astype(np.datetime64)], dims=["time"])
        assert c.dims == ("time",)
        assert c.udims == ("time",)
        assert c.idims == ("time",)
        assert c.shape == (2,)
        assert c.ndim == 1

        c = Coordinates([xr.DataArray(dates).astype(np.datetime64)], dims=["time"])
        assert c.dims == ("time",)
        assert c.udims == ("time",)
        assert c.idims == ("time",)
        assert c.shape == (2,)
        assert c.ndim == 1
        assert c.size == 2

        # use DataArray name, but dims overrides the DataArray name
        c = Coordinates([xr.DataArray(dates, name="time").astype(np.datetime64)])
        assert c.dims == ("time",)
        assert c.udims == ("time",)
        assert c.idims == ("time",)
        assert c.shape == (2,)
        assert c.ndim == 1
        assert c.size == 2

        c = Coordinates([xr.DataArray(dates, name="a").astype(np.datetime64)], dims=["time"])
        assert c.dims == ("time",)
        assert c.udims == ("time",)
        assert c.idims == ("time",)
        assert c.shape == (2,)
        assert c.ndim == 1
        assert c.size == 2

    def test_unstacked(self):
        # single value
        c = Coordinates([0, 10], dims=["lat", "lon"])
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert c.idims == ("lat", "lon")
        assert c.shape == (1, 1)
        assert c.ndim == 2
        assert c.size == 1

        # arrays
        lat = [0, 1, 2]
        lon = [10, 20, 30, 40]

        c = Coordinates([lat, lon], dims=["lat", "lon"])
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert c.idims == ("lat", "lon")
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

        # use DataArray names
        c = Coordinates([xr.DataArray(lat, name="lat"), xr.DataArray(lon, name="lon")])
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert c.idims == ("lat", "lon")
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

        # dims overrides the DataArray names
        c = Coordinates([xr.DataArray(lat, name="a"), xr.DataArray(lon, name="b")], dims=["lat", "lon"])
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert c.idims == ("lat", "lon")
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

    def test_stacked(self):
        # single value
        c = Coordinates([[0, 10]], dims=["lat_lon"])
        assert c.dims == ("lat_lon",)
        assert c.udims == ("lat", "lon")
        assert c.idims == ("lat_lon",)
        assert c.shape == (1,)
        assert c.ndim == 1
        assert c.size == 1

        # arrays
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        c = Coordinates([[lat, lon]], dims=["lat_lon"])
        assert c.dims == ("lat_lon",)
        assert c.udims == ("lat", "lon")
        assert c.idims == ("lat_lon",)
        assert c.shape == (3,)
        assert c.ndim == 1
        assert c.size == 3

        # nested dims version
        c = Coordinates([[lat, lon]], dims=[["lat", "lon"]])
        assert c.dims == ("lat_lon",)
        assert c.udims == ("lat", "lon")
        assert c.idims == ("lat_lon",)
        assert c.shape == (3,)
        assert c.ndim == 1
        assert c.size == 3

    def test_dependent(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        latlon = DependentCoordinates([lat, lon], dims=["lat", "lon"])
        c = Coordinates([latlon])
        assert c.dims == ("lat,lon",)
        assert c.udims == ("lat", "lon")
        assert c.idims == ("i", "j")
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = Coordinates([[lat, lon]], dims=["lat,lon"])
        assert c.dims == ("lat,lon",)
        assert c.udims == ("lat", "lon")
        assert c.idims == ("i", "j")
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

    def test_rotated(self):
        latlon = RotatedCoordinates((3, 4), np.pi / 4, [10, 20], [1.0, 2.0], dims=["lat", "lon"])
        c = Coordinates([latlon])
        assert c.dims == ("lat,lon",)
        assert c.udims == ("lat", "lon")
        assert c.idims == ("i", "j")
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

    def test_mixed(self):
        # stacked
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ["2018-01-01", "2018-01-02"]

        c = Coordinates([[lat, lon], dates], dims=["lat_lon", "time"])
        assert c.dims == ("lat_lon", "time")
        assert c.udims == ("lat", "lon", "time")
        assert c.idims == ("lat_lon", "time")
        assert c.shape == (3, 2)
        assert c.ndim == 2
        assert c.size == 6
        repr(c)

        # stacked, nested dims version
        c = Coordinates([[lat, lon], dates], dims=[["lat", "lon"], "time"])
        assert c.dims == ("lat_lon", "time")
        assert c.udims == ("lat", "lon", "time")
        assert c.idims == ("lat_lon", "time")
        assert c.shape == (3, 2)
        assert c.ndim == 2
        assert c.size == 6
        repr(c)

        # dependent
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        dates = ["2018-01-01", "2018-01-02"]
        c = Coordinates([[lat, lon], dates], dims=["lat,lon", "time"])
        assert c.dims == ("lat,lon", "time")
        assert c.udims == ("lat", "lon", "time")
        assert c.idims == ("i", "j", "time")
        assert c.shape == (3, 4, 2)
        assert c.ndim == 3
        assert c.size == 24
        repr(c)

        # rotated
        latlon = RotatedCoordinates((3, 4), np.pi / 4, [10, 20], [1.0, 2.0], dims=["lat", "lon"])
        c = Coordinates([latlon, dates], dims=["lat,lon", "time"])
        assert c.dims == ("lat,lon", "time")
        assert c.udims == ("lat", "lon", "time")
        assert c.idims == ("i", "j", "time")
        assert c.shape == (3, 4, 2)
        assert c.ndim == 3
        assert c.size == 24
        repr(c)

    def test_invalid_dims(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ["2018-01-01", "2018-01-02"]

        with pytest.raises(TypeError, match="Invalid dims type"):
            Coordinates([dates], dims="time")

        with pytest.raises(ValueError, match="coords and dims size mismatch"):
            Coordinates(dates, dims=["time"])

        with pytest.raises(ValueError, match="coords and dims size mismatch"):
            Coordinates([lat, lon, dates], dims=["lat_lon", "time"])

        with pytest.raises(ValueError, match="coords and dims size mismatch"):
            Coordinates([[lat, lon], dates], dims=["lat", "lon", "dates"])

        with pytest.raises(ValueError, match="coords and dims size mismatch"):
            Coordinates([lat, lon], dims=["lat_lon"])

        with pytest.raises(ValueError, match="coords and dims size mismatch"):
            Coordinates([[lat, lon]], dims=["lat", "lon"])

        with pytest.raises(ValueError, match="coords and dims size mismatch"):
            Coordinates([lat, lon], dims=["lat_lon"])

        with pytest.raises(ValueError, match="Invalid coordinate values"):
            Coordinates([[lat, lon]], dims=["lat"])

        with pytest.raises(TypeError, match="Cannot get dim for coordinates at position"):
            # this doesn't work because lat and lon are not named BaseCoordinates/xarray objects
            Coordinates([lat, lon])

        with pytest.raises(ValueError, match="Duplicate dimension"):
            Coordinates([lat, lon], dims=["lat", "lat"])

        with pytest.raises(ValueError, match="Duplicate dimension"):
            Coordinates([[lat, lon], lon], dims=["lat_lon", "lat"])

    def test_dims_mismatch(self):
        c1d = ArrayCoordinates1d([0, 1, 2], name="lat")

        with pytest.raises(ValueError, match="Dimension mismatch"):
            Coordinates([c1d], dims=["lon"])

    def test_invalid_coords(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ["2018-01-01", "2018-01-02"]

        with pytest.raises(TypeError, match="Invalid coords"):
            Coordinates({"lat": lat, "lon": lon})

    def test_base_coordinates(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ["2018-01-01", "2018-01-02"]

        c = Coordinates(
            [
                StackedCoordinates([ArrayCoordinates1d(lat, name="lat"), ArrayCoordinates1d(lon, name="lon")]),
                ArrayCoordinates1d(dates, name="time"),
            ]
        )

        assert c.dims == ("lat_lon", "time")
        assert c.shape == (3, 2)

        # TODO default and overridden properties

    def test_grid(self):
        # array
        lat = [0, 1, 2]
        lon = [10, 20, 30, 40]
        dates = ["2018-01-01", "2018-01-02"]

        c = Coordinates.grid(lat=lat, lon=lon, time=dates, dims=["time", "lat", "lon"])
        assert c.dims == ("time", "lat", "lon")
        assert c.udims == ("time", "lat", "lon")
        assert c.shape == (2, 3, 4)
        assert c.ndim == 3
        assert c.size == 24

        # size
        lat = (0, 1, 3)
        lon = (10, 40, 4)
        dates = ("2018-01-01", "2018-01-05", 5)

        c = Coordinates.grid(lat=lat, lon=lon, time=dates, dims=["time", "lat", "lon"])
        assert c.dims == ("time", "lat", "lon")
        assert c.udims == ("time", "lat", "lon")
        assert c.shape == (5, 3, 4)
        assert c.ndim == 3
        assert c.size == 60

        # step
        lat = (0, 1, 0.5)
        lon = (10, 40, 10.0)
        dates = ("2018-01-01", "2018-01-05", "1,D")

        c = Coordinates.grid(lat=lat, lon=lon, time=dates, dims=["time", "lat", "lon"])
        assert c.dims == ("time", "lat", "lon")
        assert c.udims == ("time", "lat", "lon")
        assert c.shape == (5, 3, 4)
        assert c.ndim == 3
        assert c.size == 60

    def test_points(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ["2018-01-01", "2018-01-02", "2018-01-03"]

        c = Coordinates.points(lat=lat, lon=lon, time=dates, dims=["time", "lat", "lon"])
        assert c.dims == ("time_lat_lon",)
        assert c.udims == ("time", "lat", "lon")
        assert c.shape == (3,)
        assert c.ndim == 1
        assert c.size == 3

    def test_grid_points_order(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30, 40]
        dates = ["2018-01-01", "2018-01-02"]

        with pytest.raises(ValueError):
            Coordinates.grid(lat=lat, lon=lon, time=dates, dims=["lat", "lon"])

        with pytest.raises(ValueError):
            Coordinates.grid(lat=lat, lon=lon, dims=["lat", "lon", "time"])

        if sys.version < "3.6":
            with pytest.raises(TypeError):
                Coordinates.grid(lat=lat, lon=lon, time=dates)
        else:
            Coordinates.grid(lat=lat, lon=lon, time=dates)

    def test_from_xarray(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ["2018-01-01", "2018-01-02"]

        c = Coordinates(
            [
                StackedCoordinates([ArrayCoordinates1d(lat, name="lat"), ArrayCoordinates1d(lon, name="lon")]),
                ArrayCoordinates1d(dates, name="time"),
            ]
        )

        # from xarray
        c2 = Coordinates.from_xarray(c.coords)
        assert c2.dims == c.dims
        assert c2.shape == c.shape
        assert isinstance(c2["lat_lon"], StackedCoordinates)
        assert isinstance(c2["time"], Coordinates1d)
        np.testing.assert_equal(c2.coords["lat"].data, np.array(lat, dtype=float))
        np.testing.assert_equal(c2.coords["lon"].data, np.array(lon, dtype=float))
        np.testing.assert_equal(c2.coords["time"].data, np.array(dates).astype(np.datetime64))

        # invalid
        with pytest.raises(TypeError, match="Coordinates.from_xarray expects xarray DataArrayCoordinates"):
            Coordinates.from_xarray([0, 10])

    def test_crs(self):
        lat = ArrayCoordinates1d([0, 1, 2], "lat")
        lon = ArrayCoordinates1d([0, 1, 2], "lon")

        # default
        c = Coordinates([lat, lon])
        assert c.crs == podpac.settings["DEFAULT_CRS"]
        assert set(c.properties.keys()) == {"crs"}

        # crs
        c = Coordinates([lat, lon], crs="EPSG:2193")
        assert c.crs == "EPSG:2193"
        assert set(c.properties.keys()) == {"crs"}

        # proj4
        c = Coordinates([lat, lon], crs="EPSG:2193")
        assert c.crs == "EPSG:2193"
        assert set(c.properties.keys()) == {"crs"}

        c = Coordinates([lat, lon], crs="+proj=merc +lat_ts=56.5 +ellps=GRS80")
        assert c.crs == "+proj=merc +lat_ts=56.5 +ellps=GRS80"
        assert set(c.properties.keys()) == {"crs"}

        # with vunits
        c = Coordinates([lat, lon], crs="+proj=merc +lat_ts=56.5 +ellps=GRS80 +vunits=ft")
        assert c.crs == "+proj=merc +lat_ts=56.5 +ellps=GRS80 +vunits=ft"
        assert set(c.properties.keys()) == {"crs"}

        # invalid
        with pytest.raises(pyproj.crs.CRSError):
            Coordinates([lat, lon], crs="abcd")

    def test_crs_with_vertical_units(self):

        alt = ArrayCoordinates1d([0, 1, 2], name="alt")

        c = Coordinates([alt], crs="+proj=merc +vunits=us-ft")
        assert set(c.properties.keys()) == {"crs"}

        # with crs
        ct = c.transform("+proj=merc +vunits=m")
        np.testing.assert_array_almost_equal(ct["alt"].coordinates, 0.30480061 * c['alt'].coordinates)

        # invalid
        with pytest.raises(ValueError):
            Coordinates([alt], crs="EPSG:2193")

    def test_ctype(self):
        # assign
        lat = ArrayCoordinates1d([0, 1, 2])
        lon = ArrayCoordinates1d([0, 1, 2])

        c = Coordinates([lat, lon], dims=["lat", "lon"], ctype="left")
        assert c["lat"].ctype == "left"
        assert c["lon"].ctype == "left"

        # don't overwrite
        lat = ArrayCoordinates1d([0, 1, 2], ctype="right")
        lon = ArrayCoordinates1d([0, 1, 2])

        c = Coordinates([lat, lon], dims=["lat", "lon"], ctype="left")
        assert c["lat"].ctype == "right"
        assert c["lon"].ctype == "left"


class TestCoordinatesSerialization(object):
    def test_definition(self):
        # this tests array coordinates, uniform coordinates, and stacked coordinates
        c = Coordinates(
            [[[0, 1, 2], [10, 20, 30]], ["2018-01-01", "2018-01-02"], crange(0, 10, 0.5)],
            dims=["lat_lon", "time", "alt"],
            crs="+proj=merc +vunits=us-ft"
        )
        d = c.definition
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)
        c2 = Coordinates.from_definition(d)
        assert c2 == c

    def test_definition_dependent(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = Coordinates([[lat, lon]], dims=["lat,lon"])
        d = c.definition
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)
        c2 = Coordinates.from_definition(d)
        assert c2 == c

    def test_definition_rotated(self):
        latlon = RotatedCoordinates((3, 4), np.pi / 4, [10, 20], [1.0, 2.0], dims=["lat", "lon"])
        c = Coordinates([latlon])
        d = c.definition
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)
        c2 = Coordinates.from_definition(d)
        assert c2 == c

    def test_definition_properties(self):
        lat = ArrayCoordinates1d([0, 1, 2], "lat")
        lon = ArrayCoordinates1d([0, 1, 2], "lon")

        # default
        c = Coordinates([lat, lon], crs="EPSG:2193")
        d = c.definition
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)
        c2 = Coordinates.from_definition(d)
        assert c2 == c
        assert c2.crs == "EPSG:2193"

    def test_from_definition(self):
        d = {
            "coords": [{"name": "lat", "values": [0, 1, 2]}, {"name": "lon", "start": 0, "stop": 10, "size": 6}],
            "crs": "EPSG:2193",
        }

        c = Coordinates.from_definition(d)
        assert c.dims == ("lat", "lon")
        assert c.crs == "EPSG:2193"
        assert_equal(c["lat"].coordinates, [0, 1, 2])
        assert_equal(c["lon"].coordinates, [0, 2, 4, 6, 8, 10])

    def test_invalid_definition(self):
        with pytest.raises(TypeError, match="Could not parse coordinates definition"):
            Coordinates.from_definition([0, 1, 2])

        with pytest.raises(ValueError, match="Could not parse coordinates definition"):
            Coordinates.from_definition({"data": [0, 1, 2]})

        with pytest.raises(TypeError, match="Could not parse coordinates definition"):
            Coordinates.from_definition({"coords": {}})

        with pytest.raises(ValueError, match="Could not parse coordinates definition item"):
            Coordinates.from_definition({"coords": [{}]})

    def test_json(self):
        c = Coordinates(
            [[[0, 1, 2], [10, 20, 30]], ["2018-01-01", "2018-01-02"], crange(0, 10, 0.5)],
            dims=["lat_lon", "time", "alt"],
            crs="+proj=merc +vunits=us-ft"
        )

        s = c.json

        json.loads(s)

        c2 = Coordinates.from_json(s)
        assert c2 == c

    def test_from_url(self):
        crds = Coordinates([[41, 40], [-71, -70], "2018-05-19"], dims=["lat", "lon", "time"])
        with pytest.warns(UserWarning, match="transformation of coordinate segment lengths not yet implemented"):
            crds2 = crds.transform("EPSG:3857")

        url = (
            r"http://testwms/?map=map&&service=WMS&request=GetMap&layers=layer&styles=&format=image%2Fpng"
            r"&transparent=true&version={version}&transparency=true&width=256&height=256&srs=EPSG%3A{epsg}"
            r"&bbox={},{},{},{}&time={}"
        )

        version = "1.1.1"
        for cc, epsg in zip([crds, crds2], ["3857", "4326"]):
            c = Coordinates.from_url(
                url.format(
                    crds2.bounds["lon"].min(),
                    crds2.bounds["lat"].min(),
                    crds2.bounds["lon"].max(),
                    crds2.bounds["lat"].max(),
                    crds2.bounds["time"][0],
                    version=version,
                    epsg=epsg,
                )
            )
            for d in crds.dims:
                assert np.allclose(c.bounds[d].astype(float), crds2.bounds[d].astype(float))

        version = "1.3"
        for cc, epsg in zip([crds, crds2], ["3857", "4326"]):
            c = Coordinates.from_url(
                url.format(
                    crds2.bounds["lat"].min(),
                    crds2.bounds["lon"].min(),
                    crds2.bounds["lat"].max(),
                    crds2.bounds["lon"].max(),
                    crds2.bounds["time"][0],
                    version=version,
                    epsg=epsg,
                )
            )
            for d in crds.dims:
                assert np.allclose(c.bounds[d].astype(float), crds2.bounds[d].astype(float))


class TestCoordinatesProperties(object):
    def test_xarray_coords(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30, 40]
        dates = ["2018-01-01", "2018-01-02"]

        c = Coordinates(
            [
                ArrayCoordinates1d(lat, name="lat"),
                ArrayCoordinates1d(lon, name="lon"),
                ArrayCoordinates1d(dates, name="time"),
            ]
        )

        assert isinstance(c.coords, xr.core.coordinates.DataArrayCoordinates)
        assert c.coords.dims == ("lat", "lon", "time")
        np.testing.assert_equal(c.coords["lat"].data, np.array(lat, dtype=float))
        np.testing.assert_equal(c.coords["lon"].data, np.array(lon, dtype=float))
        np.testing.assert_equal(c.coords["time"].data, np.array(dates).astype(np.datetime64))

    def test_xarray_coords_stacked(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ["2018-01-01", "2018-01-02"]

        c = Coordinates(
            [
                StackedCoordinates([ArrayCoordinates1d(lat, name="lat"), ArrayCoordinates1d(lon, name="lon")]),
                ArrayCoordinates1d(dates, name="time"),
            ]
        )

        assert isinstance(c.coords, xr.core.coordinates.DataArrayCoordinates)
        assert c.coords.dims == ("lat_lon", "time")
        np.testing.assert_equal(c.coords["lat"].data, np.array(lat, dtype=float))
        np.testing.assert_equal(c.coords["lon"].data, np.array(lon, dtype=float))
        np.testing.assert_equal(c.coords["time"].data, np.array(dates).astype(np.datetime64))

    def test_xarray_coords_dependent(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        dates = ["2018-01-01", "2018-01-02"]

        c = Coordinates([DependentCoordinates([lat, lon], dims=["lat", "lon"]), ArrayCoordinates1d(dates, name="time")])

        assert isinstance(c.coords, xr.core.coordinates.DataArrayCoordinates)
        assert c.coords.dims == ("i", "j", "time")
        np.testing.assert_equal(c.coords["i"].data, np.arange(3))
        np.testing.assert_equal(c.coords["j"].data, np.arange(4))
        np.testing.assert_equal(c.coords["lat"].data, lat)
        np.testing.assert_equal(c.coords["lon"].data, lon)
        np.testing.assert_equal(c.coords["time"].data, np.array(dates).astype(np.datetime64))

    def test_bounds(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ["2018-01-01", "2018-01-02"]

        c = Coordinates([[lat, lon], dates], dims=["lat_lon", "time"])
        bounds = c.bounds
        assert isinstance(bounds, dict)
        assert set(bounds.keys()) == set(c.udims)
        assert_equal(bounds["lat"], c["lat"].bounds)
        assert_equal(bounds["lon"], c["lon"].bounds)
        assert_equal(bounds["time"], c["time"].bounds)

    def test_area_bounds(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ["2018-01-01", "2018-01-02"]

        c = Coordinates([[lat, lon], dates], dims=["lat_lon", "time"])
        area_bounds = c.area_bounds
        assert isinstance(area_bounds, dict)
        assert set(area_bounds.keys()) == set(c.udims)
        assert_equal(area_bounds["lat"], c["lat"].area_bounds)
        assert_equal(area_bounds["lon"], c["lon"].area_bounds)
        assert_equal(area_bounds["time"], c["time"].area_bounds)


class TestCoordinatesDict(object):
    coords = Coordinates([[[0, 1, 2], [10, 20, 30]], ["2018-01-01", "2018-01-02"]], dims=["lat_lon", "time"])

    def test_keys(self):
        assert [dim for dim in self.coords.keys()] == ["lat_lon", "time"]

    def test_values(self):
        assert [c for c in self.coords.values()] == [self.coords["lat_lon"], self.coords["time"]]

    def test_items(self):
        assert [dim for dim, c in self.coords.items()] == ["lat_lon", "time"]
        assert [c for dim, c in self.coords.items()] == [self.coords["lat_lon"], self.coords["time"]]

    def test_iter(self):
        assert [dim for dim in self.coords] == ["lat_lon", "time"]

    def test_getitem(self):
        lat = ArrayCoordinates1d([0, 1, 2], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02"], name="time")
        lat_lon = StackedCoordinates([lat, lon])
        coords = Coordinates([lat_lon, time])

        assert coords["lat_lon"] == lat_lon
        assert coords["time"] == time
        assert coords["lat"] == lat
        assert coords["lon"] == lon

        with pytest.raises(KeyError, match="Dimension 'alt' not found in Coordinates"):
            coords["alt"]

    def test_get(self):
        assert self.coords.get("lat_lon") is self.coords["lat_lon"]
        assert self.coords.get("lat") is self.coords["lat"]
        assert self.coords.get("alt") == None
        assert self.coords.get("alt", "DEFAULT") == "DEFAULT"

    def test_setitem(self):
        coords = deepcopy(self.coords)

        coords["time"] = [1, 2, 3]
        coords["time"] = ArrayCoordinates1d([1, 2, 3])
        coords["time"] = ArrayCoordinates1d([1, 2, 3], name="time")
        coords["time"] = Coordinates([[1, 2, 3]], dims=["time"])

        # coords['lat_lon'] = [np.linspace(0, 10, 5), np.linspace(0, 10, 5)]
        coords["lat_lon"] = clinspace((0, 1), (10, 20), 5)
        coords["lat_lon"] = (np.linspace(0, 10, 5), np.linspace(0, 10, 5))
        coords["lat_lon"] = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lat_lon"])

        # update a single stacked dimension
        coords["lat"] = np.linspace(5, 20, 5)
        assert coords["lat"] == ArrayCoordinates1d(np.linspace(5, 20, 5), name="lat")

        coords = deepcopy(self.coords)
        coords["lat_lon"]["lat"] = np.linspace(5, 20, 3)
        assert coords["lat"] == ArrayCoordinates1d(np.linspace(5, 20, 3), name="lat")

        with pytest.raises(KeyError, match="Cannot set dimension"):
            coords["alt"] = ArrayCoordinates1d([1, 2, 3], name="alt")

        with pytest.raises(ValueError, match="Dimension mismatch"):
            coords["alt"] = ArrayCoordinates1d([1, 2, 3], name="lat")

        with pytest.raises(ValueError, match="Dimension mismatch"):
            coords["time"] = ArrayCoordinates1d([1, 2, 3], name="alt")

        with pytest.raises(KeyError, match="not found in Coordinates"):
            coords["lat_lon"] = Coordinates([(np.linspace(0, 10, 5), np.linspace(0, 10, 5))], dims=["lon_lat"])

        with pytest.raises(ValueError, match="Dimension mismatch"):
            coords["lat_lon"] = clinspace((0, 1), (10, 20), 5, name="lon_lat")

        with pytest.raises(ValueError, match="Size mismatch"):
            coords["lat"] = np.linspace(5, 20, 5)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            coords["lat"] = clinspace(0, 10, 3, name="lon")

    def test_delitem(self):
        # unstacked
        coords = deepcopy(self.coords)
        del coords["time"]
        assert coords.dims == ("lat_lon",)

        # stacked
        coords = deepcopy(self.coords)
        del coords["lat_lon"]
        assert coords.dims == ("time",)

        # missing
        coords = deepcopy(self.coords)
        with pytest.raises(KeyError, match="Cannot delete dimension 'alt' in Coordinates"):
            del coords["alt"]

        # part of stacked dimension
        coords = deepcopy(self.coords)
        with pytest.raises(KeyError, match="Cannot delete dimension 'lat' in Coordinates"):
            del coords["lat"]

    def test_update(self):
        # add a new dimension
        coords = deepcopy(self.coords)
        c = Coordinates([[100, 200, 300]], dims=["alt"])
        coords.update(c)
        assert coords.dims == ("lat_lon", "time", "alt")
        assert coords["lat_lon"] == self.coords["lat_lon"]
        assert coords["time"] == self.coords["time"]
        assert coords["alt"] == c["alt"]

        # overwrite a dimension
        coords = deepcopy(self.coords)
        c = Coordinates([[100, 200, 300]], dims=["time"])
        coords.update(c)
        assert coords.dims == ("lat_lon", "time")
        assert coords["lat_lon"] == self.coords["lat_lon"]
        assert coords["time"] == c["time"]

        # overwrite a stacked dimension
        coords = deepcopy(self.coords)
        c = Coordinates([clinspace((0, 1), (10, 20), 5)], dims=["lat_lon"])
        coords.update(c)
        assert coords.dims == ("lat_lon", "time")
        assert coords["lat_lon"] == c["lat_lon"]
        assert coords["time"] == self.coords["time"]

        # mixed
        coords = deepcopy(self.coords)
        c = Coordinates([clinspace((0, 1), (10, 20), 5), [100, 200, 300]], dims=["lat_lon", "alt"])
        coords.update(c)
        assert coords.dims == ("lat_lon", "time", "alt")
        assert coords["lat_lon"] == c["lat_lon"]
        assert coords["time"] == self.coords["time"]
        assert coords["alt"] == c["alt"]

        # invalid
        coords = deepcopy(self.coords)
        with pytest.raises(TypeError, match="Cannot update Coordinates with object of type"):
            coords.update({"time": [1, 2, 3]})

        # duplicate dimension
        coords = deepcopy(self.coords)
        c = Coordinates([[0, 0.1, 0.2]], dims=["lat"])
        with pytest.raises(ValueError, match="Duplicate dimension 'lat'"):
            coords.update(c)

    def test_len(self):
        assert len(self.coords) == 2


class TestCoordinatesIndexing(object):
    def test_get_index(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3, 5], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"], name="time")
        c = Coordinates([lat, lon, time])

        I = [2, 1, 3]
        J = slice(1, 3)
        K = 1

        # full
        c2 = c[I, J, K]
        assert isinstance(c2, Coordinates)
        assert c2.shape == (3, 2, 1)
        assert c2.dims == c.dims
        assert_equal(c2["lat"].coordinates, c["lat"].coordinates[I])
        assert_equal(c2["lon"].coordinates, c["lon"].coordinates[J])
        assert_equal(c2["time"].coordinates, c["time"].coordinates[K])

        # partial
        c2 = c[I, J]
        assert isinstance(c2, Coordinates)
        assert c2.shape == (3, 2, 3)
        assert c2.dims == c.dims
        assert_equal(c2["lat"].coordinates, c["lat"].coordinates[I])
        assert_equal(c2["lon"].coordinates, c["lon"].coordinates[J])
        assert_equal(c2["time"].coordinates, c["time"].coordinates)

        c2 = c[I]
        assert isinstance(c2, Coordinates)
        assert c2.shape == (3, 4, 3)
        assert c2.dims == c.dims
        assert_equal(c2["lat"].coordinates, c["lat"].coordinates[I])
        assert_equal(c2["lon"].coordinates, c["lon"].coordinates)
        assert_equal(c2["time"].coordinates, c["time"].coordinates)

    def test_get_index_stacked(self):
        lat = [0, 1, 2, 3, 4]
        lon = [10, 20, 30, 40, 50]
        dates = ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"]

        c = Coordinates(
            [
                StackedCoordinates([ArrayCoordinates1d(lat, name="lat"), ArrayCoordinates1d(lon, name="lon")]),
                ArrayCoordinates1d(dates, name="time"),
            ]
        )

        I = [2, 1, 3]
        J = slice(1, 3)

        # full
        c2 = c[I, J]
        assert isinstance(c2, Coordinates)
        assert c2.shape == (3, 2)
        assert c2.dims == c.dims
        assert_equal(c2["lat"].coordinates, c["lat"].coordinates[I])
        assert_equal(c2["lon"].coordinates, c["lon"].coordinates[I])
        assert_equal(c2["time"].coordinates, c["time"].coordinates[J])

        # partial
        c2 = c[I]
        assert isinstance(c2, Coordinates)
        assert c2.shape == (3, 4)
        assert c2.dims == c.dims
        assert_equal(c2["lat"].coordinates, c["lat"].coordinates[I])
        assert_equal(c2["lon"].coordinates, c["lon"].coordinates[I])
        assert_equal(c2["time"].coordinates, c["time"].coordinates)

    def test_get_index_dependent(self):
        lat = np.linspace(0, 1, 20).reshape((5, 4))
        lon = np.linspace(10, 20, 20).reshape((5, 4))
        dates = ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"]

        c = Coordinates([DependentCoordinates([lat, lon], dims=["lat", "lon"]), ArrayCoordinates1d(dates, name="time")])

        I = [2, 1, 3]
        J = slice(1, 3)
        K = 1

        # full
        c2 = c[I, J, K]
        assert isinstance(c2, Coordinates)
        assert c2.shape == (3, 2, 1)
        assert c2.dims == c.dims
        assert_equal(c2["lat"].coordinates, c["lat"].coordinates[I, J])
        assert_equal(c2["lon"].coordinates, c["lon"].coordinates[I, J])
        assert_equal(c2["time"].coordinates, c["time"].coordinates[K])

        # partial
        c2 = c[I, J]
        assert isinstance(c2, Coordinates)
        assert c2.shape == (3, 2, 4)
        assert c2.dims == c.dims
        assert_equal(c2["lat"].coordinates, c["lat"].coordinates[I, J])
        assert_equal(c2["lon"].coordinates, c["lon"].coordinates[I, J])
        assert_equal(c2["time"].coordinates, c["time"].coordinates)

        c2 = c[I]
        assert isinstance(c2, Coordinates)
        assert c2.shape == (3, 4, 4)
        assert c2.dims == c.dims
        assert_equal(c2["lat"].coordinates, c["lat"].coordinates[I])
        assert_equal(c2["lon"].coordinates, c["lon"].coordinates[I])
        assert_equal(c2["time"].coordinates, c["time"].coordinates)

    def test_get_index_properties(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3, 5], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03"], name="time")
        c = Coordinates([lat, lon, time], crs="EPSG:2193")

        I = [2, 1, 3]
        J = slice(1, 3)
        K = 1

        c2 = c[I, J, K]
        assert c2.crs == c.crs


class TestCoordinatesMethods(object):
    coords = Coordinates(
        [[[0, 1, 2], [10, 20, 30]], ["2018-01-01", "2018-01-02"], 10],
        dims=["lat_lon", "time", "alt"],
        crs="+proj=merc +vunits=us-ft")

    def test_drop(self):
        # drop one existing dimension
        c1 = self.coords.drop("time")
        c2 = self.coords.udrop("time")
        assert c1.dims == ("lat_lon", "alt")
        assert c2.dims == ("lat_lon", "alt")

        # drop multiple existing dimensions
        c1 = self.coords.drop(["time", "alt"])
        c2 = self.coords.udrop(["time", "alt"])
        assert c1.dims == ("lat_lon",)
        assert c2.dims == ("lat_lon",)

        # drop all dimensions
        c1 = self.coords.drop(self.coords.dims)
        c2 = self.coords.udrop(self.coords.udims)
        assert c1.dims == ()
        assert c2.dims == ()

        # drop no dimensions
        c1 = self.coords.drop([])
        c2 = self.coords.udrop([])
        assert c1.dims == ("lat_lon", "time", "alt")
        assert c2.dims == ("lat_lon", "time", "alt")

        # drop a missing dimension
        c = self.coords.drop("alt")
        with pytest.raises(KeyError, match="Dimension 'alt' not found in Coordinates with dims"):
            c1 = c.drop("alt")
        with pytest.raises(KeyError, match="Dimension 'alt' not found in Coordinates with udims"):
            c2 = c.udrop("alt")

        c1 = c.drop("alt", ignore_missing=True)
        c2 = c.udrop("alt", ignore_missing=True)
        assert c1.dims == ("lat_lon", "time")
        assert c2.dims == ("lat_lon", "time")

        # drop a stacked dimension: drop works but udrop gives an exception
        c1 = self.coords.drop("lat_lon")
        assert c1.dims == ("time", "alt")

        with pytest.raises(KeyError, match="Dimension 'lat_lon' not found in Coordinates with udims"):
            c2 = self.coords.udrop("lat_lon")

        # drop part of a stacked dimension: drop gives exception but udrop does not
        # note: two udrop cases: 'lat_lon' -> 'lon' and 'lat_lon_alt' -> 'lat_lon'
        with pytest.raises(KeyError, match="Dimension 'lat' not found in Coordinates with dims"):
            c1 = self.coords.drop("lat")

        c2 = self.coords.udrop("lat")
        assert c2.dims == ("lon", "time", "alt")

        coords = Coordinates([[[0, 1], [10, 20], [100, 300]]], dims=["lat_lon_alt"], crs="+proj=merc +vunits=us-ft")
        c2 = coords.udrop("alt")
        assert c2.dims == ("lat_lon",)

    def test_drop_invalid(self):
        with pytest.raises(TypeError, match="Invalid drop dimension type"):
            self.coords.drop(2)

        with pytest.raises(TypeError, match="Invalid drop dimension type"):
            self.coords.udrop(2)

        with pytest.raises(TypeError, match="Invalid drop dimension type"):
            self.coords.drop([2, 3])

        with pytest.raises(TypeError, match="Invalid drop dimension type"):
            self.coords.udrop([2, 3])

    def test_drop_properties(self):
        coords = Coordinates(
            [[[0, 1, 2], [10, 20, 30]], ["2018-01-01", "2018-01-02"], 10],
            dims=["lat_lon", "time", "alt"],
            crs="+proj=merc +vunits=us-ft")

        c1 = coords.drop("time")
        c2 = coords.udrop("time")

        # check properties
        assert c1.crs == "+proj=merc +vunits=us-ft"
        assert c2.crs == "+proj=merc +vunits=us-ft"

    def test_unique(self):
        # unstacked (numerical, datetime, and empty)
        c = Coordinates(
            [[2, 1, 0, 1], ["2018-01-01", "2018-01-02", "2018-01-01"], []],
            dims=["lat", "time", "alt"],
            crs="+proj=merc +vunits=us-ft")
        c2 = c.unique()
        assert_equal(c2["lat"].coordinates, [0, 1, 2])
        assert_equal(c2["time"].coordinates, [np.datetime64("2018-01-01"), np.datetime64("2018-01-02")])
        assert_equal(c2["alt"].coordinates, [])

        # return indices
        c = Coordinates(
            [[2, 1, 0, 1], ["2018-01-01", "2018-01-02", "2018-01-01"], []],
            dims=["lat", "time", "alt"],
            crs="+proj=merc +vunits=us-ft")
        c2, I = c.unique(return_indices=True)
        assert_equal(c2["lat"].coordinates, [0, 1, 2])
        assert_equal(c2["time"].coordinates, [np.datetime64("2018-01-01"), np.datetime64("2018-01-02")])
        assert_equal(c2["alt"].coordinates, [])
        assert c2 == c[I]

        # stacked
        lat_lon = [(0, 0), (0, 1), (0, 2), (0, 2), (1, 0), (1, 1), (1, 1)]  # duplicate  # duplicate
        lat, lon = zip(*lat_lon)
        c = Coordinates([[lat, lon]], dims=["lat_lon"])
        c2 = c.unique()
        assert_equal(c2["lat"].coordinates, [0.0, 0.0, 0.0, 1.0, 1.0])
        assert_equal(c2["lon"].coordinates, [0.0, 1.0, 2.0, 0.0, 1.0])

    def test_unique_properties(self):
        c = Coordinates(
            [[2, 1, 0, 1], ["2018-01-01", "2018-01-02", "2018-01-01"], []],
            dims=["lat", "time", "alt"],
            crs="+proj=merc +vunits=us-ft")
        c2 = c.unique()

        # check properties
        assert c2.crs == "+proj=merc +vunits=us-ft"

    def test_unstack(self):
        c1 = Coordinates([[[0, 1], [10, 20], [100, 300]]], dims=["lat_lon_alt"], crs="+proj=merc +vunits=us-ft")
        c2 = c1.unstack()
        assert c1.dims == ("lat_lon_alt",)
        assert c2.dims == ("lat", "lon", "alt")
        assert c1["lat"] == c2["lat"]
        assert c1["lon"] == c2["lon"]
        assert c1["alt"] == c2["alt"]

        # mixed
        c1 = Coordinates([[[0, 1], [10, 20]], [100, 200, 300]], dims=["lat_lon", "alt"], crs="+proj=merc +vunits=us-ft")
        c2 = c1.unstack()
        assert c1.dims == ("lat_lon", "alt")
        assert c2.dims == ("lat", "lon", "alt")
        assert c1["lat"] == c2["lat"]
        assert c1["lon"] == c2["lon"]
        assert c1["alt"] == c2["alt"]

    def test_unstack_properties(self):
        c1 = Coordinates([[[0, 1], [10, 20], [100, 300]]], dims=["lat_lon_alt"], crs="+proj=merc +vunits=us-ft")
        c2 = c1.unstack()

        # check properties
        assert c2.crs == "+proj=merc +vunits=us-ft"

    def test_iterchunks(self):
        c = Coordinates(
            [clinspace(0, 1, 100), clinspace(0, 1, 200), ["2018-01-01", "2018-01-02"]], dims=["lat", "lon", "time"]
        )

        for chunk in c.iterchunks(shape=(10, 10, 10)):
            assert isinstance(chunk, Coordinates)
            assert chunk.dims == c.dims
            assert chunk.shape == (10, 10, 2)

        for chunk, slices in c.iterchunks(shape=(10, 10, 10), return_slices=True):
            assert isinstance(chunk, Coordinates)
            assert chunk.dims == c.dims
            assert chunk.shape == (10, 10, 2)

            assert isinstance(slices, tuple)
            assert len(slices) == 3
            assert all(isinstance(slc, slice) for slc in slices)

    def test_iterchunks_properties(self):
        c = Coordinates(
            [clinspace(0, 1, 100), clinspace(0, 1, 200), ["2018-01-01", "2018-01-02"]],
            dims=["lat", "lon", "time"],
            crs="EPSG:2193",
        )

        for chunk in c.iterchunks(shape=(10, 10, 10)):
            # check properties
            assert chunk.crs == "EPSG:2193"

    def test_tranpose(self):
        c = Coordinates([[0, 1], [10, 20], ["2018-01-01", "2018-01-02"]], dims=["lat", "lon", "time"])

        # transpose
        t = c.transpose("lon", "lat", "time")
        assert c.dims == ("lat", "lon", "time")
        assert t.dims == ("lon", "lat", "time")

        # default: full transpose
        t = c.transpose()
        assert c.dims == ("lat", "lon", "time")
        assert t.dims == ("time", "lon", "lat")

        # in place
        t = c.transpose("lon", "lat", "time", in_place=False)
        assert c.dims == ("lat", "lon", "time")
        assert t.dims == ("lon", "lat", "time")

        c.transpose("lon", "lat", "time", in_place=True)
        assert c.dims == ("lon", "lat", "time")

        with pytest.raises(ValueError, match="Invalid transpose dimensions"):
            c.transpose("lon", "lat")

    def test_transform(self):
        c = Coordinates(
            [[0, 1], [10, 20, 30, 40], ["2018-01-01", "2018-01-02"]],
            dims=["lat", "lon", "time"]
        )

        # transform
        with pytest.warns(UserWarning, match="transformation of coordinate segment lengths not yet implemented"):
            t = c.transform("EPSG:2193")
        assert c.crs == "EPSG:4326"
        assert t.crs == "EPSG:2193"
        assert round(t["lat"].coordinates[0, 0]) == 29995930.0

        # no transform needed
        t = c.transform("EPSG:4326")
        assert c.crs == "EPSG:4326"
        assert t.crs == "EPSG:4326"
        assert t is not c
        assert t == c

        # support proj4 strings
        proj = "+proj=merc +lat_ts=56.5 +ellps=GRS80"
        with pytest.warns(UserWarning, match="transformation of coordinate segment lengths not yet implemented"):
            t = c.transform(proj)
        assert c.crs == "EPSG:4326"
        assert t.crs == proj
        assert round(t["lat"].coordinates[0, 0]) == 0.0

        # no parameter
        with pytest.raises(TypeError, match="transform requires crs argument"):
            c.transform()

    def test_transform_stacked(self):
        c = Coordinates(
            [[[0, 1], [10, 20]], ["2018-01-01", "2018-01-02", "2018-01-03"]],
            dims=["lat_lon", "time"])

        proj = "+proj=merc +lat_ts=56.5 +ellps=GRS80"
        t = c.transform(proj)
        assert c.crs == "EPSG:4326"
        assert t.crs == proj
        assert round(t["lat"].coordinates[0]) == 0.0

    def test_transform_alt(self):
        c = Coordinates(
            [[0, 1], [10, 20, 30, 40], ["2018-01-01", "2018-01-02"], [100, 200, 300]],
            dims=["lat", "lon", "time", 'alt'],
            crs="+proj=merc +vunits=us-ft"
        )

        proj = "+proj=merc +vunits=m"
        with pytest.warns(UserWarning, match="transformation of coordinate segment lengths not yet implemented"):
            t = c.transform(proj)
        assert c.crs == "+proj=merc +vunits=us-ft"
        assert t.crs == "+proj=merc +vunits=m"
        np.testing.assert_array_almost_equal(t['lat'].coordinates[0], [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(t['lat'].coordinates[1], [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(t['lon'].coordinates[:, 0], [10.0, 10.0])
        np.testing.assert_array_almost_equal(t['lon'].coordinates[:, 1], [20.0, 20.0])
        np.testing.assert_array_almost_equal(t['lon'].coordinates[:, 2], [30.0, 30.0])
        np.testing.assert_array_almost_equal(t['lon'].coordinates[:, 3], [40.0, 40.0])
        assert t['time'] == c['time']
        np.testing.assert_array_almost_equal(t['alt'].coordinates, 0.30480061 * c['alt'].coordinates)

    def test_select_single(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], name="time")
        c = Coordinates([lat, lon, time])

        # single dimension
        s = c.select({"lat": [0.5, 2.5]})
        assert isinstance(s, Coordinates)
        assert s.dims == c.dims
        assert s["lat"] == c["lat"][1:3]
        assert s["lon"] == c["lon"]
        assert s["time"] == c["time"]

        s, I = c.select({"lat": [0.5, 2.5]}, return_indices=True)
        assert isinstance(s, Coordinates)
        assert s.dims == c.dims
        assert s["lat"] == c["lat"][1:3]
        assert s["lon"] == c["lon"]
        assert s["time"] == c["time"]
        assert s == c[I]

        # a different single dimension
        s = c.select({"lon": [5, 25]})
        assert isinstance(s, Coordinates)
        assert s.dims == c.dims
        assert s["lat"] == c["lat"]
        assert s["lon"] == c["lon"][0:2]
        assert s["time"] == c["time"]

        s, I = c.select({"lon": [5, 25]}, return_indices=True)
        assert isinstance(s, Coordinates)
        assert s.dims == c.dims
        assert s["lat"] == c["lat"]
        assert s["lon"] == c["lon"][0:2]
        assert s["time"] == c["time"]
        assert s == c[I]

        # outer
        s = c.select({"lat": [0.5, 2.5]}, outer=True)
        assert isinstance(s, Coordinates)
        assert s.dims == c.dims
        assert s["lat"] == c["lat"][0:4]
        assert s["lon"] == c["lon"]
        assert s["time"] == c["time"]

        s, I = c.select({"lat": [0.5, 2.5]}, outer=True, return_indices=True)
        assert isinstance(s, Coordinates)
        assert s.dims == c.dims
        assert s["lat"] == c["lat"][0:4]
        assert s["lon"] == c["lon"]
        assert s["time"] == c["time"]
        assert s == c[I]

        # no matching dimension
        s = c.select({"alt": [0, 10]})
        assert s == c

        s, I = c.select({"alt": [0, 10]}, return_indices=True)
        assert s == c[I]
        assert s == c

    def test_select_multiple(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3, 4, 5], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40, 50, 60], name="lon")
        c = Coordinates([lat, lon])

        s = c.select({"lat": [0.5, 3.5], "lon": [25, 55]})
        assert isinstance(s, Coordinates)
        assert s.dims == c.dims
        assert s["lat"] == c["lat"][1:4]
        assert s["lon"] == c["lon"][2:5]

        s, I = c.select({"lat": [0.5, 3.5], "lon": [25, 55]}, return_indices=True)
        assert isinstance(s, Coordinates)
        assert s.dims == c.dims
        assert s["lat"] == c["lat"][1:4]
        assert s["lon"] == c["lon"][2:5]
        assert s == c[I]

    def test_select_properties(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40], name="lon")
        time = ArrayCoordinates1d(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"], name="time")
        c = Coordinates([lat, lon, time], crs="EPSG:2193")

        s = c.select({"lat": [0.5, 2.5]})

        # check properties
        assert s.crs == "EPSG:2193"

    def test_intersect(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3, 4, 5], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40, 50, 60], name="lon")
        c = Coordinates([lat, lon])

        other_lat = ArrayCoordinates1d([0.5, 2.5, 3.5], name="lat")
        other_lon = ArrayCoordinates1d([25, 35, 55], name="lon")

        other = Coordinates([other_lat, other_lon])
        c2 = c.intersect(other)
        assert isinstance(c2, Coordinates)
        assert c2.dims == c.dims
        assert c2["lat"] == c["lat"][1:4]
        assert c2["lon"] == c["lon"][2:5]

        c2, I = c.intersect(other, return_indices=True)
        assert isinstance(c2, Coordinates)
        assert c2.dims == c.dims
        assert c2["lat"] == c["lat"][1:4]
        assert c2["lon"] == c["lon"][2:5]
        assert c2 == c[I]

        other = Coordinates([other_lat, other_lon])
        c2 = c.intersect(other, outer=True)
        assert isinstance(c2, Coordinates)
        assert c2.dims == c.dims
        assert c2["lat"] == c["lat"][0:5]
        assert c2["lon"] == c["lon"][1:6]

        # missing dimension
        other = Coordinates([other_lat])
        c2 = c.intersect(other)
        assert isinstance(c2, Coordinates)
        assert c2.dims == c.dims
        assert c2["lat"] == c["lat"][1:4]
        assert c2["lon"] == c["lon"]

        other = Coordinates([other_lon])
        c2 = c.intersect(other)
        assert isinstance(c2, Coordinates)
        assert c2.dims == c.dims
        assert c2["lat"] == c["lat"]
        assert c2["lon"] == c["lon"][2:5]

        # extra dimension
        other = Coordinates([other_lat, other_lon, [0, 1, 2]], dims=["lat", "lon", "time"])
        c2 = c.intersect(other)
        assert isinstance(c2, Coordinates)
        assert c2.dims == c.dims
        assert c2["lat"] == c["lat"][1:4]
        assert c2["lon"] == c["lon"][2:5]

    def test_intersect_dims(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3, 4, 5], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40, 50, 60], name="lon")
        c = Coordinates([lat, lon])

        other_lat = ArrayCoordinates1d([0.5, 2.5, 3.5], name="lat")
        other_lon = ArrayCoordinates1d([25, 35, 55], name="lon")
        other = Coordinates([other_lat, other_lon])

        c2 = c.intersect(other, dims=["lat", "lon"])
        assert isinstance(c2, Coordinates)
        assert c2.dims == c.dims
        assert c2["lat"] == c["lat"][1:4]
        assert c2["lon"] == c["lon"][2:5]

        c2 = c.intersect(other, dims=["lat"])
        assert isinstance(c2, Coordinates)
        assert c2.dims == c.dims
        assert c2["lat"] == c["lat"][1:4]
        assert c2["lon"] == c["lon"]

        c2 = c.intersect(other, dims=["lon"])
        assert isinstance(c2, Coordinates)
        assert c2.dims == c.dims
        assert c2["lat"] == c["lat"]
        assert c2["lon"] == c["lon"][2:5]

    def test_intersect_crs(self):
        # should change the other coordinates crs into the native coordinates crs for intersect
        c = Coordinates(
            [np.linspace(0, 10, 11), np.linspace(0, 10, 11), ["2018-01-01", "2018-01-02"]], dims=["lat", "lon", "time"]
        )
        o = Coordinates(
            [np.linspace(28000000, 29500000, 20), np.linspace(-280000, 400000, 20), ["2018-01-01", "2018-01-02"]],
            dims=["lat", "lon", "time"],
            crs="EPSG:2193",
        )

        with pytest.warns(UserWarning, match="transformation of coordinate segment lengths not yet implemented"):
            c_int = c.intersect(o)
        assert c_int.crs == c.crs
        assert o.crs == "EPSG:2193"  # didn't get changed
        assert np.all(c_int["lat"].bounds == np.array([5.0, 10.0]))
        assert np.all(c_int["lon"].bounds == np.array([4.0, 10.0]))
        assert np.all(c_int["time"] == c["time"])

    def test_intersect_invalid(self):
        lat = ArrayCoordinates1d([0, 1, 2, 3, 4, 5], name="lat")
        lon = ArrayCoordinates1d([10, 20, 30, 40, 50, 60], name="lon")
        c = Coordinates([lat, lon])

        with pytest.raises(TypeError, match="Coordinates cannot be intersected with type"):
            c.intersect(lat)

        with pytest.raises(TypeError, match="Coordinates cannot be intersected with type"):
            c.intersect({"lat": [0, 1]})


class TestCoordinatesSpecial(object):
    def test_repr(self):
        repr(Coordinates([[0, 1], [10, 20], ["2018-01-01", "2018-01-02"]], dims=["lat", "lon", "time"]))
        repr(Coordinates([[[0, 1], [10, 20]], ["2018-01-01", "2018-01-02"]], dims=["lat_lon", "time"]))
        repr(Coordinates([0, 10, []], dims=["lat", "lon", "time"], ctype="point"))
        repr(Coordinates([crange(0, 10, 0.5)], dims=["alt"], crs="+proj=merc +vunits=us-ft"))
        repr(Coordinates([]))
        # TODO dependent coordinates

    def test_eq_ne_hash(self):
        c1 = Coordinates([[[0, 1, 2], [10, 20, 30]], ["2018-01-01", "2018-01-02"]], dims=["lat_lon", "time"])
        c2 = Coordinates([[[0, 2, 1], [10, 20, 30]], ["2018-01-01", "2018-01-02"]], dims=["lat_lon", "time"])
        c3 = Coordinates([[[0, 1, 2], [10, 20, 30]], ["2018-01-01", "2018-01-02"]], dims=["lon_lat", "time"])
        c4 = Coordinates([[[0, 1, 2], [10, 20, 30]], ["2018-01-01"]], dims=["lat_lon", "time"])
        c5 = Coordinates([[0, 1, 2], [10, 20, 30], ["2018-01-01", "2018-01-02"]], dims=["lat", "lon", "time"])

        # eq
        assert c1 == c1
        assert c1 == deepcopy(c1)

        assert not c1 == None
        assert not c1 == c2
        assert not c1 == c3
        assert not c1 == c4
        assert not c1 == c5

        # ne (this only matters in python 2)
        assert not c1 != c1
        assert not c1 != deepcopy(c1)

        assert c1 != None
        assert c1 != c3
        assert c1 != c2
        assert c1 != c4
        assert c1 != c5

        # hash
        assert c1.hash == c1.hash
        assert c1.hash == deepcopy(c1).hash

        assert c1.hash != c3.hash
        assert c1.hash != c2.hash
        assert c1.hash != c4.hash
        assert c1.hash != c5.hash

    def test_eq_ne_hash_ctype(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        c1 = Coordinates([lat, lon], dims=["lat", "lon"])
        c2 = Coordinates([lat, lon], dims=["lat", "lon"], ctype="point")

        # eq
        assert not c1 == c2
        assert c2 == deepcopy(c2)

        # ne (this only matters in python 2)
        assert c1 != c2
        assert not c2 != deepcopy(c2)

        # hash
        assert c1.hash != c2.hash
        assert c2.hash == deepcopy(c2).hash

    def test_eq_ne_hash_crs(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        c1 = Coordinates([lat, lon], dims=["lat", "lon"])
        c2 = Coordinates([lat, lon], dims=["lat", "lon"], crs="EPSG:2193")

        # eq
        assert not c1 == c2
        assert c2 == deepcopy(c2)

        # ne (this only matters in python 2)
        assert c1 != c2
        assert not c2 != deepcopy(c2)

        # hash
        assert c1.hash != c2.hash
        assert c2.hash == deepcopy(c2).hash


class TestCoordinatesFunctions(object):
    def test_merge_dims(self):
        ctime = Coordinates([["2018-01-01", "2018-01-02"]], dims=["time"])
        clatlon = Coordinates([[2, 4, 5], [3, -1, 5]], dims=["lat", "lon"])
        clatlon_stacked = Coordinates([[[2, 4, 5], [3, -1, 5]]], dims=["lat_lon"])
        clat = Coordinates([[2, 4, 5]], dims=["lat"])

        c = merge_dims([clatlon, ctime])
        assert c.dims == ("lat", "lon", "time")

        c = merge_dims([ctime, clatlon])
        assert c.dims == ("time", "lat", "lon")

        c = merge_dims([clatlon_stacked, ctime])
        assert c.dims == ("lat_lon", "time")

        c = merge_dims([ctime, clatlon_stacked])
        assert c.dims == ("time", "lat_lon")

        c = merge_dims([])
        assert c.dims == ()

        with pytest.raises(ValueError, match="Duplicate dimension 'lat'"):
            merge_dims([clatlon, clat])

        with pytest.raises(ValueError, match="Duplicate dimension 'lat'"):
            merge_dims([clatlon_stacked, clat])

        with pytest.raises(TypeError, match="Cannot merge"):
            merge_dims([clat, 0])

    def test_merge_dims_crs(self):
        clat = Coordinates([[2, 4, 5]], dims=["lat"], crs="EPSG:4326")
        clon = Coordinates([[3, -1, 5]], dims=["lon"], crs="EPSG:2193")

        with pytest.raises(ValueError, match="Cannot merge Coordinates"):
            merge_dims([clat, clon])

    def test_concat_and_union(self):
        c1 = Coordinates([[2, 4, 5], [3, -1, 5]], dims=["lat", "lon"])
        c2 = Coordinates([[2, 3], [3, 0], ["2018-01-01", "2018-01-02"]], dims=["lat", "lon", "time"])
        c3 = Coordinates([[[2, 3], [3, 0]]], dims=["lat_lon"])

        c = concat([c1, c2])
        assert c.shape == (5, 5, 2)

        c = union([c1, c2])
        assert c.shape == (4, 4, 2)

        c = concat([])
        assert c.dims == ()

        c = union([])
        assert c.dims == ()

        with pytest.raises(TypeError, match="Cannot concat"):
            concat([c1, [1, 2]])

        with pytest.raises(ValueError, match="Duplicate dimension 'lat' in dims"):
            concat([c1, c3])

    def test_concat_stacked_datetimes(self):
        c1 = Coordinates([[0, 0.5, "2018-01-01"]], dims=["lat_lon_time"])
        c2 = Coordinates([[1, 1.5, "2018-01-02"]], dims=["lat_lon_time"])
        c = concat([c1, c2])
        np.testing.assert_array_equal(c["lat"].coordinates, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(c["lon"].coordinates, np.array([0.5, 1.5]))
        np.testing.assert_array_equal(
            c["time"].coordinates, np.array(["2018-01-01", "2018-01-02"]).astype(np.datetime64)
        )

        c1 = Coordinates([[0, 0.5, "2018-01-01T01:01:01"]], dims=["lat_lon_time"])
        c2 = Coordinates([[1, 1.5, "2018-01-01T01:01:02"]], dims=["lat_lon_time"])
        c = concat([c1, c2])
        np.testing.assert_array_equal(c["lat"].coordinates, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(c["lon"].coordinates, np.array([0.5, 1.5]))
        np.testing.assert_array_equal(
            c["time"].coordinates, np.array(["2018-01-01T01:01:01", "2018-01-01T01:01:02"]).astype(np.datetime64)
        )

    def test_concat_crs(self):
        c1 = Coordinates([[0, 0.5, "2018-01-01"]], dims=["lat_lon_time"], crs="EPSG:4326")
        c2 = Coordinates([[1, 1.5, "2018-01-02"]], dims=["lat_lon_time"], crs="EPSG:2193")

        with pytest.raises(ValueError, match="Cannot concat Coordinates"):
            concat([c1, c2])
