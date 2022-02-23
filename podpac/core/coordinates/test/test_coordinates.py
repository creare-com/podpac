import sys
import json
from copy import deepcopy


import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import xarray as xr
import pyproj

import podpac
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.affine_coordinates import AffineCoordinates
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.cfunctions import crange, clinspace
from podpac.core.coordinates.coordinates import Coordinates
from podpac.core.coordinates.coordinates import concat, union, merge_dims


class TestCoordinateCreation(object):
    def test_empty(self):
        c = Coordinates([])
        assert c.dims == tuple()
        assert c.udims == tuple()
        assert c.xdims == tuple()
        assert c.shape == tuple()
        assert c.ndim == 0
        assert c.size == 0

    def test_single_dim(self):
        # single value
        date = "2018-01-01"

        c = Coordinates([date], dims=["time"])
        assert c.dims == ("time",)
        assert c.udims == ("time",)
        assert c.xdims == ("time",)
        assert c.shape == (1,)
        assert c.ndim == 1
        assert c.size == 1

        # array
        dates = ["2018-01-01", "2018-01-02"]

        c = Coordinates([dates], dims=["time"])
        assert c.dims == ("time",)
        assert c.udims == ("time",)
        assert c.xdims == ("time",)
        assert c.shape == (2,)
        assert c.ndim == 1
        assert c.size == 2

        c = Coordinates([np.array(dates).astype(np.datetime64)], dims=["time"])
        assert c.dims == ("time",)
        assert c.udims == ("time",)
        assert c.xdims == ("time",)
        assert c.shape == (2,)
        assert c.ndim == 1

        c = Coordinates([xr.DataArray(dates).astype(np.datetime64)], dims=["time"])
        assert c.dims == ("time",)
        assert c.udims == ("time",)
        assert c.xdims == ("time",)
        assert c.shape == (2,)
        assert c.ndim == 1
        assert c.size == 2

        # use DataArray name, but dims overrides the DataArray name
        c = Coordinates([xr.DataArray(dates, name="time").astype(np.datetime64)])
        assert c.dims == ("time",)
        assert c.udims == ("time",)
        assert c.xdims == ("time",)
        assert c.shape == (2,)
        assert c.ndim == 1
        assert c.size == 2

        c = Coordinates([xr.DataArray(dates, name="a").astype(np.datetime64)], dims=["time"])
        assert c.dims == ("time",)
        assert c.udims == ("time",)
        assert c.xdims == ("time",)
        assert c.shape == (2,)
        assert c.ndim == 1
        assert c.size == 2

    def test_unstacked(self):
        # single value
        c = Coordinates([0, 10], dims=["lat", "lon"])
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert c.xdims == ("lat", "lon")
        assert c.shape == (1, 1)
        assert c.ndim == 2
        assert c.size == 1

        # arrays
        lat = [0, 1, 2]
        lon = [10, 20, 30, 40]

        c = Coordinates([lat, lon], dims=["lat", "lon"])
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert c.xdims == ("lat", "lon")
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

        # use DataArray names
        c = Coordinates([xr.DataArray(lat, name="lat"), xr.DataArray(lon, name="lon")])
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert c.xdims == ("lat", "lon")
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

        # dims overrides the DataArray names
        c = Coordinates([xr.DataArray(lat, name="a"), xr.DataArray(lon, name="b")], dims=["lat", "lon"])
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert c.xdims == ("lat", "lon")
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

    def test_stacked(self):
        # single value
        c = Coordinates([[0, 10]], dims=["lat_lon"])
        assert c.dims == ("lat_lon",)
        assert c.udims == ("lat", "lon")
        assert c.xdims == ("lat_lon",)
        assert c.shape == (1,)
        assert c.ndim == 1
        assert c.size == 1

        # arrays
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        c = Coordinates([[lat, lon]], dims=["lat_lon"])
        assert c.dims == ("lat_lon",)
        assert c.udims == ("lat", "lon")
        assert c.xdims == ("lat_lon",)
        assert c.shape == (3,)
        assert c.ndim == 1
        assert c.size == 3

        # nested dims version
        c = Coordinates([[lat, lon]], dims=[["lat", "lon"]])
        assert c.dims == ("lat_lon",)
        assert c.udims == ("lat", "lon")
        assert c.xdims == ("lat_lon",)
        assert c.shape == (3,)
        assert c.ndim == 1
        assert c.size == 3

    def test_stacked_shaped(self):
        # explicit
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        latlon = StackedCoordinates([lat, lon], dims=["lat", "lon"])
        c = Coordinates([latlon])
        assert c.dims == ("lat_lon",)
        assert c.udims == ("lat", "lon")
        assert len(set(c.xdims)) == 2  # doesn't really matter what they are called
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

        # implicit
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = Coordinates([[lat, lon]], dims=["lat_lon"])
        assert c.dims == ("lat_lon",)
        assert c.udims == ("lat", "lon")
        assert len(set(c.xdims)) == 2  # doesn't really matter what they are called
        assert c.shape == (3, 4)
        assert c.ndim == 2
        assert c.size == 12

    def test_rotated(self):
        latlon = AffineCoordinates(geotransform=(10.0, 2.0, 0.0, 20.0, 0.0, -3.0), shape=(3, 4))
        c = Coordinates([latlon])
        assert c.dims == ("lat_lon",)
        assert c.udims == ("lat", "lon")
        assert len(set(c.xdims)) == 2  # doesn't really matter what they are called
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
        assert c.xdims == ("lat_lon", "time")
        assert c.shape == (3, 2)
        assert c.ndim == 2
        assert c.size == 6
        repr(c)

        # stacked, nested dims version
        c = Coordinates([[lat, lon], dates], dims=[["lat", "lon"], "time"])
        assert c.dims == ("lat_lon", "time")
        assert c.udims == ("lat", "lon", "time")
        assert c.xdims == ("lat_lon", "time")
        assert c.shape == (3, 2)
        assert c.ndim == 2
        assert c.size == 6
        repr(c)

    def test_mixed_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        dates = [["2018-01-01", "2018-01-02", "2018-01-03"], ["2019-01-01", "2019-01-02", "2019-01-03"]]
        c = Coordinates([[lat, lon], dates], dims=["lat_lon", "time"])
        assert c.dims == ("lat_lon", "time")
        assert c.udims == ("lat", "lon", "time")
        assert len(set(c.xdims)) == 4  # doesn't really matter what they are called
        assert c.shape == (3, 4, 2, 3)
        assert c.ndim == 4
        assert c.size == 72
        repr(c)

    def test_mixed_affine(sesf):
        latlon = AffineCoordinates(geotransform=(10.0, 2.0, 0.0, 20.0, 0.0, -3.0), shape=(3, 4))
        dates = [["2018-01-01", "2018-01-02", "2018-01-03"], ["2019-01-01", "2019-01-02", "2019-01-03"]]
        c = Coordinates([latlon, dates], dims=["lat_lon", "time"])
        assert c.dims == ("lat_lon", "time")
        assert c.udims == ("lat", "lon", "time")
        assert len(set(c.xdims)) == 4  # doesn't really matter what they are called
        assert c.shape == (3, 4, 2, 3)
        assert c.ndim == 4
        assert c.size == 72
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

    def test_from_url(self):
        crds = Coordinates([[41, 40], [-71, -70], "2018-05-19"], dims=["lat", "lon", "time"])
        crds2 = crds.transform("EPSG:3857")

        url = (
            r"http://testwms/?map=map&&service=WMS&request=GetMap&layers=layer&styles=&format=image%2Fpng"
            r"&transparent=true&version={version}&transparency=true&width=256&height=256&srs=EPSG%3A{epsg}"
            r"&bbox={},{},{},{}&time={}"
        )

        # version 1.1.1
        c = Coordinates.from_url(
            url.format(
                min(crds2.bounds["lon"]),
                min(crds2.bounds["lat"]),
                max(crds2.bounds["lon"]),
                max(crds2.bounds["lat"]),
                crds2.bounds["time"][0],
                version="1.1.1",
                epsg="3857",
            )
        )
        assert c.bounds == crds2.bounds

        c = Coordinates.from_url(
            url.format(
                min(crds.bounds["lon"]),
                min(crds.bounds["lat"]),
                max(crds.bounds["lon"]),
                max(crds.bounds["lat"]),
                crds.bounds["time"][0],
                version="1.1.1",
                epsg="4326",
            )
        )
        assert c.bounds == crds.bounds

        # version 1.3
        c = Coordinates.from_url(
            url.format(
                min(crds2.bounds["lon"]),
                min(crds2.bounds["lat"]),
                max(crds2.bounds["lon"]),
                max(crds2.bounds["lat"]),
                crds2.bounds["time"][0],
                version="1.3",
                epsg="3857",
            )
        )
        assert c.bounds == crds2.bounds

        c = Coordinates.from_url(
            url.format(
                min(crds.bounds["lat"]),
                min(crds.bounds["lon"]),
                max(crds.bounds["lat"]),
                max(crds.bounds["lon"]),
                crds.bounds["time"][0],
                version="1.3",
                epsg="4326",
            )
        )

        assert c.bounds == crds.bounds

        # WCS version
        crds = Coordinates([[41, 40], [-71, -70], "2018-05-19"], dims=["lat", "lon", "time"])
        crds2 = crds.transform("EPSG:3857")

        url = (
            r"http://testwms/?map=map&&service=WCS&request=GetMap&layers=layer&styles=&format=image%2Fpng"
            r"&transparent=true&version={version}&transparency=true&width=256&height=256&srs=EPSG%3A{epsg}"
            r"&bbox={},{},{},{}&time={}"
        )

        c = Coordinates.from_url(
            url.format(
                min(crds2.bounds["lon"]),
                min(crds2.bounds["lat"]),
                max(crds2.bounds["lon"]),
                max(crds2.bounds["lat"]),
                crds2.bounds["time"][0],
                version="1.1",
                epsg="3857",
            )
        )
        assert c.bounds == crds2.bounds

        # Based on all the documentation I've read, this should be correct, but
        # based on the server's I've checked, this does not seem correct
        # c = Coordinates.from_url(
        #     url.format(
        #         min(crds.bounds["lat"]),
        #         min(crds.bounds["lon"]),
        #         max(crds.bounds["lat"]),
        #         max(crds.bounds["lon"]),
        #         crds.bounds["time"][0],
        #         version="1.1",
        #         epsg="4326",
        #     )
        # )

        c = Coordinates.from_url(
            url.format(
                min(crds.bounds["lon"]),
                min(crds.bounds["lat"]),
                max(crds.bounds["lon"]),
                max(crds.bounds["lat"]),
                crds.bounds["time"][0],
                version="1.1",
                epsg="4326",
            )
        )

        assert c.bounds == crds.bounds

    def test_from_xarray(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]
        dates = ["2018-01-01", "2018-01-02"]

        c = Coordinates(
            [
                StackedCoordinates([ArrayCoordinates1d(lat, name="lat"), ArrayCoordinates1d(lon, name="lon")]),
                ArrayCoordinates1d(dates, name="time"),
            ],
            crs="EPSG:2193",
        )

        # from DataArray
        x = xr.DataArray(np.empty(c.shape), coords=c.xcoords, dims=c.xdims, attrs={"crs": c.crs})
        c2 = Coordinates.from_xarray(x)
        assert c2 == c
        assert c2.crs == "EPSG:2193"

        # prefer crs argument over attrs.crs
        x = xr.DataArray(np.empty(c.shape), coords=c.xcoords, dims=c.xdims, attrs={"crs": c.crs})
        c3 = Coordinates.from_xarray(x, crs="EPSG:4326")
        assert c3.crs == "EPSG:4326"

        # from DataArrayCoords
        c4 = Coordinates.from_xarray(x.coords, crs="EPSG:2193")
        assert c4 == c
        assert c4.crs == "EPSG:2193"

        # crs warning
        with pytest.warns(UserWarning, match="using default crs"):
            c2 = Coordinates.from_xarray(x.coords)

        # invalid
        with pytest.raises(TypeError, match="Coordinates.from_xarray expects an xarray"):
            Coordinates.from_xarray([0, 10])

    def test_from_xarray_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        dates = [["2018-01-01", "2018-01-02", "2018-01-03"], ["2019-01-01", "2019-01-02", "2019-01-03"]]
        c = Coordinates([[lat, lon], dates], dims=["lat_lon", "time"], crs="EPSG:2193")

        # from xarray
        x = xr.DataArray(np.empty(c.shape), coords=c.xcoords, dims=c.xdims, attrs={"crs": c.crs})
        c2 = Coordinates.from_xarray(x)
        assert c2 == c

    def test_from_xarray_with_outputs(self):
        lat = [0, 1, 2]
        lon = [10, 20, 30]

        c = Coordinates([lat, lon], dims=["lat", "lon"], crs="EPSG:2193")

        # from xarray
        dims = c.xdims + ("output",)
        coords = {"output": ["a", "b"], **c.xcoords}
        shape = c.shape + (2,)

        x = xr.DataArray(np.empty(c.shape + (2,)), coords=coords, dims=dims, attrs={"crs": c.crs})
        c2 = Coordinates.from_xarray(x)
        assert c2 == c

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
        np.testing.assert_array_almost_equal(ct["alt"].coordinates, 0.30480061 * c["alt"].coordinates)

        # invalid
        with pytest.raises(ValueError):
            Coordinates([alt], crs="EPSG:2193")

    def test_CRS(self):
        lat = ArrayCoordinates1d([0, 1, 2], "lat")
        lon = ArrayCoordinates1d([0, 1, 2], "lon")
        c = Coordinates([lat, lon])
        assert isinstance(c.CRS, pyproj.CRS)

    def test_alt_units(self):
        lat = ArrayCoordinates1d([0, 1, 2], "lat")
        lon = ArrayCoordinates1d([0, 1, 2], "lon")
        alt = ArrayCoordinates1d([0, 1, 2], name="alt")

        c = Coordinates([lat, lon], crs="proj=merc")
        assert c.alt_units is None

        c = Coordinates([alt], crs="+proj=merc +vunits=us-ft")

        with pytest.warns(UserWarning):
            assert c.alt_units in ["us-ft", "US survey foot"]  # pyproj < 3.0  # pyproj >= 3.0


class TestCoordinatesSerialization(object):
    def test_definition(self):
        # this tests array coordinates, uniform coordinates, and stacked coordinates
        c = Coordinates(
            [[[0, 1, 2], [10, 20, 30]], ["2018-01-01", "2018-01-02"], crange(0, 10, 0.5)],
            dims=["lat_lon", "time", "alt"],
            crs="+proj=merc +vunits=us-ft",
        )
        d = c.definition
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)
        c2 = Coordinates.from_definition(d)
        assert c2 == c

    def test_definition_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        c = Coordinates([[lat, lon]], dims=["lat_lon"])
        d = c.definition
        json.dumps(d, cls=podpac.core.utils.JSONEncoder)
        c2 = Coordinates.from_definition(d)
        assert c2 == c

    def test_definition_affine(self):
        latlon = AffineCoordinates(geotransform=(10.0, 2.0, 0.0, 20.0, 0.0, -3.0), shape=(3, 4))
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
            crs="+proj=merc +vunits=us-ft",
        )

        s = c.json

        json.loads(s)

        c2 = Coordinates.from_json(s)
        assert c2 == c


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

        x = xr.DataArray(np.empty(c.shape), dims=c.xdims, coords=c.xcoords)

        assert x.dims == ("lat", "lon", "time")
        np.testing.assert_equal(x["lat"], np.array(lat, dtype=float))
        np.testing.assert_equal(x["lon"], np.array(lon, dtype=float))
        np.testing.assert_equal(x["time"], np.array(dates).astype(np.datetime64))

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

        x = xr.DataArray(np.empty(c.shape), dims=c.xdims, coords=c.xcoords)

        assert x.dims == ("lat_lon", "time")
        np.testing.assert_equal(x["lat"], np.array(lat, dtype=float))
        np.testing.assert_equal(x["lon"], np.array(lon, dtype=float))
        np.testing.assert_equal(x["time"], np.array(dates).astype(np.datetime64))

    def test_xarray_coords_stacked_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        dates = ["2018-01-01", "2018-01-02"]

        c = Coordinates([StackedCoordinates([lat, lon], dims=["lat", "lon"]), ArrayCoordinates1d(dates, name="time")])

        x = xr.DataArray(np.empty(c.shape), dims=c.xdims, coords=c.xcoords)

        assert len(x.dims) == 3
        np.testing.assert_equal(x["lat"], np.array(lat, dtype=float))
        np.testing.assert_equal(x["lon"], np.array(lon, dtype=float))
        np.testing.assert_equal(x["time"], np.array(dates).astype(np.datetime64))

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


class TestCoordinatesDict(object):
    coords = Coordinates([[[0, 1, 2], [10, 20, 30]], ["2018-01-01", "2018-01-02"]], dims=["lat_lon", "time"])

    def test_keys(self):
        assert set(self.coords.keys()) == {"lat_lon", "time"}

    def test_values(self):
        values = list(self.coords.values())
        assert len(values) == 2
        assert self.coords["lat_lon"] in values
        assert self.coords["time"] in values

    def test_items(self):
        keys, values = zip(*self.coords.items())
        assert set(keys) == {"lat_lon", "time"}
        assert len(values) == 2
        assert self.coords["lat_lon"] in values
        assert self.coords["time"] in values

    def test_iter(self):
        assert set(self.coords) == {"lat_lon", "time"}

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

        with pytest.raises(ValueError, match="Shape mismatch"):
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

    def test_get_index_stacked_shaped(self):
        lat = np.linspace(0, 1, 20).reshape((5, 4))
        lon = np.linspace(10, 20, 20).reshape((5, 4))
        dates = ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"]

        c = Coordinates([StackedCoordinates([lat, lon], dims=["lat", "lon"]), ArrayCoordinates1d(dates, name="time")])

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
        crs="+proj=merc +vunits=us-ft",
    )

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
            crs="+proj=merc +vunits=us-ft",
        )

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
            crs="+proj=merc +vunits=us-ft",
        )
        c2 = c.unique()
        assert_equal(c2["lat"].coordinates, [0, 1, 2])
        assert_equal(c2["time"].coordinates, [np.datetime64("2018-01-01"), np.datetime64("2018-01-02")])
        assert_equal(c2["alt"].coordinates, [])

        # return indices
        c = Coordinates(
            [[2, 1, 0, 1], ["2018-01-01", "2018-01-02", "2018-01-01"], []],
            dims=["lat", "time", "alt"],
            crs="+proj=merc +vunits=us-ft",
        )
        c2, I = c.unique(return_index=True)
        assert_equal(c2["lat"].coordinates, [0, 1, 2])
        assert_equal(c2["time"].coordinates, [np.datetime64("2018-01-01"), np.datetime64("2018-01-02")])
        assert_equal(c2["alt"].coordinates, [])
        assert c2 == c[I]

        # stacked
        lat_lon = [(0, 0), (0, 1), (0, 2), (0, 2), (1, 0), (1, 1), (1, 1)]
        lat, lon = zip(*lat_lon)
        c = Coordinates([[lat, lon]], dims=["lat_lon"])
        c2 = c.unique()
        assert_equal(c2["lat"].coordinates, [0.0, 0.0, 0.0, 1.0, 1.0])
        assert_equal(c2["lon"].coordinates, [0.0, 1.0, 2.0, 0.0, 1.0])

        # empty
        c = Coordinates([])
        c2 = c.unique()
        assert c2.size == 0

        c2, I = c.unique(return_index=True)
        assert c2.size == 0
        assert c2 == c[I]

    def test_unique_properties(self):
        c = Coordinates(
            [[2, 1, 0, 1], ["2018-01-01", "2018-01-02", "2018-01-01"], []],
            dims=["lat", "time", "alt"],
            crs="+proj=merc +vunits=us-ft",
        )
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

        with pytest.raises(ValueError, match="Invalid transpose dimensions"):
            c.transpose("lat", "lon", "alt")

    def test_transpose_stacked_shaped(self):
        lat = np.linspace(0, 1, 12).reshape((3, 4))
        lon = np.linspace(10, 20, 12).reshape((3, 4))
        dates = ["2018-01-01", "2018-01-02"]
        c = Coordinates([[lat, lon], dates], dims=["lat_lon", "time"])

        t = c.transpose("time", "lon_lat", in_place=False)
        assert c.dims == ("lat_lon", "time")
        assert t.dims == ("time", "lon_lat")

        c.transpose("time", "lon_lat", in_place=True)
        assert c.dims == ("time", "lon_lat")

    def test_transpose_stacked(self):
        lat = np.linspace(0, 1, 12)
        lon = np.linspace(10, 20, 12)
        dates = ["2018-01-01", "2018-01-02"]
        c = Coordinates([[lat, lon], dates], dims=["lat_lon", "time"])

        t = c.transpose("time", "lon_lat", in_place=False)
        assert c.dims == ("lat_lon", "time")
        assert t.dims == ("time", "lon_lat")

        c.transpose("time", "lon_lat", in_place=True)
        assert c.dims == ("time", "lon_lat")

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

        s, I = c.select({"lat": [0.5, 2.5]}, return_index=True)
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

        s, I = c.select({"lon": [5, 25]}, return_index=True)
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

        s, I = c.select({"lat": [0.5, 2.5]}, outer=True, return_index=True)
        assert isinstance(s, Coordinates)
        assert s.dims == c.dims
        assert s["lat"] == c["lat"][0:4]
        assert s["lon"] == c["lon"]
        assert s["time"] == c["time"]
        assert s == c[I]

        # no matching dimension
        s = c.select({"alt": [0, 10]})
        assert s == c

        s, I = c.select({"alt": [0, 10]}, return_index=True)
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

        s, I = c.select({"lat": [0.5, 3.5], "lon": [25, 55]}, return_index=True)
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

        c2, I = c.intersect(other, return_index=True)
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

        # Confusing time intersection
        ct = Coordinates([["2012-05-19T12:00:00", "2012-05-19T13:00:00", "2012-05-20T14:00:00"]], ["time"])
        cti = Coordinates([["2012-05-18", "2012-05-19"]], ["time"])
        ct2 = ct.intersect(cti, outer=True)
        assert ct2.size == 3

        ct2 = ct.intersect(cti, outer=False)
        assert ct2.size == 0  # Is this behavior desired?

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
        # should change the other coordinates crs into the coordinates crs for intersect
        c = Coordinates(
            [np.linspace(0, 10, 11), np.linspace(0, 10, 11), ["2018-01-01", "2018-01-02"]], dims=["lat", "lon", "time"]
        )
        o = Coordinates(
            [np.linspace(28000000, 29500000, 20), np.linspace(-280000, 400000, 20), ["2018-01-01", "2018-01-02"]],
            dims=["lat", "lon", "time"],
            crs="EPSG:2193",
        )

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

    def test_issubset(self):
        c1 = Coordinates([[0, 1, 2, 3], [10, 20, 30, 40]], dims=["lat", "lon"])
        c2 = Coordinates([[1, 2, 3, 4], [10, 20, 30, 40]], dims=["lat", "lon"])
        c3 = Coordinates([[1, 3], [40, 30, 20, 10]], dims=["lat", "lon"])

        # self
        assert c1.issubset(c1)
        assert c2.issubset(c2)
        assert c3.issubset(c3)

        # other
        assert not c1.issubset(c2)
        assert not c1.issubset(c3)
        assert not c2.issubset(c1)
        assert not c2.issubset(c3)
        assert c3.issubset(c1)
        assert c3.issubset(c2)

        # missing dims
        c4 = c1.drop("lat")
        assert not c1.issubset(c4)
        assert not c4.issubset(c1)

    def test_issubset_stacked(self):
        lat1, lon1 = [0, 1, 2, 3], [10, 20, 30, 40]
        u1 = Coordinates([lat1, lon1], dims=["lat", "lon"])
        s1 = Coordinates([[lat1, lon1]], dims=["lat_lon"])

        lat2, lon2 = [1, 3], [20, 40]
        u2 = Coordinates([lat2, lon2], dims=["lat", "lon"])
        s2 = Coordinates([[lat2, lon2]], dims=["lat_lon"])

        lat3, lon3 = [1, 3], [40, 20]
        u3 = Coordinates([lat3, lon3], dims=["lat", "lon"])
        s3 = Coordinates([[lat3, lon3]], dims=["lat_lon"])

        # stacked issubset of stacked: must check stacked dims together
        assert s1.issubset(s1)
        assert s2.issubset(s1)
        assert not s1.issubset(s2)
        assert not s3.issubset(s1)  # this is an important case because the udims are all subsets

        # stacked issubset of unstacked: check udims individually
        assert s1.issubset(u1)
        assert s2.issubset(u2)
        assert s3.issubset(u3)

        assert s2.issubset(u1)
        assert s3.issubset(u1)

        assert not s1.issubset(u2)

        # unstacked issubset of stacked: must check other's stacked dims together
        assert not u1.issubset(s1)
        assert not u2.issubset(s2)
        assert not u3.issubset(s3)

        # unstacked issubset of stacked: sometimes it is a subset, not yet implemented
        # lat, lon = np.meshgrid(lat1, lon1)
        # s = Coordinates([[lat.flatten(), lon.flatten()]], dims=['lat_lon'])
        # assert u1.issubset(s)
        # assert u2.issubset(s)
        # assert u3.issubset(s)

    def test_issubset_stacked_shaped(self):
        lat1, lon1 = np.array([0, 1, 2, 3]), np.array([10, 20, 30, 40])
        u1 = Coordinates([lat1, lon1], dims=["lat", "lon"])
        d1 = Coordinates([[lat1.reshape((2, 2)), lon1.reshape((2, 2))]], dims=["lat_lon"])

        lat2, lon2 = np.array([1, 3]), np.array([20, 40])
        u2 = Coordinates([lat2, lon2], dims=["lat", "lon"])
        d2 = Coordinates([[lat2.reshape((2, 1)), lon2.reshape((2, 1))]], dims=["lat_lon"])

        lat3, lon3 = np.array([1, 3]), np.array([40, 20])
        u3 = Coordinates([lat3, lon3], dims=["lat", "lon"])
        d3 = Coordinates([[lat3.reshape((2, 1)), lon3.reshape((2, 1))]], dims=["lat_lon"])

        # dependent issubset of dependent: must check dependent dims together
        assert d1.issubset(d1)
        assert d2.issubset(d1)
        assert not d1.issubset(d2)
        assert not d3.issubset(d1)  # this is an important case because the udims are all subsets

        # dependent issubset of unstacked: check udims individually
        assert d1.issubset(u1)
        assert d2.issubset(u2)
        assert d3.issubset(u3)

        assert d2.issubset(u1)
        assert d3.issubset(u1)

        assert not d1.issubset(u2)

        # unstacked issubset of dependent: must check other's dependent dims together
        assert not u1.issubset(d1)
        assert not u2.issubset(d2)
        assert not u3.issubset(d3)

        # unstacked issubset of dependent: sometimes it is a subset, not yet implemented
        # lat, lon = np.meshgrid(lat1, lon1)
        # d = Coordinates([[lat, lon]], dims=['lat_lon'])
        # assert u1.issubset(d)
        # assert u2.issubset(d)
        # assert u3.issubset(d)

    def test_issubset_time(self):
        c1 = Coordinates([["2020-01-01", "2020-01-02", "2020-01-03"]], dims=["time"])
        c2 = Coordinates([["2020-01-02", "2020-01-03"]], dims=["time"])
        c3 = Coordinates([["2020-01-01T00:00:00", "2020-01-02T00:00:00", "2020-01-03T00:00:00"]], dims=["time"])

        # self
        assert c1.issubset(c1)
        assert c2.issubset(c2)
        assert c3.issubset(c3)

        # other
        assert not c1.issubset(c2)
        assert c1.issubset(c3)
        assert c2.issubset(c1)
        assert c2.issubset(c3)
        assert c3.issubset(c1)
        assert not c3.issubset(c2)


class TestCoordinatesSpecial(object):
    def test_repr(self):
        repr(Coordinates([[0, 1], [10, 20], ["2018-01-01", "2018-01-02"]], dims=["lat", "lon", "time"]))
        repr(Coordinates([[[0, 1], [10, 20]], ["2018-01-01", "2018-01-02"]], dims=["lat_lon", "time"]))
        repr(Coordinates([[[[0, 1]], [[10, 20]]], [["2018-01-01", "2018-01-02"]]], dims=["lat_lon", "time"]))
        repr(Coordinates([0, 10, []], dims=["lat", "lon", "time"]))
        repr(Coordinates([crange(0, 10, 0.5)], dims=["alt"], crs="+proj=merc +vunits=us-ft"))
        repr(Coordinates([]))

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


class TestCoordinatesGeoTransform(object):
    def uniform_working(self):
        # order: -lat, lon
        c = Coordinates([clinspace(1.5, 0.5, 5, "lat"), clinspace(1, 2, 9, "lon")])
        tf = np.array(c.geotransform).reshape(2, 3)
        np.testing.assert_almost_equal(
            tf, np.array([[c["lon"].bounds[0], c["lon"].step, 0], [c["lat"].bounds[1], 0, c["lat"].step]])
        )
        # order: lon, lat
        c = Coordinates([clinspace(0.5, 1.5, 5, "lon"), clinspace(1, 2, 9, "lat")])
        tf = np.array(c.geotransform).reshape(2, 3)
        np.testing.assert_almost_equal(
            tf, np.array([[c["lon"].bounds[0], 0, c["lon"].step], [c["lat"].bounds[0], c["lat"].step, 0]])
        )

        # order: lon, -lat, time
        c = Coordinates([clinspace(0.5, 1.5, 5, "lon"), clinspace(2, 1, 9, "lat"), crange(10, 11, 2, "time")])
        tf = np.array(c.geotransform).reshape(2, 3)
        np.testing.assert_almost_equal(
            tf, np.array([[c["lon"].bounds[0], 0, c["lon"].step], [c["lat"].bounds[1], c["lat"].step, 0]])
        )
        # order: -lon, -lat, time, alt
        c = Coordinates(
            [
                clinspace(1.5, 0.5, 5, "lon"),
                clinspace(2, 1, 9, "lat"),
                crange(10, 11, 2, "time"),
                crange(10, 11, 2, "alt"),
            ]
        )
        tf = np.array(c.geotransform).reshape(2, 3)
        np.testing.assert_almost_equal(
            tf, np.array([[c["lon"].bounds[1], 0, c["lon"].step], [c["lat"].bounds[1], c["lat"].step, 0]])
        )

    def error_time_alt_too_big(self):
        # time
        c = Coordinates(
            [
                clinspace(1.5, 0.5, 5, "lon"),
                clinspace(2, 1, 9, "lat"),
                crange(1, 11, 2, "time"),
                crange(1, 11, 2, "alt"),
            ]
        )
        with pytest.raises(
            TypeError, match='Only 2-D coordinates have a GDAL transform. This array has a "time" dimension of'
        ):
            c.geotransform
        # alt
        c = Coordinates([clinspace(1.5, 0.5, 5, "lon"), clinspace(2, 1, 9, "lat"), crange(1, 11, 2, "alt")])
        with pytest.raises(
            TypeError, match='Only 2-D coordinates have a GDAL transform. This array has a "alt" dimension of'
        ):
            c.geotransform

    @pytest.mark.skip(reason="obsolete")
    def rot_coords_working(self):
        # order -lat, lon
        rc = RotatedCoordinates(shape=(4, 3), theta=np.pi / 8, origin=[10, 20], step=[-2.0, 1.0], dims=["lat", "lon"])
        c = Coordinates([rc], dims=["lat,lon"])
        tf = np.array(c.geotransform).reshape(2, 3)
        np.testing.assert_almost_equal(
            tf,
            np.array(
                [
                    [rc.origin[1] - rc.step[1] / 2, rc.step[1] * np.cos(rc.theta), -rc.step[0] * np.sin(rc.theta)],
                    [rc.origin[0] - rc.step[0] / 2, rc.step[1] * np.sin(rc.theta), rc.step[0] * np.cos(rc.theta)],
                ]
            ),
        )
        # order lon, lat
        rc = RotatedCoordinates(shape=(4, 3), theta=np.pi / 8, origin=[10, 20], step=[2.0, 1.0], dims=["lon", "lat"])
        c = Coordinates([rc], dims=["lon,lat"])
        tf = np.array(c.geotransform).reshape(2, 3)
        np.testing.assert_almost_equal(
            tf,
            np.array(
                [
                    [rc.origin[0] - rc.step[0] / 2, rc.step[1] * np.sin(rc.theta), rc.step[0] * np.cos(rc.theta)],
                    [rc.origin[1] - rc.step[1] / 2, rc.step[1] * np.cos(rc.theta), -rc.step[0] * np.sin(rc.theta)],
                ]
            ),
        )

        # order -lon, lat
        rc = RotatedCoordinates(shape=(4, 3), theta=np.pi / 8, origin=[10, 20], step=[-2.0, 1.0], dims=["lon", "lat"])
        c = Coordinates([rc], dims=["lon,lat"])
        tf = np.array(c.geotransform).reshape(2, 3)
        np.testing.assert_almost_equal(
            tf,
            np.array(
                [
                    [rc.origin[0] - rc.step[0] / 2, rc.step[1] * np.sin(rc.theta), rc.step[0] * np.cos(rc.theta)],
                    [rc.origin[1] - rc.step[1] / 2, rc.step[1] * np.cos(rc.theta), -rc.step[0] * np.sin(rc.theta)],
                ]
            ),
        )
        # order -lat, -lon
        rc = RotatedCoordinates(shape=(4, 3), theta=np.pi / 8, origin=[10, 20], step=[-2.0, -1.0], dims=["lat", "lon"])
        c = Coordinates([rc], dims=["lat,lon"])
        tf = np.array(c.geotransform).reshape(2, 3)
        np.testing.assert_almost_equal(
            tf,
            np.array(
                [
                    [rc.origin[1] - rc.step[1] / 2, rc.step[1] * np.cos(rc.theta), -rc.step[0] * np.sin(rc.theta)],
                    [rc.origin[0] - rc.step[0] / 2, rc.step[1] * np.sin(rc.theta), rc.step[0] * np.cos(rc.theta)],
                ]
            ),
        )


class TestCoordinatesMethodTransform(object):
    def test_transform(self):
        c = Coordinates(
            [[0, 1], [10, 20, 30, 40], ["2018-01-01", "2018-01-02"]], dims=["lat", "lon", "time"], crs="EPSG:4326"
        )

        # transform
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
        t = c.transform(proj)
        assert c.crs == "EPSG:4326"
        assert t.crs == proj
        assert round(t["lat"].coordinates[0]) == 0.0

    def test_transform_stacked(self):
        c = Coordinates(
            [[[0, 1], [10, 20]], ["2018-01-01", "2018-01-02", "2018-01-03"]], dims=["lat_lon", "time"], crs="EPSG:4326"
        )

        proj = "+proj=merc +lat_ts=56.5 +ellps=GRS80"
        t = c.transform(proj)
        assert c.crs == "EPSG:4326"
        assert t.crs == proj
        assert round(t["lat"].coordinates[0]) == 0.0

    def test_transform_alt(self):
        c = Coordinates(
            [[0, 1], [10, 20, 30, 40], ["2018-01-01", "2018-01-02"], [100, 200, 300]],
            dims=["lat", "lon", "time", "alt"],
            crs="+proj=merc +vunits=us-ft",
        )

        proj = "+proj=merc +vunits=m"
        t = c.transform(proj)
        assert c.crs == "+proj=merc +vunits=us-ft"
        assert t.crs == "+proj=merc +vunits=m"
        np.testing.assert_array_almost_equal(t["lat"].coordinates, c["lat"].coordinates)
        np.testing.assert_array_almost_equal(t["lon"].coordinates, c["lon"].coordinates)
        assert t["time"] == c["time"]
        np.testing.assert_array_almost_equal(t["alt"].coordinates, 0.30480061 * c["alt"].coordinates)

    def test_transform_uniform_to_uniform(self):
        c = Coordinates([clinspace(-90, 90, 5, "lat"), clinspace(-180, 180, 11, "lon"), clinspace(0, 1, 5, "time")])
        t = c.transform("EPSG:4269")  # NAD 1983 uses same ellipsoid

        assert isinstance(t["lat"], UniformCoordinates1d)
        assert isinstance(t["lon"], UniformCoordinates1d)
        assert t.crs == "EPSG:4269"
        assert t.dims == c.dims

        # Same thing, change the order of the inputs
        c = Coordinates(
            [clinspace(90, -90, 5, "lat"), clinspace(180, -180, 11, "lon"), clinspace(0, 1, 5, "time")][::-1]
        )
        t = c.transform("EPSG:4269")  # NAD 1983 uses same ellipsoid

        assert isinstance(t["lat"], UniformCoordinates1d)
        assert isinstance(t["lon"], UniformCoordinates1d)
        assert t.crs == "EPSG:4269"

        assert t.dims == c.dims
        for d in ["lat", "lon"]:
            for a in ["start", "stop", "step"]:
                np.testing.assert_almost_equal(getattr(c[d], a), getattr(t[d], a))

    def test_transform_uniform_stacked(self):
        # TODO: Fix this test
        c = Coordinates(
            [[clinspace(-90, 90, 11, "lat"), clinspace(-180, 180, 11, "lon")], clinspace(0, 1, 5, "time")],
            [["lat", "lon"], "time"],
        )
        t = c.transform("EPSG:4269")  # NAD 1983 uses same ellipsoid

        assert isinstance(t["lat"], UniformCoordinates1d)
        assert isinstance(t["lon"], UniformCoordinates1d)
        np.testing.assert_array_almost_equal(t["lat"].coordinates, c["lat"].coordinates)
        np.testing.assert_array_almost_equal(t["lon"].coordinates, c["lon"].coordinates)

    def test_transform_uniform_to_array(self):
        c = Coordinates([clinspace(-45, 45, 5, "lat"), clinspace(-180, 180, 11, "lon")])

        # Ok for array coordinates
        t = c.transform("EPSG:3395")

        assert isinstance(t["lat"], ArrayCoordinates1d)
        assert isinstance(t["lon"], UniformCoordinates1d)
        assert t["lon"].is_descending == c["lon"].is_descending
        assert t["lat"].is_descending == c["lat"].is_descending

        t2 = t.transform(c.crs)

        for d in ["lon", "lat"]:
            for a in ["start", "stop", "step"]:
                np.testing.assert_almost_equal(getattr(c[d], a), getattr(t2[d], a))

        # Reverse the order of the coordinates
        c = Coordinates([clinspace(45, -45, 5, "lat"), clinspace(180, -180, 11, "lon")])

        # Ok for array coordinates
        t = c.transform("EPSG:3395")

        assert isinstance(t["lat"], ArrayCoordinates1d)
        assert isinstance(t["lon"], UniformCoordinates1d)
        assert t["lon"].is_descending == c["lon"].is_descending
        assert t["lat"].is_descending == c["lat"].is_descending

        t2 = t.transform(c.crs)

        for d in ["lon", "lat"]:
            for a in ["start", "stop", "step"]:
                np.testing.assert_almost_equal(getattr(c[d], a), getattr(t2[d], a))

    def test_transform_uniform_to_stacked_to_uniform(self):
        c = Coordinates([clinspace(50, 45, 7, "lat"), clinspace(70, 75, 11, "lon")])

        # Ok for array coordinates
        t = c.transform("EPSG:32629")
        assert "lat_lon" in t.dims

        t2 = t.transform(c.crs)

        np.testing.assert_allclose(t2["lat"].start, c["lat"].start)
        np.testing.assert_allclose(t2["lat"].stop, c["lat"].stop)
        np.testing.assert_allclose(t2["lat"].step, c["lat"].step)
        np.testing.assert_allclose(t2["lon"].start, c["lon"].start)
        np.testing.assert_allclose(t2["lon"].stop, c["lon"].stop)
        np.testing.assert_allclose(t2["lon"].step, c["lon"].step)

        # TODO JXM test this with time, alt, etc

    def test_transform_stacked_to_stacked(self):
        c = Coordinates([[np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])]], ["lat_lon"])
        c2 = Coordinates([[np.array([1, 2, 3, 4, 5, 6]), np.array([7, 8, 9, 10, 11, 12])]], ["lat_lon"])

        # Ok for array coordinates
        t = c.transform("EPSG:32629")
        assert "lat_lon" in t.dims
        t_s = c2.transform("EPSG:32629")
        assert "lat_lon" in t_s.dims

        for d in ["lat", "lon"]:
            np.testing.assert_almost_equal(t[d].coordinates.ravel(), t_s[d].coordinates.ravel())

        t2 = t.transform(c.crs)
        t2_s = t_s.transform(c.crs)

        for d in ["lat", "lon"]:
            np.testing.assert_almost_equal(t2[d].coordinates, c[d].coordinates)
            np.testing.assert_almost_equal(t2_s[d].coordinates, c2[d].coordinates)

        # Reverse order
        c = Coordinates([[np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])]], ["lon_lat"])
        c2 = Coordinates([[np.array([1, 2, 3, 4, 5, 6]), np.array([7, 8, 9, 10, 11, 12])]], ["lon_lat"])

        # Ok for array coordinates
        t = c.transform("EPSG:32629")
        assert "lon_lat" in t.dims
        t_s = c2.transform("EPSG:32629")
        assert "lon_lat" in t_s.dims

        for d in ["lat", "lon"]:
            np.testing.assert_almost_equal(t[d].coordinates.ravel(), t_s[d].coordinates.ravel())

        t2 = t.transform(c.crs)
        t2_s = t_s.transform(c.crs)

        for d in ["lat", "lon"]:
            np.testing.assert_almost_equal(t2[d].coordinates, c[d].coordinates)
            np.testing.assert_almost_equal(t2_s[d].coordinates, c2[d].coordinates)

    def test_transform_missing_lat_lon(self):
        with pytest.raises(ValueError, match="Cannot transform lat coordinates without lon coordinates"):
            grid_coords = Coordinates([np.linspace(-10, 10, 21)], dims=["lat"])
            grid_coords.transform(crs="EPSG:2193")

        with pytest.raises(ValueError, match="Cannot transform lon coordinates without lat coordinates"):
            stack_coords = Coordinates([(np.linspace(-10, 10, 21), np.linspace(-30, -10, 21))], dims=["lon_time"])
            stack_coords.transform(crs="EPSG:2193")

        with pytest.raises(ValueError, match="nonadjacent lat and lon"):
            grid_coords = Coordinates([np.linspace(-10, 10, 21), [1], [1, 2, 3]], dims=["lat", "time", "lon"])
            grid_coords.transform(crs="EPSG:2193")

    def test_transform_same_crs_same_result(self):
        c1 = Coordinates(
            [[1, 2, 4], clinspace(0, 4, 4)], dims=["lat", "lon"], crs="+proj=longlat +datum=WGS84 +no_defs +vunits=m"
        )
        c2 = c1.transform("EPSG:4326")

        assert_array_equal(c2["lat"].coordinates, c1["lat"].coordinates)
        assert_array_equal(c2["lon"].coordinates, c1["lon"].coordinates)


class TestCoordinatesMethodSimplify(object):
    def test_simplify_array_to_uniform(self):
        c1 = Coordinates([[1, 2, 3, 4], [4, 6, 8]], dims=["lat", "lon"])
        c2 = Coordinates([[1, 2, 3, 5], [4, 6, 8]], dims=["lat", "lon"])
        c3 = Coordinates([clinspace(1, 4, 4), clinspace(4, 8, 3)], dims=["lat", "lon"])

        # array -> uniform
        assert c1.simplify() == c3.simplify()

        # array -> array
        assert c2.simplify() == c2.simplify()

        # uniform -> uniform
        assert c3.simplify() == c3.simplify()

    @pytest.mark.skip(reason="not implemented, spec uncertain")
    def test_simplify_stacked_to_unstacked_arrays(self):
        stacked = Coordinates([np.meshgrid([1, 2, 3, 5], [4, 6, 8])], dims=["lat_lon"])
        unstacked = Coordinates([[1, 2, 3, 5], [4, 6, 8]], dims=["lat", "lon"])

        assert stacked.simplify() == unstacked
        assert unstacked.simplify() == unstacked

    def test_stacked_to_unstacked_uniform(self):
        stacked = Coordinates([np.meshgrid([4, 6, 8], [1, 2, 3, 4])[::-1]], dims=["lat_lon"])
        unstacked_uniform = Coordinates([clinspace(1, 4, 4), clinspace(4, 8, 3)], dims=["lat", "lon"])

        # stacked grid -> uniform
        assert stacked.simplify() == unstacked_uniform

        # uniform -> uniform
        assert unstacked_uniform.simplify() == unstacked_uniform

    def test_stacked_to_affine(self):
        geotransform_rotated = (10.0, 1.879, -1.026, 20.0, 0.684, 2.819)
        affine = Coordinates([AffineCoordinates(geotransform=geotransform_rotated, shape=(4, 6))])
        stacked = Coordinates([StackedCoordinates([affine["lat_lon"]["lat"], affine["lat_lon"]["lon"]])])

        # stacked -> affine
        assert stacked.simplify() == affine

        # affine -> affine
        assert affine.simplify() == affine

    def test_affine_to_uniform(self):
        # NOTE: this assumes that podpac prefers unstacked UniformCoordinates to AffineCoordinates
        geotransform_northup = (10.0, 2.0, 0.0, 20.0, 0.0, -3.0)
        geotransform_rotated = (10.0, 1.879, -1.026, 20.0, 0.684, 2.819)

        c1 = Coordinates([AffineCoordinates(geotransform=geotransform_northup, shape=(4, 6))])
        c2 = Coordinates([AffineCoordinates(geotransform=geotransform_rotated, shape=(4, 6))])
        c3 = Coordinates([clinspace(18.5, 9.5, 4, name="lat"), clinspace(11, 21, 6, name="lon")])

        # unrotated affine -> unstacked uniform
        assert c1.simplify() == c3

        # rotated affine -> rotated affine
        assert c2.simplify() == c2

        # unstacked uniform -> unstacked uniform
        assert c3.simplify() == c3
