from datetime import datetime
import json

import pytest
import traitlets as tl
import numpy as np
import pandas as pd
import xarray as xr
import rasterio

import podpac
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.affine_coordinates import AffineCoordinates

# origin [10, 20], pixel size [3, 2], north up
GEOTRANSFORM_NORTHUP = (10.0, 2.0, 0.0, 20.0, 0.0, -3.0)

# origin [10, 20], step [2, 3], rotated 20 degrees
GEOTRANSFORM_ROTATED = (10.0, 1.879, -1.026, 20.0, 0.684, 2.819)
#                      (10.0, 1.879, -0.684, 20.0, 1.026, 2.819) # TODO which?

from podpac import Coordinates

UNIFORM = Coordinates.from_geotransform(geotransform=GEOTRANSFORM_NORTHUP, shape=(3, 4))


class TestAffineCoordinatesCreation(object):
    def test_init(self):
        c = AffineCoordinates(geotransform=GEOTRANSFORM_NORTHUP, shape=(3, 4))

        assert c.geotransform == GEOTRANSFORM_NORTHUP
        assert c.shape == (3, 4)
        assert c.is_affine
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert len(set(c.xdims)) == 2
        assert c.name == "lat_lon"
        repr(c)

    def test_rotated(self):
        c = AffineCoordinates(geotransform=GEOTRANSFORM_ROTATED, shape=(3, 4))

        assert c.geotransform == GEOTRANSFORM_ROTATED
        assert c.shape == (3, 4)
        assert c.is_affine
        assert c.dims == ("lat", "lon")
        assert c.udims == ("lat", "lon")
        assert len(set(c.xdims)) == 2
        assert c.name == "lat_lon"
        repr(c)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid shape"):
            AffineCoordinates(geotransform=GEOTRANSFORM_NORTHUP, shape=(-3, 4))

        with pytest.raises(ValueError, match="Invalid shape"):
            AffineCoordinates(geotransform=GEOTRANSFORM_NORTHUP, shape=(3, 0))

    # def test_copy(self):
    #     c = AffineCoordinates(geotransform=GEOTRANSFORM_NORTHUP, shape=(3, 4))
    #     c2 = c.copy()
    #     assert c2 is not c
    #     assert c2 == c


# class TestAffineCoordinatesStandardMethods(object):
#     def test_eq_type(self):
#         c = RotatedCoordinates(shape=(3, 4), theta=np.pi / 4, origin=[10, 20], step=[1.0, 2.0], dims=["lat", "lon"])
#         assert c != []

#     def test_eq_shape(self):
#         c1 = RotatedCoordinates(shape=(3, 4), theta=np.pi / 4, origin=[10, 20], step=[1.0, 2.0], dims=["lat", "lon"])
#         c2 = RotatedCoordinates(shape=(4, 3), theta=np.pi / 4, origin=[10, 20], step=[1.0, 2.0], dims=["lat", "lon"])
#         assert c1 != c2

#     def test_eq_affine(self):
#         c1 = RotatedCoordinates(shape=(3, 4), theta=np.pi / 4, origin=[10, 20], step=[1.0, 2.0], dims=["lat", "lon"])
#         c2 = RotatedCoordinates(shape=(3, 4), theta=np.pi / 4, origin=[10, 20], step=[1.0, 2.0], dims=["lat", "lon"])
#         c3 = RotatedCoordinates(shape=(3, 4), theta=np.pi / 3, origin=[10, 20], step=[1.0, 2.0], dims=["lat", "lon"])
#         c4 = RotatedCoordinates(shape=(3, 4), theta=np.pi / 4, origin=[11, 20], step=[1.0, 2.0], dims=["lat", "lon"])
#         c5 = RotatedCoordinates(shape=(3, 4), theta=np.pi / 4, origin=[10, 20], step=[1.1, 2.0], dims=["lat", "lon"])

#         assert c1 == c2
#         assert c1 != c3
#         assert c1 != c4
#         assert c1 != c5

#     def test_eq_dims(self):
#         c1 = RotatedCoordinates(shape=(3, 4), theta=np.pi / 4, origin=[10, 20], step=[1.0, 2.0], dims=["lat", "lon"])
#         c2 = RotatedCoordinates(shape=(3, 4), theta=np.pi / 4, origin=[10, 20], step=[1.0, 2.0], dims=["lon", "lat"])
#         assert c1 != c2


# class TestRotatedCoordinatesSerialization(object):
#     def test_definition(self):
#         c = RotatedCoordinates(shape=(3, 4), theta=np.pi / 4, origin=[10, 20], step=[1.0, 2.0], dims=["lat", "lon"])
#         d = c.definition

#         assert isinstance(d, dict)
#         json.dumps(d, cls=podpac.core.utils.JSONEncoder)  # test serializable
#         c2 = RotatedCoordinates.from_definition(d)
#         assert c2 == c

#     def test_from_definition_corner(self):
#         c1 = RotatedCoordinates(shape=(3, 4), theta=np.pi / 4, origin=[10, 20], corner=[15, 17], dims=["lat", "lon"])

#         d = {"shape": (3, 4), "theta": np.pi / 4, "origin": [10, 20], "corner": [15, 17], "dims": ["lat", "lon"]}
#         c2 = RotatedCoordinates.from_definition(d)

#         assert c1 == c2

#     def test_invalid_definition(self):
#         d = {"theta": np.pi / 4, "origin": [10, 20], "step": [1.0, 2.0], "dims": ["lat", "lon"]}
#         with pytest.raises(ValueError, match='RotatedCoordinates definition requires "shape"'):
#             RotatedCoordinates.from_definition(d)

#         d = {"shape": (3, 4), "origin": [10, 20], "step": [1.0, 2.0], "dims": ["lat", "lon"]}
#         with pytest.raises(ValueError, match='RotatedCoordinates definition requires "theta"'):
#             RotatedCoordinates.from_definition(d)

#         d = {"shape": (3, 4), "theta": np.pi / 4, "step": [1.0, 2.0], "dims": ["lat", "lon"]}
#         with pytest.raises(ValueError, match='RotatedCoordinates definition requires "origin"'):
#             RotatedCoordinates.from_definition(d)

#         d = {"shape": (3, 4), "theta": np.pi / 4, "origin": [10, 20], "dims": ["lat", "lon"]}
#         with pytest.raises(ValueError, match='RotatedCoordinates definition requires "step" or "corner"'):
#             RotatedCoordinates.from_definition(d)

#         d = {"shape": (3, 4), "theta": np.pi / 4, "origin": [10, 20], "step": [1.0, 2.0]}
#         with pytest.raises(ValueError, match='RotatedCoordinates definition requires "dims"'):
#             RotatedCoordinates.from_definition(d)

#     def test_full_definition(self):
#         c = RotatedCoordinates(shape=(3, 4), theta=np.pi / 4, origin=[10, 20], step=[1.0, 2.0], dims=["lat", "lon"])
#         d = c.full_definition

#         assert isinstance(d, dict)
#         assert set(d.keys()) == {"dims", "shape", "theta", "origin", "step"}
#         json.dumps(d, cls=podpac.core.utils.JSONEncoder)  # test serializable


class TestAffineCoordinatesProperties(object):
    def test_origin(self):
        c = AffineCoordinates(geotransform=GEOTRANSFORM_NORTHUP, shape=(3, 4))
        np.testing.assert_array_equal(c.origin, [20.0, 10.0])

    def test_origin_rotated(self):
        # lat, lon
        c = AffineCoordinates(geotransform=GEOTRANSFORM_ROTATED, shape=(3, 4))
        np.testing.assert_array_equal(c.origin, [20.0, 10.0])

    def test_coordinates(self):
        c = AffineCoordinates(geotransform=GEOTRANSFORM_NORTHUP, shape=(3, 4))

        assert c.coordinates.shape == (3, 4, 2)

        lat = c.coordinates[:, :, 0]
        lon = c.coordinates[:, :, 1]

        np.testing.assert_allclose(
            lat,
            [
                [18.5, 18.5, 18.5, 18.5],
                [15.5, 15.5, 15.5, 15.5],
                [12.5, 12.5, 12.5, 12.5],
            ],
        )

        np.testing.assert_allclose(
            lon,
            [
                [11.0, 13.0, 15.0, 17.0],
                [11.0, 13.0, 15.0, 17.0],
                [11.0, 13.0, 15.0, 17.0],
            ],
        )

    # def test_coordinates_rotated(self):
    #     c = AffineCoordinates(geotransform=GEOTRANSFORM_ROTATED, shape=(3, 4))

    #     assert c.coordinates.shape == (3, 4, 2)

    #     # lat
    #     np.testing.assert_allclose(
    #         c.coordinates[:, :, 0],
    #         [
    #             [20.000, 22.819, 25.638, 28.457],
    #             [20.684, 23.503, 26.322, 29.141],
    #             [21.368, 24.187, 27.006, 29.825]
    #         ],
    #     )

    #     # lon
    #     np.testing.assert_allclose(
    #         c.coordinates[:, :, 1],
    #         [
    #             [10.000,  8.974,  7.948,  6.922],
    #             [11.879, 10.853,  9.827,  8.801],
    #             [13.758, 12.732, 11.706, 10.680],
    #         ],
    #     )


# class TestRotatedCoordinatesIndexing(object):
#     def test_get_dim(self):
#         c = RotatedCoordinates(shape=(3, 4), theta=np.pi / 4, origin=[10, 20], step=[1.0, 2.0], dims=["lat", "lon"])

#         lat = c["lat"]
#         lon = c["lon"]
#         assert isinstance(lat, ArrayCoordinates1d)
#         assert isinstance(lon, ArrayCoordinates1d)
#         assert lat.name == "lat"
#         assert lon.name == "lon"
#         assert_equal(lat.coordinates, c.coordinates[0])
#         assert_equal(lon.coordinates, c.coordinates[1])

#         with pytest.raises(KeyError, match="Dimension .* not found"):
#             c["other"]

#     def test_get_index_slices(self):
#         c = RotatedCoordinates(shape=(5, 7), theta=np.pi / 4, origin=[10, 20], step=[1.0, 2.0], dims=["lat", "lon"])

#         # full
#         c2 = c[1:4, 2:4]
#         assert isinstance(c2, RotatedCoordinates)
#         assert c2.shape == (3, 2)
#         assert c2.theta == c.theta
#         np.testing.assert_allclose(c2.origin, [c.coordinates[0][1, 2], c.coordinates[1][1, 2]])
#         np.testing.assert_allclose(c2.corner, [c.coordinates[0][3, 3], c.coordinates[1][3, 3]])
#         assert c2.dims == c.dims
#         np.testing.assert_allclose(c2.coordinates[0], c.coordinates[0][1:4, 2:4])
#         np.testing.assert_allclose(c2.coordinates[1], c.coordinates[1][1:4, 2:4])

#         # partial/implicit
#         c2 = c[1:4]
#         assert isinstance(c2, RotatedCoordinates)
#         assert c2.shape == (3, 7)
#         assert c2.theta == c.theta
#         np.testing.assert_allclose(c2.origin, c.coordinates[0][1, 0], c.coordinates[1][1, 0])
#         np.testing.assert_allclose(c2.corner, c.coordinates[0][3, -1], c.coordinates[1][3, -1])
#         assert c2.dims == c.dims
#         np.testing.assert_allclose(c2.coordinates[0], c.coordinates[0][1:4])
#         np.testing.assert_allclose(c2.coordinates[1], c.coordinates[1][1:4])

#         # stepped
#         c2 = c[1:4:2, 2:4]
#         assert isinstance(c2, RotatedCoordinates)
#         assert c2.shape == (2, 2)
#         assert c2.theta == c.theta
#         np.testing.assert_allclose(c2.origin, [c.coordinates[0][1, 2], c.coordinates[1][1, 2]])
#         np.testing.assert_allclose(c2.corner, [c.coordinates[0][3, 3], c.coordinates[1][3, 3]])
#         assert c2.dims == c.dims
#         np.testing.assert_allclose(c2.coordinates[0], c.coordinates[0][1:4:2, 2:4])
#         np.testing.assert_allclose(c2.coordinates[1], c.coordinates[1][1:4:2, 2:4])

#         # reversed
#         c2 = c[4:1:-1, 2:4]
#         assert isinstance(c2, RotatedCoordinates)
#         assert c2.shape == (3, 2)
#         assert c2.theta == c.theta
#         np.testing.assert_allclose(c2.origin, c.coordinates[0][1, 2], c.coordinates[0][1, 2])
#         np.testing.assert_allclose(c2.corner, c.coordinates[0][3, 3], c.coordinates[0][3, 3])
#         assert c2.dims == c.dims
#         np.testing.assert_allclose(c2.coordinates[0], c.coordinates[0][4:1:-1, 2:4])
#         np.testing.assert_allclose(c2.coordinates[1], c.coordinates[1][4:1:-1, 2:4])

#     def test_get_index_fallback(self):
#         c = RotatedCoordinates(shape=(5, 7), theta=np.pi / 4, origin=[10, 20], step=[1.0, 2.0], dims=["lat", "lon"])
#         lat, lon = c.coordinates

#         I = [3, 1]
#         J = slice(1, 4)
#         B = lat > 6

#         # int/slice/indices
#         c2 = c[I, J]
#         assert isinstance(c2, StackedCoordinates)
#         assert c2.shape == (2, 3)
#         assert c2.dims == c.dims
#         assert_equal(c2["lat"].coordinates, lat[I, J])
#         assert_equal(c2["lon"].coordinates, lon[I, J])

#         # boolean
#         c2 = c[B]
#         assert isinstance(c2, StackedCoordinates)
#         assert c2.shape == (21,)
#         assert c2.dims == c.dims
#         assert_equal(c2["lat"].coordinates, lat[B])
#         assert_equal(c2["lon"].coordinates, lon[B])


# class TestRotatedCoordinatesSelection(object):
#     def test_select_single(self):
#         c = RotatedCoordinates([LAT, LON], dims=['lat', 'lon'])

#         # single dimension
#         bounds = {'lat': [0.25, .55]}
#         E0, E1 = [0, 1, 1, 1], [3, 0, 1, 2] # expected

#         s = c.select(bounds)
#         assert isinstance(s, StackedCoordinates)
#         assert s == c[E0, E1]

#         s, I = c.select(bounds, return_index=True)
#         assert isinstance(s, StackedCoordinates)
#         assert s == c[I]
#         assert_equal(I[0], E0)
#         assert_equal(I[1], E1)

#         # a different single dimension
#         bounds = {'lon': [12.5, 17.5]}
#         E0, E1 = [0, 1, 1, 1, 1, 2], [3, 0, 1, 2, 3, 0]

#         s = c.select(bounds)
#         assert isinstance(s, StackedCoordinates)
#         assert s == c[E0, E1]

#         s, I = c.select(bounds, return_index=True)
#         assert isinstance(s, StackedCoordinates)
#         assert s == c[I]
#         assert_equal(I[0], E0)
#         assert_equal(I[1], E1)

#         # outer
#         bounds = {'lat': [0.25, .75]}
#         E0, E1 = [0, 0, 1, 1, 1, 1, 2, 2], [2, 3, 0, 1, 2, 3, 0, 1]

#         s = c.select(bounds, outer=True)
#         assert isinstance(s, StackedCoordinates)
#         assert s == c[E0, E1]

#         s, I = c.select(bounds, outer=True, return_index=True)
#         assert isinstance(s, StackedCoordinates)
#         assert s == c[E0, E1]
#         assert_equal(I[0], E0)
#         assert_equal(I[1], E1)

#         # no matching dimension
#         bounds = {'alt': [0, 10]}
#         s = c.select(bounds)
#         assert s == c

#         s, I = c.select(bounds, return_index=True)
#         assert s == c[I]
#         assert s == c

#     def test_select_multiple(self):
#         c = RotatedCoordinates([LAT, LON], dims=['lat', 'lon'])

#         # this should be the AND of both intersections
#         bounds = {'lat': [0.25, 0.95], 'lon': [10.5, 17.5]}
#         E0, E1 = [0, 1, 1, 1, 1, 2], [3, 0, 1, 2, 3, 0]
#         s = c.select(bounds)
#         assert s == c[E0, E1]

#         s, I = c.select(bounds, return_index=True)
#         assert s == c[E0, E1]
#         assert_equal(I[0], E0)
#         assert_equal(I[1], E1)

#     def test_intersect(self):
#         c = RotatedCoordinates([LAT, LON], dims=['lat', 'lon'])


#         other_lat = ArrayCoordinates1d([0.25, 0.5, .95], name='lat')
#         other_lon = ArrayCoordinates1d([10.5, 15, 17.5], name='lon')

#         # single other
#         E0, E1 = [0, 1, 1, 1, 1, 2, 2, 2], [3, 0, 1, 2, 3, 0, 1, 2]
#         s = c.intersect(other_lat)
#         assert s == c[E0, E1]

#         s, I = c.intersect(other_lat, return_index=True)
#         assert s == c[E0, E1]
#         assert s == c[I]

#         E0, E1 = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2], [2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
#         s = c.intersect(other_lat, outer=True)
#         assert s == c[E0, E1]

#         E0, E1 = [0, 0, 0, 1, 1, 1, 1, 2], [1, 2, 3, 0, 1, 2, 3, 0]
#         s = c.intersect(other_lon)
#         assert s == c[E0, E1]

#         # multiple, in various ways
#         E0, E1 = [0, 1, 1, 1, 1, 2], [3, 0, 1, 2, 3, 0]

#         other = StackedCoordinates([other_lat, other_lon])
#         s = c.intersect(other)
#         assert s == c[E0, E1]

#         other = StackedCoordinates([other_lon, other_lat])
#         s = c.intersect(other)
#         assert s == c[E0, E1]

#         from podpac.coordinates import Coordinates
#         other = Coordinates([other_lat, other_lon])
#         s = c.intersect(other)
#         assert s == c[E0, E1]

#         # full
#         other = Coordinates(['2018-01-01'], dims=['time'])
#         s = c.intersect(other)
#         assert s == c

#         s, I = c.intersect(other, return_index=True)
#         assert s == c
#         assert s == c[I]

#     def test_intersect_invalid(self):
#         c = RotatedCoordinates([LAT, LON], dims=['lat', 'lon'])

#         with pytest.raises(TypeError, match="Cannot intersect with type"):
#             c.intersect({})

#         with pytest.raises(ValueError, match="Cannot intersect mismatched dtypes"):
#             c.intersect(ArrayCoordinates1d(['2018-01-01'], name='lat'))
