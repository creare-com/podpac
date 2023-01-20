from __future__ import division, unicode_literals, print_function, absolute_import

import io
import tempfile

import pytest
import numpy as np
import xarray as xr
from pint.errors import DimensionalityError

from podpac.core.coordinates import Coordinates, clinspace, AffineCoordinates
from podpac.core.style import Style

from podpac.core.units_data_array import ureg
from podpac.core.units_data_array import UnitsDataArray
from podpac.core.units_data_array import to_image

from podpac.data import Array, Rasterio


class TestUnitDataArray(object):
    def test_no_units_to_base_units_has_no_units(self):
        a = UnitsDataArray(
            np.arange(24, dtype=np.float64).reshape((3, 4, 2)),
            coords={"x": np.arange(3), "y": np.arange(4) * 10, "z": np.arange(2) + 100},
            dims=["x", "y", "z"],
        )
        b = a.to_base_units()
        assert b.attrs.get("units", None) is None

    def test_reductions_maintain_units(self):
        n_lats = 3
        n_lons = 4
        n_alts = 2
        a = UnitsDataArray(
            np.arange(n_lats * n_lons * n_alts).reshape((n_lats, n_lons, n_alts)),
            dims=["lat", "lon", "alt"],
            attrs={"units": ureg.meter},
        )
        assert a.mean(axis=0).attrs.get("units", None) is not None
        assert a.sum(axis=0).attrs.get("units", None) is not None
        assert a.cumsum(axis=0).attrs.get("units", None) is not None
        assert a.min(axis=0).attrs.get("units", None) is not None
        assert a.max("lon").attrs.get("units", None) is not None
        assert np.mean(a, axis=0).attrs.get("units", None) is not None
        assert np.sum(a, axis=0).attrs.get("units", None) is not None
        assert np.cumsum(a, axis=0).attrs.get("units", None) is not None
        assert np.min(a, axis=0).attrs.get("units", None) is not None
        assert np.max(a, axis=0).attrs.get("units", None) is not None

    def test_reductions_over_named_axes(self):
        n_lats = 3
        n_lons = 4
        n_alts = 2
        a = UnitsDataArray(
            np.arange(n_lats * n_lons * n_alts).reshape((n_lats, n_lons, n_alts)),
            dims=["lat", "lon", "alt"],
            attrs={"units": ureg.meter},
        )
        assert len(a.mean(["lat", "lon"]).data) == 2

    def test_serialization_deserialization(self):
        n_lats = 3
        n_lons = 4
        n_alts = 2
        a = UnitsDataArray(
            np.arange(n_lats * n_lons * n_alts).reshape((n_lats, n_lons, n_alts)),
            dims=["lat", "lon", "alt"],
            attrs={"units": ureg.meter, "layer_style": Style()},
        )
        f = a.to_netcdf()
        b = UnitsDataArray(xr.open_dataarray(f))
        assert a.attrs["units"] == b.attrs["units"]
        assert a.attrs["layer_style"].json == b.attrs["layer_style"].json

    def test_pow(self):
        n_lats = 3
        n_lons = 4
        n_alts = 2

        a = UnitsDataArray(
            np.arange(n_lats * n_lons * n_alts).reshape((n_lats, n_lons, n_alts)),
            dims=["lat", "lon", "alt"],
            attrs={"units": ureg.meter},
        )
        assert (a**2).attrs["units"] == ureg.meter**2

    def test_set_to_value_using_UnitsDataArray_as_mask_does_nothing_if_mask_has_dim_not_in_array(self):
        a = UnitsDataArray(
            np.arange(24, dtype=np.float64).reshape((3, 4, 2)),
            coords={"x": np.arange(3), "y": np.arange(4) * 10, "z": np.arange(2) + 100},
            dims=["x", "y", "z"],
        )
        b = UnitsDataArray(
            np.arange(24, dtype=np.float64).reshape((3, 4, 2)),
            coords={"i": np.arange(3), "y": np.arange(4) * 10, "z": np.arange(2) + 100},
            dims=["i", "y", "z"],
        )

        mask = b > -10
        value = np.nan

        a.set(value, mask)
        # dims of a remain unchanged
        assert not np.any(np.isnan(a.data))

    def test_set_to_value_using_UnitsDataArray_as_mask_broadcasts_to_dimensions_not_in_mask(self):
        a = UnitsDataArray(
            np.arange(24, dtype=np.float64).reshape((3, 4, 2)),
            coords={"x": np.arange(3), "y": np.arange(4) * 10, "z": np.arange(2) + 100},
            dims=["x", "y", "z"],
        )
        b = a[0, :, :]
        b = b < 3

        mask = b.transpose(*("z", "y"))
        value = np.nan

        a.set(value, mask)
        # dims of a remain unchanged
        assert np.all(np.array(a.dims) == np.array(("x", "y", "z")))
        # shape of a remains unchanged
        assert np.all(np.array(a.values.shape) == np.array((3, 4, 2)))
        # a.set was broadcast across the 'x' dimension
        for x in range(3):
            for y in range(4):
                for z in range(2):
                    if y == 0 and (z == 0 or z == 1):
                        assert np.isnan(a[x, y, z])
                    elif y == 1 and z == 0:
                        assert np.isnan(a[x, y, z])
                    else:
                        assert not np.isnan(a[x, y, z])

    def test_get_item_with_1d_units_data_array_as_key_boradcasts_to_correct_dimension(self):
        a = UnitsDataArray(
            np.arange(24).reshape((3, 4, 2)),
            coords={"x": np.arange(3), "y": np.arange(4) * 10, "z": np.arange(2) + 100},
            dims=["x", "y", "z"],
        )
        b = a[0, :, 0]
        b = b < 3
        c = a[b]
        # dims of a remain unchanged
        assert np.all(np.array(a.dims) == np.array(("x", "y", "z")))
        # shape of a remains unchanged
        assert np.all(np.array(a.values.shape) == np.array((3, 4, 2)))
        # dims of a remain unchanged
        assert np.all(np.array(c.dims) == np.array(("x", "y", "z")))
        # shape of a remains unchanged
        assert np.all(np.array(c.values.shape) == np.array((3, 2, 2)))
        # a[b] was broadcast across the 'y' dimension
        for x in range(3):
            for y in range(2):
                for z in range(2):
                    c[x, y, z] in [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]

    def test_get_item_with_units_data_array_as_key_throws_index_error(self):
        """
        I believe this is not the desired behavior. We should look at this function.
        """
        a = UnitsDataArray(
            np.arange(24, dtype=np.float64).reshape((3, 4, 2)),
            coords={"x": np.arange(3), "y": np.arange(4) * 10, "z": np.arange(2) + 100},
            dims=["x", "y", "z"],
        )
        b = a < 3
        with pytest.raises(IndexError):
            a[b]

    def test_partial_transpose_specify_just_lon_swaps_lat_lon(self):
        n_lats = 3
        n_lons = 4
        lat_lon = UnitsDataArray(
            np.arange(12).reshape((n_lats, n_lons)), dims=["lat", "lon"], attrs={"units": ureg.meter}
        )
        lon_lat = lat_lon.part_transpose(["lon"])
        for lat in range(n_lats):
            for lon in range(n_lons):
                lat_lon[lat, lon] == lon_lat[lon, lat]

    def test_partial_transpose_specify_both_swaps_lat_lon(self):
        n_lats = 3
        n_lons = 4
        lat_lon = UnitsDataArray(
            np.arange(12).reshape((n_lats, n_lons)), dims=["lat", "lon"], attrs={"units": ureg.meter}
        )
        lon_lat = lat_lon.part_transpose(["lon", "lat"])
        for lat in range(n_lats):
            for lon in range(n_lons):
                lat_lon[lat, lon] == lon_lat[lon, lat]

    def test_partial_transpose_specify_none_leaves_lat_lon_untouched(self):
        n_lats = 3
        n_lons = 4
        lat_lon = UnitsDataArray(
            np.arange(12).reshape((n_lats, n_lons)), dims=["lat", "lon"], attrs={"units": ureg.meter}
        )
        lat_lon_2 = lat_lon.part_transpose([])
        for lat in range(n_lats):
            for lon in range(n_lons):
                lat_lon[lat, lon] == lat_lon_2[lat, lon]

    def test_no_units_coord(self):
        a1 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={})
        a2 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={})
        a3 = a1 + a2
        a3b = a2 + a1
        a4 = a1 > a2
        a5 = a1 < a2
        a6 = a1 == a2
        a7 = a1 * a2
        a8 = a2 / a1
        a9 = a1 // a2
        a10 = a1 % a2

    def test_first_units_coord(self):
        a1 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={"units": ureg.meter})
        a2 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={})
        with pytest.raises(DimensionalityError):
            a3 = a1 + a2
        with pytest.raises(DimensionalityError):
            a3b = a2 + a1
        with pytest.raises(DimensionalityError):
            a4 = a1 > a2
        with pytest.raises(DimensionalityError):
            a5 = a1 < a2
        with pytest.raises(DimensionalityError):
            a6 = a1 == a2
        a7 = a1 * a2
        a8 = a2 / a1
        with pytest.raises(DimensionalityError):
            a9 = a1 // a2
        with pytest.raises(DimensionalityError):
            a10 = a1 % a2

    def test_second_units_coord(self):
        a1 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={})
        a2 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={"units": ureg.inch})
        with pytest.raises(DimensionalityError):
            a3 = a1 + a2
        with pytest.raises(DimensionalityError):
            a3b = a2 + a1
        with pytest.raises(DimensionalityError):
            a4 = a1 > a2
        with pytest.raises(DimensionalityError):
            a5 = a1 < a2
        with pytest.raises(DimensionalityError):
            a6 = a1 == a2
        a7 = a1 * a2
        a8 = a2 / a1
        with pytest.raises(DimensionalityError):
            a9 = a1 // a2
        with pytest.raises(DimensionalityError):
            a10 = a1 % a2

    def test_units_allpass(self):
        a1 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={"units": ureg.meter})
        a2 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={"units": ureg.inch})
        a3 = a1 + a2
        assert a3[0, 0].data[()] == (1 * ureg.meter + 1 * ureg.inch).to(ureg.meter).magnitude

        a3b = a2 + a1
        assert a3b[0, 0].data[()] == (1 * ureg.meter + 1 * ureg.inch).to(ureg.inch).magnitude

        a4 = a1 > a2
        assert a4[0, 0].data[()] == True

        a5 = a1 < a2
        assert a5[0, 0].data[()] == False

        a6 = a1 == a2
        assert a6[0, 0].data[()] == False

        a7 = a1 * a2
        assert a7[0, 0].to(ureg.m**2).data[()] == (1 * ureg.meter * ureg.inch).to(ureg.meter**2).magnitude

        a8 = a2 / a1
        assert a8[0, 0].to_base_units().data[()] == (1 * ureg.inch / ureg.meter).to_base_units().magnitude

        a9 = a1 // a2
        assert a9[0, 0].to_base_units().data[()] == ((1 * ureg.meter) // (1 * ureg.inch)).to_base_units().magnitude

        a10 = a1 % a2
        assert a10[0, 0].to_base_units().data[()] == ((1 * ureg.meter) % (1 * ureg.inch)).to_base_units().magnitude

    def test_units_somefail(self):
        a1 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={"units": ureg.meter})
        a2 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={"units": ureg.kelvin})
        with pytest.raises(DimensionalityError):
            a3 = a1 + a2
        with pytest.raises(DimensionalityError):
            a3b = a2 + a1
        with pytest.raises(DimensionalityError):
            a4 = a1 > a2
        with pytest.raises(DimensionalityError):
            a5 = a1 < a2
        with pytest.raises(DimensionalityError):
            a6 = a1 == a2

        a7 = a1 * a2
        assert a7[0, 0].to(ureg.meter * ureg.kelvin).data[()] == (1 * ureg.meter * ureg.kelvin).magnitude

        a8 = a1 / a2
        assert a8[0, 0].to(ureg.meter / ureg.kelvin).data[()] == (1 * ureg.meter / ureg.kelvin).magnitude

        with pytest.raises(DimensionalityError):
            a9 = a1 // a2

        with pytest.raises(DimensionalityError):
            a10 = a1 % a2

    def test_to_image(self):
        uda = UnitsDataArray(np.ones((10, 10)))
        assert isinstance(uda.to_image(return_base64=True), bytes)
        assert isinstance(uda.to_image(), io.BytesIO)

    def test_to_image_vmin_vmax(self):
        uda = UnitsDataArray(np.ones((10, 10)))
        assert isinstance(uda.to_image(vmin=0, vmax=2, return_base64=True), bytes)
        assert isinstance(uda.to_image(vmin=0, vmax=2), io.BytesIO)

    def test_ufuncs(self):
        a1 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={"units": ureg.meter})
        a2 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={"units": ureg.kelvin})

        np.sqrt(a1)
        np.mean(a1)
        np.min(a1)
        np.max(a1)
        a1**2

        # These don't have units!
        np.dot(a2.T, a1)
        np.std(a1)
        np.var(a1)

    def test_keep_attrs(self):
        # This tests #265
        # Create Nodes to use the convience methods for making units data arrays
        a1 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={"units": ureg.meter, "test": "test"})
        a2 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={"units": ureg.yard})

        assert "test" in (a1 + a2).attrs
        assert "test" in (a1 * a2).attrs

        # No units
        a1 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"], attrs={"test": "test"})
        a2 = UnitsDataArray(np.ones((4, 3)), dims=["lat", "lon"])

        assert "test" in (a1 + 1).attrs
        assert "test" in (a1 + a2).attrs
        assert "test" in (a1 * 1).attrs
        assert "test" in (a1 * a2).attrs

        # Order is important
        assert "test" not in (1 + a1).attrs
        assert "test" not in (a2 + a1).attrs
        assert "test" not in (1 * a1).attrs
        assert "test" not in (a2 * a1).attrs


class TestCreateDataArray(object):
    @classmethod
    def setup_class(cls):
        cls.coords = Coordinates([[0, 1, 2], [0, 1, 2, 3]], dims=["lat", "lon"])

    def test_default(self):
        a = UnitsDataArray.create(self.coords)
        assert isinstance(a, UnitsDataArray)
        assert a.shape == self.coords.shape
        assert np.all(np.isnan(a))

    def test_empty(self):
        a = UnitsDataArray.create(self.coords, data=None)
        assert isinstance(a, UnitsDataArray)
        assert a.shape == self.coords.shape
        assert a.dtype == float

        a = UnitsDataArray.create(self.coords, data=None, dtype=bool)
        assert isinstance(a, UnitsDataArray)
        assert a.shape == self.coords.shape
        assert a.dtype == bool

    def test_zeros(self):
        a = UnitsDataArray.create(self.coords, data=0)
        assert isinstance(a, UnitsDataArray)
        assert a.shape == self.coords.shape
        assert a.dtype == float
        assert np.all(a == 0.0)

        a = UnitsDataArray.create(self.coords, data=0, dtype=bool)
        assert isinstance(a, UnitsDataArray)
        assert a.shape == self.coords.shape
        assert a.dtype == bool
        assert np.all(~a)

    def test_ones(self):
        a = UnitsDataArray.create(self.coords, data=1)
        assert isinstance(a, UnitsDataArray)
        assert a.shape == self.coords.shape
        assert a.dtype == float
        assert np.all(a == 1.0)

        a = UnitsDataArray.create(self.coords, data=1, dtype=bool)
        assert isinstance(a, UnitsDataArray)
        assert a.shape == self.coords.shape
        assert a.dtype == bool
        assert np.all(a)

    def test_full(self):
        a = UnitsDataArray.create(self.coords, data=10)
        assert isinstance(a, UnitsDataArray)
        assert a.shape == self.coords.shape
        assert a.dtype == float
        assert np.all(a == 10)

        a = UnitsDataArray.create(self.coords, data=10, dtype=int)
        assert isinstance(a, UnitsDataArray)
        assert a.shape == self.coords.shape
        assert a.dtype == int
        assert np.all(a == 10)

    def test_array(self):
        data = np.random.random(self.coords.shape)
        a = UnitsDataArray.create(self.coords, data=data)
        assert isinstance(a, UnitsDataArray)
        assert a.dtype == float
        np.testing.assert_equal(a.data, data)

        data = np.round(10 * np.random.random(self.coords.shape))
        a = UnitsDataArray.create(self.coords, data=data, dtype=int)
        assert isinstance(a, UnitsDataArray)
        assert a.dtype == int
        np.testing.assert_equal(a.data, data.astype(int))

    def test_outputs(self):
        a = UnitsDataArray.create(self.coords, outputs=["a", "b", "c"])
        assert a.dims == self.coords.dims + ("output",)
        np.testing.assert_array_equal(a["output"], ["a", "b", "c"])

        a = UnitsDataArray.create(self.coords, data=0, outputs=["a", "b", "c"])
        assert a.dims == self.coords.dims + ("output",)
        np.testing.assert_array_equal(a["output"], ["a", "b", "c"])

        a = UnitsDataArray.create(self.coords, data=1, outputs=["a", "b", "c"])
        assert a.dims == self.coords.dims + ("output",)
        np.testing.assert_array_equal(a["output"], ["a", "b", "c"])

        a = UnitsDataArray.create(self.coords, data=np.nan, outputs=["a", "b", "c"])
        assert a.dims == self.coords.dims + ("output",)
        np.testing.assert_array_equal(a["output"], ["a", "b", "c"])

        data = np.random.random(self.coords.shape + (3,))
        a = UnitsDataArray.create(self.coords, data=data, outputs=["a", "b", "c"])
        assert a.dims == self.coords.dims + ("output",)
        np.testing.assert_array_equal(a["output"], ["a", "b", "c"])

        data = np.random.random(self.coords.shape + (2,))
        with pytest.raises(ValueError, match="data with shape .* does not match"):
            a = UnitsDataArray.create(self.coords, data=data, outputs=["a", "b", "c"])

        data = np.random.random(self.coords.shape)
        with pytest.raises(ValueError, match="data with shape .* does not match"):
            a = UnitsDataArray.create(self.coords, data=data, outputs=["a", "b", "c"])

    def test_invalid_coords(self):
        with pytest.raises(TypeError):
            UnitsDataArray.create((3, 4))


class TestOpenDataArray(object):
    def test_open_after_create(self):
        coords = Coordinates([[0, 1, 2], [0, 1, 2, 3]], dims=["lat", "lon"])
        uda_1 = UnitsDataArray.create(coords, data=np.random.rand(3, 4))
        ncdf = uda_1.to_netcdf()
        uda_2 = UnitsDataArray.open(ncdf)

        assert isinstance(uda_2, UnitsDataArray)
        assert np.all(uda_2.data == uda_1.data)

    def test_open_after_create_with_attrs(self):
        coords = Coordinates([[0, 1, 2], [0, 1, 2, 3]], dims=["lat", "lon"], crs="EPSG:4193")
        uda_1 = UnitsDataArray.create(coords, data=np.random.rand(3, 4), attrs={"some_attr": 5})
        ncdf = uda_1.to_netcdf()
        uda_2 = UnitsDataArray.open(ncdf)

        assert isinstance(uda_2, UnitsDataArray)
        assert np.all(uda_2.data == uda_1.data)

        assert "some_attr" in uda_2.attrs
        assert uda_2.attrs.get("some_attr") == uda_1.attrs.get("some_attr")

        assert "crs" in uda_2.attrs
        assert uda_2.attrs.get("crs") == uda_1.attrs.get("crs")

    def test_open_after_eval(self):

        # mock node
        data = np.random.rand(5, 5)
        lat = np.linspace(-10, 10, 5)
        lon = np.linspace(-10, 10, 5)
        native_coords = Coordinates([lat, lon], ["lat", "lon"])
        node = Array(source=data, coordinates=native_coords)
        uda = node.eval(node.coordinates)

        ncdf = uda.to_netcdf()
        uda_2 = UnitsDataArray.open(ncdf)

        assert isinstance(uda_2, UnitsDataArray)
        assert np.all(uda_2.data == uda.data)

        assert "layer_style" in uda_2.attrs
        assert uda_2.attrs.get("layer_style").json == uda.attrs.get("layer_style").json

        assert "crs" in uda_2.attrs
        assert uda_2.attrs.get("crs") == uda.attrs.get("crs")


class TestToImage(object):
    def test_to_image(self):
        data = np.ones((10, 10))
        assert isinstance(to_image(UnitsDataArray(data), return_base64=True), bytes)  # UnitsDataArray input
        assert isinstance(to_image(xr.DataArray(data), return_base64=True), bytes)  # xr.DataArray input
        assert isinstance(to_image(data, return_base64=True), bytes)  # np.ndarray input
        assert isinstance(to_image(np.array([data]), return_base64=True), bytes)  # squeeze

    def test_to_image_vmin_vmax(self):
        data = np.ones((10, 10))
        assert isinstance(to_image(data, vmin=0, vmax=2, return_base64=True), bytes)


class TestToGeoTiff(object):
    def make_square_array(self, order=1, bands=1):
        node = Array(
            source=np.arange(8 * bands).reshape(3 - order, 3 + order, bands),
            coordinates=Coordinates([clinspace(4, 0, 2, "lat"), clinspace(1, 4, 4, "lon")][::order], crs="EPSG:4326"),
            outputs=[str(s) for s in list(range(bands))],
        )
        return node

    def make_rot_array(self, order=1, bands=1):
        if order == 1:
            geotransform = (10.0, 1.879, -1.026, 20.0, 0.684, 2.819)
        else:
            # I think this requires changing the geotransform? Not yet supported
            raise NotImplementedError("TODO")

        rc = AffineCoordinates(geotransform=geotransform, shape=(2, 4))
        c = Coordinates([rc], crs="EPSG:4326")
        node = Array(
            source=np.arange(8 * bands).reshape(3 - order, 3 + order, bands),
            coordinates=c,
            outputs=[str(s) for s in list(range(bands))],
        )
        return node

    def test_to_geotiff_roundtrip_1band(self):
        # lat/lon order, usual
        node = self.make_square_array()
        out = node.eval(node.coordinates)
        with tempfile.NamedTemporaryFile("wb") as fp:
            out.to_geotiff(fp)
            fp.write(b"a")  # for some reason needed to get good comparison

            fp.seek(0)
            rnode = Rasterio(source=fp.name, outputs=node.outputs)
            assert rnode.coordinates == node.coordinates

            rout = rnode.eval(rnode.coordinates)
            np.testing.assert_almost_equal(rout.data, out.data)

        # lon/lat order, unusual
        node = self.make_square_array(order=-1)
        out = node.eval(node.coordinates)
        with tempfile.NamedTemporaryFile("wb") as fp:
            out.to_geotiff(fp)
            fp.write(b"a")  # for some reason needed to get good comparison

            fp.seek(0)
            rnode = Rasterio(source=fp.name, outputs=node.outputs)
            assert rnode.coordinates == node.coordinates

            rout = rnode.eval(rnode.coordinates)
            np.testing.assert_almost_equal(rout.data, out.data)

    def test_to_geotiff_roundtrip_2band(self):
        # lat/lon order, usual
        node = self.make_square_array(bands=2)
        out = node.eval(node.coordinates)
        with tempfile.NamedTemporaryFile("wb") as fp:
            out.to_geotiff(fp)
            fp.write(b"a")  # for some reason needed to get good comparison

            fp.seek(0)
            rnode = Rasterio(source=fp.name, outputs=node.outputs)
            assert rnode.coordinates == node.coordinates

            rout = rnode.eval(rnode.coordinates)
            np.testing.assert_almost_equal(rout.data, out.data)

        # lon/lat order, unsual
        node = self.make_square_array(order=-1, bands=2)
        out = node.eval(node.coordinates)
        with tempfile.NamedTemporaryFile("wb") as fp:
            out.to_geotiff(fp)
            fp.write(b"a")  # for some reason needed to get good comparison

            fp.seek(0)
            rnode = Rasterio(source=fp.name, outputs=node.outputs)
            assert rnode.coordinates == node.coordinates

            rout = rnode.eval(rnode.coordinates)
            np.testing.assert_almost_equal(rout.data, out.data)

            # Check single output
            fp.seek(0)
            rnode = Rasterio(source=fp.name, outputs=node.outputs, output=node.outputs[1])
            rout = rnode.eval(rnode.coordinates)
            np.testing.assert_almost_equal(out.data[..., 1], rout.data)

            # Check single band 1
            fp.seek(0)
            rnode = Rasterio(source=fp.name, band=1)
            rout = rnode.eval(rnode.coordinates)
            np.testing.assert_almost_equal(out.data[..., 0], rout.data)

            # Check single band 2
            fp.seek(0)
            rnode = Rasterio(source=fp.name, band=2)
            rout = rnode.eval(rnode.coordinates)
            np.testing.assert_almost_equal(out.data[..., 1], rout.data)

    def test_to_geotiff_roundtrip_rotcoords(self):
        # lat/lon order, usual
        node = self.make_rot_array()

        out = node.eval(node.coordinates)

        with tempfile.NamedTemporaryFile("wb") as fp:
            out.to_geotiff(fp)
            fp.write(b"a")  # for some reason needed to get good comparison

            fp.seek(0)
            rnode = Rasterio(source=fp.name, outputs=node.outputs, mode="r")
            assert node.coordinates == rnode.coordinates

            rout = rnode.eval(rnode.coordinates)
            np.testing.assert_almost_equal(out.data, rout.data)

        # # lon/lat order, unsual
        # node = self.make_square_array(order=-1)
        # out = node.eval(node.coordinates)
        # with tempfile.NamedTemporaryFile("wb") as fp:
        #     out.to_geotiff(fp)
        #     fp.write(b"a")  # for some reason needed to get good comparison

        #     fp.seek(0)
        #     rnode = Rasterio(source=fp.name, outputs=node.outputs)
        #     assert node.coordinates == rnode.coordinates

        #     rout = rnode.eval(rnode.coordinates)
        #     np.testing.assert_almost_equal(out.data, rout.data)
