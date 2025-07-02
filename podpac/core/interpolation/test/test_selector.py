import pytest
import traitlets as tl
import numpy as np

from podpac.core.node import Node
from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.interpolation.selector import Selector, _higher_precision_time_coords1d

_ERR_MSG = "Selection using source {} and request {} failed with {} != {} (truth)"


class TestSelector(object):
    lat_coarse = np.linspace(0, 1, 3)
    lat_fine = np.linspace(-0.1, 1.15, 8)
    lat_random_fine = [0.72, -0.05, 1.3, 0.35, 0.22, 0.543, 0.44, 0.971]
    lat_random_coarse = [0.64, -0.25, 0.83]
    lon_coarse = lat_coarse + 1
    lon_fine = lat_fine + 1
    time_coarse = clinspace("2020-01-01T12", "2020-01-02T12", 3)
    time_fine = clinspace("2020-01-01T09:36", "2020-01-02T15:35", 8)
    alt_coarse = lat_coarse + 3
    alt_fine = lat_fine + 3

    nn_request_fine_from_coarse = [0, 1, 2]
    nn_request_coarse_from_fine = [1, 3, 6]
    lin_request_fine_from_coarse = [0, 1, 2]
    lin_request_coarse_from_fine = [0, 1, 3, 4, 6, 7]
    # nn_request_fine_from_random_fine = [1, 1, 4, 6, 5, 0, 7, 2]
    nn_request_fine_from_random_fine = [0, 1, 2, 4, 5, 6, 7]
    nn_request_coarse_from_random_fine = [1, 5, 7]
    nn_request_fine_from_random_coarse = [0, 1, 2]
    nn_request_coarse_from_random_coarse = [0, 1, 2]
    nn_request_coarse_from_fine_grid = [1, 2, 3, 5, 6]

    coords: Dict[str, Coordinates]

    @classmethod
    def setup_class(cls):
        cls.make_coord_combos()

    @classmethod
    def make_coord_combos(cls):
        cls.coords = {}
        dims = ["lat", "lon", "time", "alt"]
        resolutions = ["fine", "coarse"]
        # Generate all possible orders of dimensions, from length 1 to 4
        dim_sequences: List[Tuple] = []
        for i in range(0, len(dims)):
            possible_combos = itertools.permutations(dims, r=i + 1)
            # Exclude sequences of dimensions that begin with time.
            # The Coordinates() constructor fails if we start with that one.
            # When we fix that bug we can turn this back on.
            dim_sequences += [combo for combo in possible_combos if combo[0] != "time"]

        # Make Coordinates objects from those dimensions
        for r in resolutions:
            for dim_seq in dim_sequences:
                key = "_".join(dim_seq + (r,))
                if len(dim_seq) <= 1:
                    new_coords = Coordinates([getattr(cls, d + "_" + r) for d in dim_seq], list(dim_seq))
                else: 
                    new_coords = Coordinates([[getattr(cls, d + "_" + r) for d in dim_seq]], [list(dim_seq)])
                cls.coords[key] = new_coords

    def test_nn_nonmonotonic_selector(self):
        selector = Selector("nearest")
        for request in ["lat_coarse", "lat_fine"]:
            for source in ["lat_random_fine", "lat_random_coarse"]:
                if "fine" in request and "fine" in source:
                    truth = set(self.nn_request_fine_from_random_fine)
                if "coarse" in request and "coarse" in source:
                    truth = set(self.nn_request_coarse_from_random_coarse)
                if "coarse" in request and "fine" in source:
                    truth = set(self.nn_request_coarse_from_random_fine)
                if "fine" in request and "coarse" in source:
                    truth = set(self.nn_request_fine_from_random_coarse)

                src_coords = Coordinates([getattr(self, source)], ["lat"])
                req_coords = Coordinates([getattr(self, request)], ["lat"])
                c, ci = selector.select(src_coords, req_coords)
                np.testing.assert_array_equal(
                    ci,
                    (np.array(list(truth)),),
                    err_msg=_ERR_MSG.format(
                        source, request, ci, list(truth)
                    ),
                )

    def test_linear_selector(self):
        selector = Selector("linear")
        for request in self.coords:
            for source in self.coords:
                dims = [d for d in self.coords[source].udims if d in self.coords[request].udims]
                if len(dims) == 0:
                    continue  # Invalid combination
                if "fine" in request and "fine" in source:
                    continue
                if "coarse" in request and "coarse" in source:
                    continue
                if "coarse" in request and "fine" in source:
                    truth = self.lin_request_coarse_from_fine
                if "fine" in request and "coarse" in source:
                    truth = self.lin_request_fine_from_coarse

                c, ci = selector.select(self.coords[source], self.coords[request])
                np.testing.assert_array_equal(
                    ci,
                    (np.array(truth),),
                    err_msg=_ERR_MSG.format(
                        source, request, ci, truth
                    ),
                )

    def test_bilinear_selector(self):
        selector = Selector("bilinear")
        for request in self.coords:
            for source in self.coords:
                dims = [d for d in self.coords[source].udims if d in self.coords[request].udims]
                if len(dims) == 0:
                    continue  # Invalid combination
                if "fine" in request and "fine" in source:
                    continue
                if "coarse" in request and "coarse" in source:
                    continue
                if "coarse" in request and "fine" in source:
                    truth = self.lin_request_coarse_from_fine
                if "fine" in request and "coarse" in source:
                    truth = self.lin_request_fine_from_coarse

                c, ci = selector.select(self.coords[source], self.coords[request])
                np.testing.assert_array_equal(
                    ci,
                    (np.array(truth),),
                    err_msg=_ERR_MSG.format(
                        source, request, ci, truth
                    ),
                )

    def test_bilinear_selector_negative_step(self):
        selector = Selector("bilinear")
        request1 = Coordinates([clinspace(-0.5, -1, 11)], ["lat"])
        request2 = Coordinates([clinspace(-1, -0.5, 11)], ["lat"])
        source1 = Coordinates([clinspace(-2, 0, 100)], ["lat"])
        source2 = Coordinates([clinspace(0, -2, 100)], ["lat"])
        c11, ci11 = selector.select(source1, request1)
        assert len(c11["lat"]) == 22
        assert len(ci11[0]) == 22

        c12, ci12 = selector.select(source1, request2)
        assert len(c12["lat"]) == 22
        assert len(ci12[0]) == 22

        c21, ci21 = selector.select(source2, request1)
        assert len(c21["lat"]) == 22
        assert len(ci21[0]) == 22

        c22, ci22 = selector.select(source2, request2)
        assert len(c22["lat"]) == 22
        assert len(ci22[0]) == 22

        np.testing.assert_equal(ci11[0], ci12[0])
        np.testing.assert_equal(ci21[0], ci22[0])

    def test_nearest_selector_negative_step(self):
        selector = Selector("nearest")
        request1 = Coordinates([clinspace(-0.5, -1, 11)], ["lat"])
        request2 = Coordinates([clinspace(-1, -0.5, 11)], ["lat"])
        source1 = Coordinates([clinspace(-2, 0, 100)], ["lat"])
        source2 = Coordinates([clinspace(0, -2, 100)], ["lat"])
        c11, ci11 = selector.select(source1, request1)
        assert len(c11["lat"]) == 11
        assert len(ci11[0]) == 11

        c12, ci12 = selector.select(source1, request2)
        assert len(c12["lat"]) == 11
        assert len(ci12[0]) == 11

        c21, ci21 = selector.select(source2, request1)
        assert len(c21["lat"]) == 11
        assert len(ci21[0]) == 11

        c22, ci22 = selector.select(source2, request2)
        assert len(c22["lat"]) == 11
        assert len(ci22[0]) == 11

        np.testing.assert_equal(ci11[0], ci12[0])
        np.testing.assert_equal(ci21[0], ci22[0])

    def test_nearest_selector_negative_time_step(self):
        selector = Selector("nearest")
        request1 = Coordinates([clinspace("2020-01-01", "2020-01-11", 11)], ["time"])
        request2 = Coordinates([clinspace("2020-01-11", "2020-01-01", 11)], ["time"])
        source1 = Coordinates([clinspace("2020-01-22T00", "2020-01-01T00", 126)], ["time"])
        source2 = Coordinates([clinspace("2020-01-01T00", "2020-01-22T00", 126)], ["time"])
        c11, ci11 = selector.select(source1, request1)
        assert len(c11["time"]) == 11
        assert len(ci11[0]) == 11

        c12, ci12 = selector.select(source1, request2)
        assert len(c12["time"]) == 11
        assert len(ci12[0]) == 11

        c21, ci21 = selector.select(source2, request1)
        assert len(c21["time"]) == 11
        assert len(ci21[0]) == 11

        c22, ci22 = selector.select(source2, request2)
        assert len(c22["time"]) == 11
        assert len(ci22[0]) == 11

        np.testing.assert_equal(ci11[0], ci12[0])
        np.testing.assert_equal(ci21[0], ci22[0])

    def test_nn_selector(self):
        selector = Selector("nearest")
        for request in self.coords:
            for source in self.coords:
                dims = [d for d in self.coords[source].udims if d in self.coords[request].udims]
                if len(dims) == 0:
                    continue  # Invalid combination
                if "fine" in request and "fine" in source:
                    continue
                if "coarse" in request and "coarse" in source:
                    continue
                if "coarse" in request and "fine" in source:
                    truth = self.nn_request_coarse_from_fine
                if "fine" in request and "coarse" in source:
                    truth = self.nn_request_fine_from_coarse

                c, ci = selector.select(self.coords[source], self.coords[request])
                np.testing.assert_array_equal(
                    ci,
                    (np.array(truth),),
                    err_msg=_ERR_MSG.format(
                        source, request, ci, truth
                    ),
                )

    def test_uniform2uniform(self):
        fine = Coordinates([self.lat_fine, self.lon_fine], ["lat", "lon"])
        coarse = Coordinates([self.lat_coarse, self.lon_coarse], ["lat", "lon"])

        selector = Selector("nearest")

        c, ci = selector.select(fine, coarse)
        for cci, trth in zip(ci, np.ix_(self.nn_request_coarse_from_fine, self.nn_request_coarse_from_fine)):
            np.testing.assert_array_equal(cci, trth)

        c, ci = selector.select(coarse, fine)
        for cci, trth in zip(ci, np.ix_(self.nn_request_fine_from_coarse, self.nn_request_fine_from_coarse)):
            np.testing.assert_array_equal(cci, trth)

    def test_point2uniform(self):
        u_fine = Coordinates([self.lat_fine, self.lon_fine], ["lat", "lon"])
        u_coarse = Coordinates([self.lat_coarse, self.lon_coarse], ["lat", "lon"])

        p_fine = Coordinates([[self.lat_fine, self.lon_fine]], [["lat", "lon"]])
        p_coarse = Coordinates([[self.lat_coarse, self.lon_coarse]], [["lat", "lon"]])

        selector = Selector("nearest")

        c, ci = selector.select(u_fine, p_coarse)
        for cci, trth in zip(ci, np.ix_(self.nn_request_coarse_from_fine, self.nn_request_coarse_from_fine)):
            np.testing.assert_array_equal(cci, trth)

        c, ci = selector.select(u_coarse, p_fine)
        for cci, trth in zip(ci, np.ix_(self.nn_request_fine_from_coarse, self.nn_request_fine_from_coarse)):
            np.testing.assert_array_equal(cci, trth)

        c, ci = selector.select(p_fine, u_coarse)
        np.testing.assert_array_equal(ci, (self.nn_request_coarse_from_fine_grid,))

        c, ci = selector.select(p_coarse, u_fine)
        np.testing.assert_array_equal(ci, (self.nn_request_fine_from_coarse,))

        # Respect bounds
        selector.respect_bounds = True
        c, ci = selector.select(u_fine, p_coarse)
        for cci, trth in zip(ci, np.ix_(self.nn_request_coarse_from_fine, self.nn_request_coarse_from_fine)):
            np.testing.assert_array_equal(cci, trth)

    def test_point2uniform_non_square(self):
        u_fine = Coordinates([self.lat_fine, self.lon_fine[:-1]], ["lat", "lon"])
        u_coarse = Coordinates([self.lat_coarse[:-1], self.lon_coarse], ["lat", "lon"])

        p_fine = Coordinates([[self.lat_fine, self.lon_fine]], [["lat", "lon"]])
        p_coarse = Coordinates([[self.lat_coarse, self.lon_coarse]], [["lat", "lon"]])

        selector = Selector("nearest")

        c, ci = selector.select(u_fine, p_coarse)
        for cci, trth in zip(ci, np.ix_(self.nn_request_coarse_from_fine, self.nn_request_coarse_from_fine)):
            np.testing.assert_array_equal(cci, trth)

        c, ci = selector.select(u_coarse, p_fine)
        for cci, trth in zip(ci, np.ix_(self.nn_request_fine_from_coarse[:-1], self.nn_request_fine_from_coarse)):
            np.testing.assert_array_equal(cci, trth)

        c, ci = selector.select(p_fine, u_coarse)
        np.testing.assert_array_equal(ci, (self.nn_request_coarse_from_fine_grid[:-1],))

        c, ci = selector.select(p_coarse, u_fine)
        np.testing.assert_array_equal(ci, (self.nn_request_fine_from_coarse,))

        # Respect bounds
        selector.respect_bounds = True
        c, ci = selector.select(u_fine, p_coarse)
        for cci, trth in zip(ci, np.ix_(self.nn_request_coarse_from_fine, self.nn_request_coarse_from_fine)):
            np.testing.assert_array_equal(cci, trth)

    def test_point2uniform_non_square_xarray_type(self):
        u_fine = Coordinates([self.lat_fine, self.lon_fine[:-1]], ["lat", "lon"])
        u_coarse = Coordinates([self.lat_coarse[:-1], self.lon_coarse], ["lat", "lon"])

        p_fine = Coordinates([[self.lat_fine, self.lon_fine]], [["lat", "lon"]])
        p_coarse = Coordinates([[self.lat_coarse, self.lon_coarse]], [["lat", "lon"]])

        selector = Selector("nearest")
        # Test xarray indices instead
        cx, cix = selector.select(u_fine, p_coarse, index_type="xarray")
        cn, cin = selector.select(u_fine, p_coarse, index_type="numpy")
        xarr = Node().create_output_array(u_fine)
        xarr[...] = np.random.rand(*xarr.shape)

        np.testing.assert_equal(xarr[cix], xarr.data[cin])

    def test_slice_index(self):
        selector = Selector("nearest")

        src = Coordinates([[0, 1, 2, 3, 4, 5]], dims=["lat"])

        # uniform
        req = Coordinates([[2, 4]], dims=["lat"])
        c, ci = selector.select(src, req, index_type="slice")
        assert isinstance(ci[0], slice)
        assert c == src[ci]

        # non uniform
        req = Coordinates([[1, 2, 4]], dims=["lat"])
        c, ci = selector.select(src, req, index_type="slice")
        assert isinstance(ci[0], slice)
        assert c == src[ci]

        # empty
        req = Coordinates([[10]], dims=["lat"])
        c, ci = selector.select(src, req, index_type="slice")
        assert isinstance(ci[0], slice)
        assert c == src[ci]

        # singleton
        req = Coordinates([[2]], dims=["lat"])
        c, ci = selector.select(src, req, index_type="slice")
        assert isinstance(ci[0], slice)
        assert c == src[ci]

    def test_higher_precision_time_coords1d(self):
        """
        Test _higher_precision_time_coords1d with datetime, timedelta, and non-time coordinates
        with different precisions to ensure the output is a float64
        """
        test_cases = [
            # Same datetime64 precision
            (Coordinates([np.datetime64("2025-01-01","D")],['time']), 
             Coordinates([np.datetime64("2025-01-03","D")],['time']),
             "datetime64[D]"),
            # Same timedelta64 precision
            (Coordinates([np.timedelta64(1, "D")],['time']), 
             Coordinates([np.timedelta64(3, "D")],['time']),
             "timedelta64[D]"),
            # Different datetime64 precision
            (Coordinates([np.datetime64("2025-01-01T12:00","s")],['time']), 
             Coordinates([np.datetime64("2025-01-03","D")],['time']),
             "datetime64[s]"),  # Should upcast to higher precision
            # Different timedelta64 precision
            (Coordinates([np.timedelta64(1, "s")],['time']), 
             Coordinates([np.timedelta64(3, "D")],['time']),
             "timedelta64[s]"),  # Should upcast to hours
            # Non-time data gets converted to float64 by Coordinates
            # Same Non-time data type 
            (Coordinates([np.array([1, 2, 3], dtype=np.float32)],['lat']), 
             Coordinates([np.array([4, 5, 6], dtype=np.float32)],['lat']), "float64"),
            # Different Non-time data type 
            (Coordinates([np.array([1, 2, 3], dtype=np.int16)],['lat']), 
             Coordinates([np.array([4, 5, 6], dtype=np.float32)],['lat']), "float64"),
            # Different Non-time data type ordered backwards
            (Coordinates([np.array([1, 2, 3], dtype=np.float32)],['lat']), 
             Coordinates([np.array([4, 5, 6], dtype=np.int16)],['lat']), "float64")
        ] 
        for coords0, coords1, expected_dtype in test_cases:
            dim = coords0.dims
            result0, result1 = _higher_precision_time_coords1d(coords0[dim[0]], coords1[dim[0]])
            # Ensure values are converted to float
            if np.issubdtype(expected_dtype, np.datetime64) or np.issubdtype(expected_dtype, np.timedelta64):
                assert result0.dtype == np.float64
                assert result1.dtype == np.float64
            # Ensure non-time data is unchanged
            if np.issubdtype(expected_dtype, np.float64):
                assert result0.dtype == np.float64
                assert result1.dtype == np.float64
