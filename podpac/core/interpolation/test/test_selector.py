import pytest
import traitlets as tl
import numpy as np

from podpac.core.node import Node
from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.interpolation.selector import Selector


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

    coords = {}

    @classmethod
    def setup_class(cls):
        cls.make_coord_combos(cls)

    @staticmethod
    def make_coord_combos(self):
        # Make 1-D ones
        dims = ["lat", "lon", "time", "alt"]
        for r in ["fine", "coarse"]:
            for i in range(4):
                d = dims[i]
                k = d + "_" + r
                self.coords[k] = Coordinates([getattr(self, k)], [d])
                # stack pairs 2D
                for ii in range(i, 4):
                    d2 = dims[ii]
                    if d == d2:
                        continue
                    k2 = "_".join([d2, r])
                    k2f = "_".join([d, d2, r])
                    self.coords[k2f] = Coordinates([[getattr(self, k), getattr(self, k2)]], [[d, d2]])
                    # stack pairs 3D
                    for iii in range(ii, 4):
                        d3 = dims[iii]
                        if d3 == d or d3 == d2:
                            continue
                        k3 = "_".join([d3, r])
                        k3f = "_".join([d, d2, d3, r])
                        self.coords[k3f] = Coordinates(
                            [[getattr(self, k), getattr(self, k2), getattr(self, k3)]], [[d, d2, d3]]
                        )
                        # stack pairs 4D
                        for iv in range(iii, 4):
                            d4 = dims[iv]
                            if d4 == d or d4 == d2 or d4 == d3:
                                continue
                            k4 = "_".join([d4, r])
                            k4f = "_".join([d, d2, d3, d4, r])
                            self.coords[k4f] = Coordinates(
                                [[getattr(self, k), getattr(self, k2), getattr(self, k3), getattr(self, k4)]],
                                [[d, d2, d3, d4]],
                            )

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
                    err_msg="Selection using source {} and request {} failed with {} != {} (truth)".format(
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
                    err_msg="Selection using source {} and request {} failed with {} != {} (truth)".format(
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
                    err_msg="Selection using source {} and request {} failed with {} != {} (truth)".format(
                        source, request, ci, truth
                    ),
                )

    def test_bilinear_selector_negative_step(self):
        selector = Selector("bilinear")
        request = Coordinates([clinspace(0,-2 ,10)], ['lat'])
        source = Coordinates([clinspace(-2,0,100)], ['lat'])
        c, ci = selector.select(source, request)
        assert len(c['lat']) == 15

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
                    err_msg="Selection using source {} and request {} failed with {} != {} (truth)".format(
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
