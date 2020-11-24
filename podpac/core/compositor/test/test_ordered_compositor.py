import numpy as np

import pytest

import podpac
from podpac.core.data.array_source import Array
from podpac.core.compositor.ordered_compositor import OrderedCompositor

COORDS = podpac.Coordinates(
    [podpac.clinspace(45, 0, 16), podpac.clinspace(-70, -65, 16), podpac.clinspace(0, 1, 2)],
    dims=["lat", "lon", "time"],
)

MULTI_0_XY = Array(source=np.full(COORDS.shape + (2,), 0), coordinates=COORDS, outputs=["x", "y"])
MULTI_1_XY = Array(source=np.full(COORDS.shape + (2,), 1), coordinates=COORDS, outputs=["x", "y"])
MULTI_4_YX = Array(source=np.full(COORDS.shape + (2,), 4), coordinates=COORDS, outputs=["y", "x"])
MULTI_2_X = Array(source=np.full(COORDS.shape + (1,), 2), coordinates=COORDS, outputs=["x"])
MULTI_3_Z = Array(source=np.full(COORDS.shape + (1,), 3), coordinates=COORDS, outputs=["z"])


class TestOrderedCompositor(object):
    def test_composite(self):
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = False

            acoords = podpac.Coordinates([[-1, 0, 1], [10, 20, 30]], dims=["lat", "lon"])
            asource = np.ones(acoords.shape)
            asource[0, :] = np.nan
            a = Array(source=asource, coordinates=acoords)

            bcoords = podpac.Coordinates([[0, 1, 2, 3], [10, 20, 30, 40]], dims=["lat", "lon"])
            bsource = np.zeros(bcoords.shape)
            bsource[:, 0] = np.nan
            b = Array(source=bsource, coordinates=bcoords)

            coords = podpac.Coordinates([[0, 1, 2], [10, 20, 30, 40, 50]], dims=["lat", "lon"])

            node = OrderedCompositor(sources=[a, b], interpolation="bilinear")
            expected = np.array(
                [[1.0, 1.0, 1.0, 0.0, np.nan], [1.0, 1.0, 1.0, 0.0, np.nan], [np.nan, np.nan, 0.0, 0.0, np.nan]]
            )
            np.testing.assert_allclose(node.eval(coords), expected, equal_nan=True)

            node = OrderedCompositor(sources=[b, a], interpolation="bilinear")
            expected = np.array(
                [[1.0, 1.0, 0.0, 0.0, np.nan], [1.0, 1.0, 0.0, 0.0, np.nan], [np.nan, np.nan, 0.0, 0.0, np.nan]]
            )
            np.testing.assert_allclose(node.eval(coords), expected, equal_nan=True)

    def test_composite_multithreaded(self):
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 8

            acoords = podpac.Coordinates([[-1, 0, 1], [10, 20, 30]], dims=["lat", "lon"])
            asource = np.ones(acoords.shape)
            asource[0, :] = np.nan
            a = Array(source=asource, coordinates=acoords)

            bcoords = podpac.Coordinates([[0, 1, 2, 3], [10, 20, 30, 40]], dims=["lat", "lon"])
            bsource = np.zeros(bcoords.shape)
            bsource[:, 0] = np.nan
            b = Array(source=bsource, coordinates=bcoords)

            coords = podpac.Coordinates([[0, 1, 2], [10, 20, 30, 40, 50]], dims=["lat", "lon"])

            node = OrderedCompositor(sources=[a, b], interpolation="bilinear")
            expected = np.array(
                [[1.0, 1.0, 1.0, 0.0, np.nan], [1.0, 1.0, 1.0, 0.0, np.nan], [np.nan, np.nan, 0.0, 0.0, np.nan]]
            )
            np.testing.assert_allclose(node.eval(coords), expected, equal_nan=True)

            node = OrderedCompositor(sources=[b, a], interpolation="bilinear")
            expected = np.array(
                [[1.0, 1.0, 0.0, 0.0, np.nan], [1.0, 1.0, 0.0, 0.0, np.nan], [np.nan, np.nan, 0.0, 0.0, np.nan]]
            )
            np.testing.assert_allclose(node.eval(coords), expected, equal_nan=True)

    def test_composite_short_circuit(self):
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = False
            podpac.settings["DEBUG"] = True

            coords = podpac.Coordinates([[0, 1], [10, 20, 30]], dims=["lat", "lon"])
            a = Array(source=np.ones(coords.shape), coordinates=coords)
            b = Array(source=np.zeros(coords.shape), coordinates=coords)
            node = OrderedCompositor(sources=[a, b], interpolation="bilinear")
            output = node.eval(coords)
            np.testing.assert_array_equal(output, a.source)
            assert node._eval_sources[0]._output is not None
            assert node._eval_sources[1]._output is None

    def test_composite_short_circuit_multithreaded(self):
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 8
            podpac.settings["DEBUG"] = True

            coords = podpac.Coordinates([[0, 1], [10, 20, 30]], dims=["lat", "lon"])
            n_threads_before = podpac.core.managers.multi_threading.thread_manager._n_threads_used
            a = Array(source=np.ones(coords.shape), coordinates=coords)
            b = Array(source=np.zeros(coords.shape), coordinates=coords)
            node = OrderedCompositor(sources=[a, b], interpolation="bilinear")
            output = node.eval(coords)
            np.testing.assert_array_equal(output, a.source)
            assert node._multi_threaded == True
            assert podpac.core.managers.multi_threading.thread_manager._n_threads_used == n_threads_before

    def test_composite_into_result(self):
        coords = podpac.Coordinates([[0, 1], [10, 20, 30]], dims=["lat", "lon"])
        a = Array(source=np.ones(coords.shape), coordinates=coords)
        b = Array(source=np.zeros(coords.shape), coordinates=coords)
        node = OrderedCompositor(sources=[a, b], interpolation="bilinear")
        result = node.create_output_array(coords, data=np.random.random(coords.shape))
        output = node.eval(coords, output=result)
        np.testing.assert_array_equal(output, a.source)
        np.testing.assert_array_equal(result, a.source)

    def test_composite_multiple_outputs(self):
        node = OrderedCompositor(sources=[MULTI_0_XY, MULTI_1_XY], auto_outputs=True)
        output = node.eval(COORDS)
        assert output.dims == ("lat", "lon", "time", "output")
        np.testing.assert_array_equal(output["output"], ["x", "y"])
        np.testing.assert_array_equal(output.sel(output="x"), np.full(COORDS.shape, 0))
        np.testing.assert_array_equal(output.sel(output="y"), np.full(COORDS.shape, 0))

        node = OrderedCompositor(sources=[MULTI_1_XY, MULTI_0_XY], auto_outputs=True)
        output = node.eval(COORDS)
        assert output.dims == ("lat", "lon", "time", "output")
        np.testing.assert_array_equal(output["output"], ["x", "y"])
        np.testing.assert_array_equal(output.sel(output="x"), np.full(COORDS.shape, 1))
        np.testing.assert_array_equal(output.sel(output="y"), np.full(COORDS.shape, 1))

    def test_composite_combine_multiple_outputs(self):
        node = OrderedCompositor(sources=[MULTI_0_XY, MULTI_1_XY, MULTI_2_X, MULTI_3_Z], auto_outputs=True)
        output = node.eval(COORDS)
        assert output.dims == ("lat", "lon", "time", "output")
        np.testing.assert_array_equal(output["output"], ["x", "y", "z"])
        np.testing.assert_array_equal(output.sel(output="x"), np.full(COORDS.shape, 0))
        np.testing.assert_array_equal(output.sel(output="y"), np.full(COORDS.shape, 0))
        np.testing.assert_array_equal(output.sel(output="z"), np.full(COORDS.shape, 3))

        node = OrderedCompositor(sources=[MULTI_3_Z, MULTI_2_X, MULTI_0_XY, MULTI_1_XY], auto_outputs=True)
        output = node.eval(COORDS)
        assert output.dims == ("lat", "lon", "time", "output")
        np.testing.assert_array_equal(output["output"], ["z", "x", "y"])
        np.testing.assert_array_equal(output.sel(output="x"), np.full(COORDS.shape, 2))
        np.testing.assert_array_equal(output.sel(output="y"), np.full(COORDS.shape, 0))
        np.testing.assert_array_equal(output.sel(output="z"), np.full(COORDS.shape, 3))

        node = OrderedCompositor(sources=[MULTI_2_X, MULTI_4_YX], auto_outputs=True)
        output = node.eval(COORDS)
        assert output.dims == ("lat", "lon", "time", "output")
        np.testing.assert_array_equal(output["output"], ["x", "y"])
        np.testing.assert_array_equal(output.sel(output="x"), np.full(COORDS.shape, 2))
        np.testing.assert_array_equal(output.sel(output="y"), np.full(COORDS.shape, 4))

    def test_composite_stacked_unstacked(self):
        anative = podpac.Coordinates([podpac.clinspace((0, 1), (1, 2), size=3)], dims=["lat_lon"])
        bnative = podpac.Coordinates([podpac.clinspace(-2, 3, 3), podpac.clinspace(-1, 4, 3)], dims=["lat", "lon"])
        a = Array(source=np.random.rand(3), coordinates=anative)
        b = Array(source=np.random.rand(3, 3) + 2, coordinates=bnative)

        coords = podpac.Coordinates([podpac.clinspace(-3, 4, 32), podpac.clinspace(-2, 5, 32)], dims=["lat", "lon"])

        node = OrderedCompositor(sources=[a, b], interpolation="nearest")
        o = node.eval(coords)
        # Check that both data sources are being used in the interpolation
        assert np.any(o.data >= 2)
        assert np.any(o.data <= 1)

    def test_composite_extra_dims(self):
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = False

            coords = podpac.Coordinates([[0, 1], [10, 20, 30]], dims=["lat", "lon"])
            a = Array(source=np.ones(coords.shape), coordinates=coords)

            extra = podpac.Coordinates([coords["lat"], coords["lon"], "2020-01-01"], dims=["lat", "lon", "time"])

            # dims not provided, eval fails with extra dims
            node = OrderedCompositor(sources=[a], interpolation="bilinear")
            np.testing.assert_array_equal(node.eval(coords), a.source)
            with pytest.raises(podpac.NodeException, match="Cannot evaluate compositor with requested dims"):
                node.eval(extra)

            # dims provided, remove extra dims
            node = OrderedCompositor(sources=[a], dims=["lat", "lon"], interpolation="bilinear")
            np.testing.assert_array_equal(node.eval(coords), a.source)
            np.testing.assert_array_equal(node.eval(extra), a.source)
