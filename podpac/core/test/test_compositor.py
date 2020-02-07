import warnings

import pytest
import numpy as np

import podpac
from podpac.core.data.datasource import DataSource
from podpac.core.data.array_source import Array
from podpac.compositor import Compositor, OrderedCompositor

COORDS = podpac.Coordinates(
    [podpac.clinspace(45, 0, 16), podpac.clinspace(-70, -65, 16), podpac.clinspace(0, 1, 2)],
    dims=["lat", "lon", "time"],
)
LON, LAT, TIME = np.meshgrid(COORDS["lon"].coordinates, COORDS["lat"].coordinates, COORDS["time"].coordinates)

ARRAY_LAT = Array(source=LAT.astype(float), native_coordinates=COORDS, interpolation="bilinear")
ARRAY_LON = Array(source=LON.astype(float), native_coordinates=COORDS, interpolation="bilinear")
ARRAY_TIME = Array(source=TIME.astype(float), native_coordinates=COORDS, interpolation="bilinear")

MULTI_0_XY = Array(source=np.full(COORDS.shape + (2,), 0), native_coordinates=COORDS, outputs=["x", "y"])
MULTI_1_XY = Array(source=np.full(COORDS.shape + (2,), 1), native_coordinates=COORDS, outputs=["x", "y"])
MULTI_4_YX = Array(source=np.full(COORDS.shape + (2,), 4), native_coordinates=COORDS, outputs=["y", "x"])
MULTI_2_X = Array(source=np.full(COORDS.shape + (1,), 2), native_coordinates=COORDS, outputs=["x"])
MULTI_3_Z = Array(source=np.full(COORDS.shape + (1,), 3), native_coordinates=COORDS, outputs=["z"])


class TestCompositor(object):
    def test_init(self):
        node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        repr(node)

    def test_shared_coordinates(self):
        node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])

        with pytest.raises(NotImplementedError):
            node.get_shared_coordinates()

        with pytest.raises(NotImplementedError):
            node.shared_coordinates()

    def test_source_coordinates(self):
        # none (default)
        node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        assert node.source_coordinates is None
        assert node.get_source_coordinates() is None

        # unstacked
        node = podpac.compositor.Compositor(
            sources=[podpac.algorithm.Arange(), podpac.algorithm.SinCoords()],
            source_coordinates=podpac.Coordinates([[0, 1]], dims=["time"]),
        )

        # stacked
        node = podpac.compositor.Compositor(
            sources=[podpac.algorithm.Arange(), podpac.algorithm.SinCoords()],
            source_coordinates=podpac.Coordinates([[[0, 1], [10, 20]]], dims=["time_alt"]),
        )

        # invalid size
        with pytest.raises(ValueError, match="Invalid source_coordinates, source and source_coordinates size mismatch"):
            node = podpac.compositor.Compositor(
                sources=[podpac.algorithm.Arange(), podpac.algorithm.SinCoords()],
                source_coordinates=podpac.Coordinates([[0, 1, 2]], dims=["time"]),
            )

        with pytest.raises(ValueError, match="Invalid source_coordinates, source and source_coordinates size mismatch"):
            node = podpac.compositor.Compositor(
                sources=[podpac.algorithm.Arange(), podpac.algorithm.SinCoords()],
                source_coordinates=podpac.Coordinates([[0, 1, 2]], dims=["time"]),
            )

        # invalid ndims
        with pytest.raises(ValueError, match="Invalid source_coordinates"):
            node = podpac.compositor.Compositor(
                sources=[podpac.algorithm.Arange(), podpac.algorithm.SinCoords()],
                source_coordinates=podpac.Coordinates([[0, 1], [10, 20]], dims=["time", "alt"]),
            )

    def test_select_sources(self):
        source_coords = podpac.Coordinates([[0, 10]], ["time"])
        node = podpac.compositor.Compositor(
            sources=[podpac.algorithm.Arange(), podpac.algorithm.SinCoords()], source_coordinates=source_coords
        )

        selected = node.select_sources(source_coords)
        assert len(selected) == 2
        assert selected[0] is node.sources[0]
        assert selected[1] is node.sources[1]

        coords = podpac.Coordinates([podpac.clinspace(0, 1, 10), podpac.clinspace(0, 1, 11), 0], ["lat", "lon", "time"])
        selected = node.select_sources(coords)
        assert len(selected) == 1
        assert selected[0] is node.sources[0]

        coords = podpac.Coordinates(
            [podpac.clinspace(0, 1, 10), podpac.clinspace(0, 1, 11), 10], ["lat", "lon", "time"]
        )
        selected = node.select_sources(coords)
        assert len(selected) == 1
        assert selected[0] is node.sources[1]

        coords = podpac.Coordinates(
            [podpac.clinspace(0, 1, 10), podpac.clinspace(0, 1, 11), 100], ["lat", "lon", "time"]
        )
        selected = node.select_sources(coords)
        assert len(selected) == 0

    def test_iteroutputs_interpolation(self):
        node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME], interpolation="nearest")
        outputs = node.iteroutputs(COORDS)
        for output in outputs:
            pass
        assert node.sources[0].interpolation == "nearest"
        assert node.sources[1].interpolation == "nearest"
        assert node.sources[2].interpolation == "nearest"
        assert ARRAY_LAT.interpolation == "bilinear"
        assert ARRAY_LON.interpolation == "bilinear"
        assert ARRAY_TIME.interpolation == "bilinear"

        # if no interpolation is provided, keep the source interpolation values
        node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        outputs = node.iteroutputs(COORDS)
        for output in outputs:
            pass
        assert node.sources[0].interpolation == "bilinear"
        assert node.sources[1].interpolation == "bilinear"
        assert node.sources[2].interpolation == "bilinear"

    def test_iteroutputs_empty(self):
        node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        outputs = node.iteroutputs(podpac.Coordinates([-1, -1, -1], dims=["lat", "lon", "time"]))
        np.testing.assert_array_equal(next(outputs), [[[np.nan]]])
        np.testing.assert_array_equal(next(outputs), [[[np.nan]]])
        np.testing.assert_array_equal(next(outputs), [[[np.nan]]])
        with pytest.raises(StopIteration):
            next(outputs)

    def test_iteroutputs_singlethreaded(self):
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = False

            node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
            outputs = node.iteroutputs(COORDS)
            np.testing.assert_array_equal(next(outputs), LAT)
            np.testing.assert_array_equal(next(outputs), LON)
            np.testing.assert_array_equal(next(outputs), TIME)
            with pytest.raises(StopIteration):
                next(outputs)
            assert node._multi_threaded == False

    def test_iteroutputs_multithreaded(self):
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 8

            n_threads_before = podpac.core.managers.multi_threading.thread_manager._n_threads_used
            node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
            outputs = node.iteroutputs(COORDS)
            np.testing.assert_array_equal(next(outputs), LAT)
            np.testing.assert_array_equal(next(outputs), LON)
            np.testing.assert_array_equal(next(outputs), TIME)
            with pytest.raises(StopIteration):
                next(outputs)
            assert node._multi_threaded == True
            assert podpac.core.managers.multi_threading.thread_manager._n_threads_used == n_threads_before

    def test_iteroutputs_n_threads_1(self):
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 1

            n_threads_before = podpac.core.managers.multi_threading.thread_manager._n_threads_used
            node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
            outputs = node.iteroutputs(COORDS)
            np.testing.assert_array_equal(next(outputs), LAT)
            np.testing.assert_array_equal(next(outputs), LON)
            np.testing.assert_array_equal(next(outputs), TIME)
            with pytest.raises(StopIteration):
                next(outputs)
            assert node._multi_threaded == False
            assert podpac.core.managers.multi_threading.thread_manager._n_threads_used == n_threads_before

    def test_composite(self):
        node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        with pytest.raises(NotImplementedError):
            node.composite(COORDS, iter(()))

    def test_eval(self):
        node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        with pytest.raises(NotImplementedError):
            node.eval(COORDS)

        class MockComposite(Compositor):
            def composite(self, coordinates, outputs, result=None):
                return next(outputs)

        node = MockComposite(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        output = node.eval(COORDS)
        np.testing.assert_array_equal(output, LAT)

    def test_find_coordinates(self):
        node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])

        with pytest.raises(NotImplementedError):
            node.find_coordinates()

    def test_base_definition(self):
        node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        d = node.base_definition
        assert isinstance(d, dict)
        assert "sources" in d
        assert "interpolation" in d

    def test_outputs(self):
        # standard single-output
        node = Compositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        assert node.outputs is None

        # multi-output
        node = Compositor(sources=[MULTI_0_XY, MULTI_1_XY])
        assert node.outputs == ["x", "y"]

        node = Compositor(sources=[MULTI_0_XY, MULTI_3_Z])
        assert node.outputs == ["x", "y", "z"]

        node = Compositor(sources=[MULTI_3_Z, MULTI_0_XY])
        assert node.outputs == ["z", "x", "y"]

        node = Compositor(sources=[MULTI_0_XY, MULTI_4_YX])
        assert node.outputs == ["x", "y"]

        # multi-output, with strict source outputs checking
        node = Compositor(sources=[MULTI_0_XY, MULTI_1_XY], strict_source_outputs=True)
        assert node.outputs == ["x", "y"]

        with pytest.raises(ValueError, match="Source outputs mismatch"):
            node = Compositor(sources=[MULTI_0_XY, MULTI_2_X], strict_source_outputs=True)

        with pytest.raises(ValueError, match="Source outputs mismatch"):
            node = Compositor(sources=[MULTI_0_XY, MULTI_3_Z], strict_source_outputs=True)

        with pytest.raises(ValueError, match="Source outputs mismatch"):
            node = Compositor(sources=[MULTI_0_XY, MULTI_4_YX], strict_source_outputs=True)

        # mixed
        with pytest.raises(ValueError, match="Cannot composite standard sources with multi-output sources."):
            node = Compositor(sources=[MULTI_2_X, ARRAY_LAT])


class TestOrderedCompositor(object):
    def test_composite(self):
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = False

            acoords = podpac.Coordinates([[0, 1], [10, 20, 30]], dims=["lat", "lon"])
            asource = np.ones(acoords.shape)
            asource[0, :] = np.nan
            a = Array(source=asource, native_coordinates=acoords)

            bcoords = podpac.Coordinates([[0, 1, 2], [10, 20, 30, 40]], dims=["lat", "lon"])
            bsource = np.zeros(bcoords.shape)
            bsource[:, 0] = np.nan
            b = Array(source=bsource, native_coordinates=bcoords)

            coords = podpac.Coordinates([[0, 1, 2], [10, 20, 30, 40, 50]], dims=["lat", "lon"])

            node = OrderedCompositor(sources=[a, b], interpolation="bilinear")
            expected = np.array(
                [[np.nan, 0.0, 0.0, 0.0, np.nan], [1.0, 1.0, 1.0, 0.0, np.nan], [np.nan, 0.0, 0.0, 0.0, np.nan]]
            )
            np.testing.assert_allclose(node.eval(coords), expected, equal_nan=True)

            node = OrderedCompositor(sources=[b, a], interpolation="bilinear")
            expected = np.array(
                [[np.nan, 0.0, 0.0, 0.0, np.nan], [1.0, 0.0, 0.0, 0.0, np.nan], [np.nan, 0.0, 0.0, 0.0, np.nan]]
            )
            np.testing.assert_allclose(node.eval(coords), expected, equal_nan=True)

    def test_composite_multithreaded(self):
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 8

            acoords = podpac.Coordinates([[0, 1], [10, 20, 30]], dims=["lat", "lon"])
            asource = np.ones(acoords.shape)
            asource[0, :] = np.nan
            a = Array(source=asource, native_coordinates=acoords)

            bcoords = podpac.Coordinates([[0, 1, 2], [10, 20, 30, 40]], dims=["lat", "lon"])
            bsource = np.zeros(bcoords.shape)
            bsource[:, 0] = np.nan
            b = Array(source=bsource, native_coordinates=bcoords)

            coords = podpac.Coordinates([[0, 1, 2], [10, 20, 30, 40, 50]], dims=["lat", "lon"])

            node = OrderedCompositor(sources=[a, b], interpolation="bilinear")
            expected = np.array(
                [[np.nan, 0.0, 0.0, 0.0, np.nan], [1.0, 1.0, 1.0, 0.0, np.nan], [np.nan, 0.0, 0.0, 0.0, np.nan]]
            )
            np.testing.assert_allclose(node.eval(coords), expected, equal_nan=True)

            node = OrderedCompositor(sources=[b, a], interpolation="bilinear")
            expected = np.array(
                [[np.nan, 0.0, 0.0, 0.0, np.nan], [1.0, 0.0, 0.0, 0.0, np.nan], [np.nan, 0.0, 0.0, 0.0, np.nan]]
            )
            np.testing.assert_allclose(node.eval(coords), expected, equal_nan=True)

    def test_composite_short_circuit(self):
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = False
            podpac.settings["DEBUG"] = True

            coords = podpac.Coordinates([[0, 1], [10, 20, 30]], dims=["lat", "lon"])
            a = Array(source=np.ones(coords.shape), native_coordinates=coords)
            b = Array(source=np.zeros(coords.shape), native_coordinates=coords)
            node = OrderedCompositor(sources=[a, b], interpolation="bilinear")
            output = node.eval(coords)
            np.testing.assert_array_equal(output, a.source)
            assert node.sources[0]._output is not None
            assert node.sources[1]._output is None

    def test_composite_short_circuit_multithreaded(self):
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 8
            podpac.settings["DEBUG"] = True

            coords = podpac.Coordinates([[0, 1], [10, 20, 30]], dims=["lat", "lon"])
            n_threads_before = podpac.core.managers.multi_threading.thread_manager._n_threads_used
            a = Array(source=np.ones(coords.shape), native_coordinates=coords)
            b = Array(source=np.zeros(coords.shape), native_coordinates=coords)
            node = OrderedCompositor(sources=[a, b], interpolation="bilinear")
            output = node.eval(coords)
            np.testing.assert_array_equal(output, a.source)
            assert node._multi_threaded == True
            assert podpac.core.managers.multi_threading.thread_manager._n_threads_used == n_threads_before

    def test_composite_into_result(self):
        coords = podpac.Coordinates([[0, 1], [10, 20, 30]], dims=["lat", "lon"])
        a = Array(source=np.ones(coords.shape), native_coordinates=coords)
        b = Array(source=np.zeros(coords.shape), native_coordinates=coords)
        node = OrderedCompositor(sources=[a, b], interpolation="bilinear")
        result = node.create_output_array(coords, data=np.random.random(coords.shape))
        output = node.eval(coords, output=result)
        np.testing.assert_array_equal(output, a.source)
        np.testing.assert_array_equal(result, a.source)

    def test_composite_multiple_outputs(self):
        node = OrderedCompositor(sources=[MULTI_0_XY, MULTI_1_XY])
        output = node.eval(COORDS)
        assert output.dims == ("lat", "lon", "time", "output")
        np.testing.assert_array_equal(output["output"], ["x", "y"])
        np.testing.assert_array_equal(output.sel(output="x"), np.full(COORDS.shape, 0))
        np.testing.assert_array_equal(output.sel(output="y"), np.full(COORDS.shape, 0))

        node = OrderedCompositor(sources=[MULTI_1_XY, MULTI_0_XY], strict_source_outputs=True)
        output = node.eval(COORDS)
        assert output.dims == ("lat", "lon", "time", "output")
        np.testing.assert_array_equal(output["output"], ["x", "y"])
        np.testing.assert_array_equal(output.sel(output="x"), np.full(COORDS.shape, 1))
        np.testing.assert_array_equal(output.sel(output="y"), np.full(COORDS.shape, 1))

    def test_composite_combine_multiple_outputs(self):
        node = OrderedCompositor(sources=[MULTI_0_XY, MULTI_1_XY, MULTI_2_X, MULTI_3_Z])
        output = node.eval(COORDS)
        assert output.dims == ("lat", "lon", "time", "output")
        np.testing.assert_array_equal(output["output"], ["x", "y", "z"])
        np.testing.assert_array_equal(output.sel(output="x"), np.full(COORDS.shape, 0))
        np.testing.assert_array_equal(output.sel(output="y"), np.full(COORDS.shape, 0))
        np.testing.assert_array_equal(output.sel(output="z"), np.full(COORDS.shape, 3))

        node = OrderedCompositor(sources=[MULTI_3_Z, MULTI_2_X, MULTI_0_XY, MULTI_1_XY])
        output = node.eval(COORDS)
        assert output.dims == ("lat", "lon", "time", "output")
        np.testing.assert_array_equal(output["output"], ["z", "x", "y"])
        np.testing.assert_array_equal(output.sel(output="x"), np.full(COORDS.shape, 2))
        np.testing.assert_array_equal(output.sel(output="y"), np.full(COORDS.shape, 0))
        np.testing.assert_array_equal(output.sel(output="z"), np.full(COORDS.shape, 3))

        node = OrderedCompositor(sources=[MULTI_2_X, MULTI_4_YX])
        output = node.eval(COORDS)
        assert output.dims == ("lat", "lon", "time", "output")
        np.testing.assert_array_equal(output["output"], ["x", "y"])
        np.testing.assert_array_equal(output.sel(output="x"), np.full(COORDS.shape, 2))
        np.testing.assert_array_equal(output.sel(output="y"), np.full(COORDS.shape, 4))

    def test_composite_stacked_unstacked(self):
        anative = podpac.Coordinates([podpac.clinspace((0, 1), (1, 2), size=3)], dims=["lat_lon"])
        bnative = podpac.Coordinates([podpac.clinspace(-2, 3, 3), podpac.clinspace(-1, 4, 3)], dims=["lat", "lon"])
        a = Array(source=np.random.rand(3), native_coordinates=anative)
        b = Array(source=np.random.rand(3, 3) + 2, native_coordinates=bnative)

        coords = podpac.Coordinates([podpac.clinspace(-3, 4, 32), podpac.clinspace(-2, 5, 32)], dims=["lat", "lon"])

        node = OrderedCompositor(sources=np.array([a, b]), interpolation="nearest")
        o = node.eval(coords)
        # Check that both data sources are being used in the interpolation
        assert np.any(o.data >= 2)
        assert np.any(o.data <= 1)
