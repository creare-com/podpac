import pytest
import numpy as np

import podpac
from podpac.core.data.datasource import DataSource
from podpac.core.data.array_source import Array
from podpac.core.compositor.compositor import BaseCompositor

COORDS = podpac.Coordinates(
    [podpac.clinspace(45, 0, 16), podpac.clinspace(-70, -65, 16), podpac.clinspace(0, 1, 2)],
    dims=["lat", "lon", "time"],
)
LON, LAT, TIME = np.meshgrid(COORDS["lon"].coordinates, COORDS["lat"].coordinates, COORDS["time"].coordinates)

ARRAY_LAT = Array(source=LAT.astype(float), coordinates=COORDS, interpolation="bilinear")
ARRAY_LON = Array(source=LON.astype(float), coordinates=COORDS, interpolation="bilinear")
ARRAY_TIME = Array(source=TIME.astype(float), coordinates=COORDS, interpolation="bilinear")

MULTI_0_XY = Array(source=np.full(COORDS.shape + (2,), 0), coordinates=COORDS, outputs=["x", "y"])
MULTI_1_XY = Array(source=np.full(COORDS.shape + (2,), 1), coordinates=COORDS, outputs=["x", "y"])
MULTI_4_YX = Array(source=np.full(COORDS.shape + (2,), 4), coordinates=COORDS, outputs=["y", "x"])
MULTI_2_X = Array(source=np.full(COORDS.shape + (1,), 2), coordinates=COORDS, outputs=["x"])
MULTI_3_Z = Array(source=np.full(COORDS.shape + (1,), 3), coordinates=COORDS, outputs=["z"])


class MockComposite(BaseCompositor):
    def composite(self, coordinates, outputs, result=None):
        if result is None:
            result = self.create_output_array(coordinates)
        output = next(outputs)
        try:
            result[:] = output.transpose(*result.dims)
        except ValueError:
            raise podpac.NodeException("Cannot evaluate compositor with requested dims")
        return result


class TestBaseCompositor(object):
    def test_init(self):
        node = BaseCompositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        repr(node)

    def test_source_coordinates(self):
        # none (default)
        node = BaseCompositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        assert node.source_coordinates is None

        # unstacked
        node = BaseCompositor(
            sources=[podpac.algorithms.Arange(), podpac.algorithms.SinCoords()],
            source_coordinates=podpac.Coordinates([[0, 1]], dims=["time"]),
        )

        # stacked
        node = BaseCompositor(
            sources=[podpac.algorithms.Arange(), podpac.algorithms.SinCoords()],
            source_coordinates=podpac.Coordinates([[[0, 1], [10, 20]]], dims=["time_alt"]),
        )

        # invalid size
        with pytest.raises(ValueError, match="Invalid source_coordinates, source and source_coordinates size mismatch"):
            node = BaseCompositor(
                sources=[podpac.algorithms.Arange(), podpac.algorithms.SinCoords()],
                source_coordinates=podpac.Coordinates([[0, 1, 2]], dims=["time"]),
            )

        with pytest.raises(ValueError, match="Invalid source_coordinates, source and source_coordinates size mismatch"):
            node = BaseCompositor(
                sources=[podpac.algorithms.Arange(), podpac.algorithms.SinCoords()],
                source_coordinates=podpac.Coordinates([[0, 1, 2]], dims=["time"]),
            )

        # invalid ndims
        with pytest.raises(ValueError, match="Invalid source_coordinates"):
            node = BaseCompositor(
                sources=[podpac.algorithms.Arange(), podpac.algorithms.SinCoords()],
                source_coordinates=podpac.Coordinates([[0, 1], [10, 20]], dims=["time", "alt"]),
            )

    def test_select_sources_default(self):
        node = BaseCompositor(sources=[DataSource(), DataSource(), podpac.algorithms.Arange()])
        sources = node.select_sources(podpac.Coordinates([[0, 10]], ["time"]))

        assert isinstance(sources, list)
        assert len(sources) == 3

    def test_select_sources_intersection(self):
        source_coords = podpac.Coordinates([[0, 10]], ["time"])
        node = BaseCompositor(sources=[DataSource(), DataSource()], source_coordinates=source_coords)

        # select all
        selected = node.select_sources(source_coords)
        assert len(selected) == 2
        assert selected[0] == node.sources[0]
        assert selected[1] == node.sources[1]

        # select first
        c = podpac.Coordinates([podpac.clinspace(0, 1, 10), podpac.clinspace(0, 1, 11), 0], ["lat", "lon", "time"])
        selected = node.select_sources(c)
        assert len(selected) == 1
        assert selected[0] == node.sources[0]

        # select second
        c = podpac.Coordinates([podpac.clinspace(0, 1, 10), podpac.clinspace(0, 1, 11), 10], ["lat", "lon", "time"])
        selected = node.select_sources(c)
        assert len(selected) == 1
        assert selected[0] == node.sources[1]

        # select none
        c = podpac.Coordinates([podpac.clinspace(0, 1, 10), podpac.clinspace(0, 1, 11), 100], ["lat", "lon", "time"])
        selected = node.select_sources(c)
        assert len(selected) == 0

    def test_iteroutputs_empty(self):
        node = BaseCompositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        outputs = node.iteroutputs(podpac.Coordinates([-1, -1, -1], dims=["lat", "lon", "time"]))
        np.testing.assert_array_equal(next(outputs), [[[np.nan]]])
        np.testing.assert_array_equal(next(outputs), [[[np.nan]]])
        np.testing.assert_array_equal(next(outputs), [[[np.nan]]])
        with pytest.raises(StopIteration):
            next(outputs)

    def test_iteroutputs_singlethreaded(self):
        with podpac.settings:
            podpac.settings["MULTITHREADING"] = False

            node = BaseCompositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
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
            node = BaseCompositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
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
            node = BaseCompositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
            outputs = node.iteroutputs(COORDS)
            np.testing.assert_array_equal(next(outputs), LAT)
            np.testing.assert_array_equal(next(outputs), LON)
            np.testing.assert_array_equal(next(outputs), TIME)
            with pytest.raises(StopIteration):
                next(outputs)
            assert node._multi_threaded == False
            assert podpac.core.managers.multi_threading.thread_manager._n_threads_used == n_threads_before

    def test_composite(self):
        node = BaseCompositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        with pytest.raises(NotImplementedError):
            node.composite(COORDS, iter(()))

    def test_eval(self):
        node = BaseCompositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        with pytest.raises(NotImplementedError):
            node.eval(COORDS)

        node = MockComposite(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        output = node.eval(COORDS)
        np.testing.assert_array_equal(output, LAT)

    def test_eval_extra_dims(self):
        coords = COORDS.drop("time")
        a = Array(source=np.ones(coords.shape), coordinates=coords)
        b = Array(source=np.zeros(coords.shape), coordinates=coords)

        # no dims provided, evaluation fails with extra requested dims
        node = MockComposite(sources=[a, b])
        np.testing.assert_array_equal(node.eval(coords), a.source)
        with pytest.raises(podpac.NodeException, match="Cannot evaluate compositor with requested dims"):
            node.eval(COORDS)

        # dims provided, evaluation should succeed with extra requested dims
        node = MockComposite(sources=[a, b], dims=["lat", "lon"])
        np.testing.assert_array_equal(node.eval(coords), a.source)
        np.testing.assert_array_equal(node.eval(COORDS), a.source)

        # drop stacked dimensions if none of its dimensions are needed
        c = podpac.Coordinates(
            [COORDS["lat"], COORDS["lon"], [COORDS["time"], [10, 20]]], dims=["lat", "lon", "time_alt"]
        )
        np.testing.assert_array_equal(node.eval(c), a.source)

        # TODO
        # but don't drop stacked dimensions if any of its dimensions are needed
        # c = podpac.Coordinates([[COORDS['lat'], COORDS['lon'], np.arange(COORDS['lat'].size)]], dims=['lat_lon_alt'])
        # np.testing.assert_array_equal(node.eval(c), np.ones(COORDS['lat'].size))

        # dims can also be specified by the node
        class MockComposite2(MockComposite):
            dims = ["lat", "lon"]

        node = MockComposite2(sources=[a, b])
        np.testing.assert_array_equal(node.eval(coords), a.source)
        np.testing.assert_array_equal(node.eval(COORDS), a.source)

    def test_find_coordinates(self):
        node = BaseCompositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])

        coord_list = node.find_coordinates()
        assert isinstance(coord_list, list)
        assert len(coord_list) == 3

    def test_outputs(self):
        # standard single-output
        node = BaseCompositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME])
        assert node.outputs is None

        # even if the sources have multiple outputs, the default here is outputs
        node = BaseCompositor(sources=[MULTI_0_XY, MULTI_1_XY])
        assert node.outputs is None

    def test_auto_outputs(self):
        # autodetect single-output
        node = BaseCompositor(sources=[ARRAY_LAT, ARRAY_LON, ARRAY_TIME], auto_outputs=True)
        assert node.outputs is None

        # autodetect multi-output
        node = BaseCompositor(sources=[MULTI_0_XY, MULTI_1_XY], auto_outputs=True)
        assert node.outputs == ["x", "y"]

        node = BaseCompositor(sources=[MULTI_0_XY, MULTI_3_Z], auto_outputs=True)
        assert node.outputs == ["x", "y", "z"]

        node = BaseCompositor(sources=[MULTI_3_Z, MULTI_0_XY], auto_outputs=True)
        assert node.outputs == ["z", "x", "y"]

        node = BaseCompositor(sources=[MULTI_0_XY, MULTI_4_YX], auto_outputs=True)
        assert node.outputs == ["x", "y"]

        # mixed
        with pytest.raises(ValueError, match="Cannot composite standard sources with multi-output sources."):
            node = BaseCompositor(sources=[MULTI_2_X, ARRAY_LAT], auto_outputs=True)

        # no sources
        node = BaseCompositor(sources=[], auto_outputs=True)
        assert node.outputs is None

    def test_forced_invalid_sources(self):
        class MyCompositor(BaseCompositor):
            sources = [MULTI_2_X, ARRAY_LAT]
            auto_outputs = True

        node = MyCompositor()
        with pytest.raises(RuntimeError, match="Compositor sources were not validated correctly"):
            node.outputs
