from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import traitlets as tl
import numpy as np
import xarray as xr

import podpac
from podpac.core.algorithm.algorithm import BaseAlgorithm, Algorithm, UnaryAlgorithm


class TestBaseAlgorithm(object):
    def test_eval_not_implemented(self):
        node = BaseAlgorithm()
        c = podpac.Coordinates([])
        with pytest.raises(NotImplementedError):
            node.eval(c)

    def test_inputs(self):
        class MyAlgorithm(BaseAlgorithm):
            x = tl.Instance(podpac.Node).tag(attr=True)
            y = tl.Instance(podpac.Node).tag(attr=True)

        node = MyAlgorithm(x=podpac.Node(), y=podpac.Node())
        assert "x" in node.inputs
        assert "y" in node.inputs

    def test_find_coordinates(self):
        class MyAlgorithm(BaseAlgorithm):
            x = tl.Instance(podpac.Node).tag(attr=True)
            y = tl.Instance(podpac.Node).tag(attr=True)

        node = MyAlgorithm(
            x=podpac.data.Array(coordinates=podpac.Coordinates([[0, 1, 2]], dims=["lat"])),
            y=podpac.data.Array(coordinates=podpac.Coordinates([[10, 11, 12], [100, 200]], dims=["lat", "lon"])),
        )

        l = node.find_coordinates()
        assert isinstance(l, list)
        assert len(l) == 2
        assert node.x.coordinates in l
        assert node.y.coordinates in l


class TestAlgorithm(object):
    def test_algorithm_not_implemented(self):
        node = Algorithm()
        c = podpac.Coordinates([])
        with pytest.raises(NotImplementedError):
            node.eval(c)

    def test_eval(self):
        class MyAlgorithm(Algorithm):
            def algorithm(self, inputs, coordinates):
                data = np.arange(coordinates.size).reshape(coordinates.shape)
                return self.create_output_array(coordinates, data=data)

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        node = MyAlgorithm()
        output = node.eval(coords)
        assert isinstance(output, podpac.UnitsDataArray)
        assert output.dims == ("lat", "lon")
        np.testing.assert_array_equal(output, np.arange(6).reshape(3, 2))

    def test_eval_algorithm_returns_xarray(self):
        class MyAlgorithm(Algorithm):
            def algorithm(self, inputs, coordinates):
                data = np.arange(coordinates.size).reshape(coordinates.shape)
                return xr.DataArray(data, coords=coordinates.xcoords, dims=coordinates.dims)

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        node = MyAlgorithm()
        output = node.eval(coords)
        assert isinstance(output, podpac.UnitsDataArray)
        assert output.dims == ("lat", "lon")
        np.testing.assert_array_equal(output, np.arange(6).reshape(3, 2))

    def test_eval_algorithm_returns_numpy_array(self):
        class MyAlgorithm(Algorithm):
            def algorithm(self, inputs, coordinates):
                data = np.arange(coordinates.size).reshape(coordinates.shape)
                return data

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        node = MyAlgorithm()
        with pytest.raises(podpac.NodeException, match="algorithm returned unsupported type"):
            output = node.eval(coords)

    def test_eval_algorithm_drops_dimension(self):
        class MyAlgorithm(Algorithm):
            drop = tl.Unicode()

            def algorithm(self, inputs, coordinates):
                c = coordinates.drop(self.drop)
                data = np.arange(c.size).reshape(c.shape)
                return self.create_output_array(c, data=data)

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        node = MyAlgorithm(drop="lat")
        output = node.eval(coords)
        assert isinstance(output, podpac.UnitsDataArray)
        assert output.dims == ("lon",)
        np.testing.assert_array_equal(output, np.arange(2))

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        node = MyAlgorithm(drop="lon")
        output = node.eval(coords)
        assert isinstance(output, podpac.UnitsDataArray)
        assert output.dims == ("lat",)
        np.testing.assert_array_equal(output, np.arange(3))

    def test_eval_algorithm_adds_dimension(self):
        class MyAlgorithm(Algorithm):
            def algorithm(self, inputs, coordinates):
                c = podpac.Coordinates(
                    [["2020-01-01", "2020-01-02"], coordinates["lat"], coordinates["lon"]], dims=["time", "lat", "lon"]
                )
                data = np.arange(c.size).reshape(c.shape)
                return self.create_output_array(c, data=data)

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        node = MyAlgorithm()
        output = node.eval(coords)
        assert isinstance(output, podpac.UnitsDataArray)
        assert output.dims == ("lat", "lon", "time")
        np.testing.assert_array_equal(output, np.arange(12).reshape(2, 3, 2).transpose(1, 2, 0))

    def test_eval_algorithm_transposes(self):
        class MyAlgorithm(Algorithm):
            def algorithm(self, inputs, coordinates):
                c = coordinates.transpose("lon", "lat")
                data = np.arange(c.size).reshape(c.shape)
                return self.create_output_array(c, data=data)

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        node = MyAlgorithm()
        output = node.eval(coords)
        assert isinstance(output, podpac.UnitsDataArray)
        assert output.dims == ("lat", "lon")
        np.testing.assert_array_equal(output, np.arange(6).reshape(2, 3).T)

    def test_eval_with_output(self):
        class MyAlgorithm(Algorithm):
            def algorithm(self, inputs, coordinates):
                data = np.arange(coordinates.size).reshape(coordinates.shape)
                return self.create_output_array(coordinates, data=data)

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        output = podpac.UnitsDataArray.create(coords)

        node = MyAlgorithm()
        node.eval(podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"]), output=output)
        assert isinstance(output, podpac.UnitsDataArray)
        assert output.dims == ("lat", "lon")
        np.testing.assert_array_equal(output, np.arange(6).reshape(3, 2))

    def test_eval_with_output_missing_dims(self):
        class MyAlgorithm(Algorithm):
            def algorithm(self, inputs, coordinates):
                data = np.arange(coords.size).reshape(coords.shape)
                return self.create_output_array(coords, data=data)

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        output = podpac.UnitsDataArray.create(coords.drop("lat"))

        node = MyAlgorithm()
        with pytest.raises(podpac.NodeException, match="provided output is missing dims"):
            node.eval(coords, output=output)

    def test_eval_with_output_transposed(self):
        class MyAlgorithm(Algorithm):
            def algorithm(self, inputs, coordinates):
                data = np.arange(coordinates.size).reshape(coordinates.shape)
                return self.create_output_array(coordinates, data=data)

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        output = podpac.UnitsDataArray.create(coords).transpose("lon", "lat")

        node = MyAlgorithm()
        node.eval(coords, output=output)
        assert isinstance(output, podpac.UnitsDataArray)
        assert output.dims == ("lon", "lat")
        np.testing.assert_array_equal(output, np.arange(6).reshape(3, 2).T)

    def test_eval_with_output_algorithm_returns_xarray(self):
        class MyAlgorithm(Algorithm):
            def algorithm(self, inputs, coordinates):
                data = np.arange(coordinates.size).reshape(coordinates.shape)
                return xr.DataArray(data, coords=coordinates.xcoords, dims=coordinates.dims)

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        output = podpac.UnitsDataArray.create(coords)

        node = MyAlgorithm()
        node.eval(coords, output=output)
        assert isinstance(output, podpac.UnitsDataArray)
        assert output.dims == ("lat", "lon")
        np.testing.assert_array_equal(output, np.arange(6).reshape(3, 2))

    def test_eval_with_output_algorithm_drops_dimension(self):
        class MyAlgorithm(Algorithm):
            drop = tl.Unicode().tag(attr=True)

            def algorithm(self, inputs, coordinates):
                c = coordinates.drop(self.drop)
                data = np.arange(c.size).reshape(c.shape)
                return self.create_output_array(c, data=data)

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])

        output = podpac.UnitsDataArray.create(coords)
        node = MyAlgorithm(drop="lat")
        node.eval(coords, output=output)
        assert isinstance(output, podpac.UnitsDataArray)
        assert output.dims == ("lat", "lon")
        np.testing.assert_array_equal(output, [np.arange(2), np.arange(2), np.arange(2)])

        output = podpac.UnitsDataArray.create(coords)
        node = MyAlgorithm(drop="lon")
        node.eval(coords, output=output)
        assert isinstance(output, podpac.UnitsDataArray)
        assert output.dims == ("lat", "lon")
        np.testing.assert_array_equal(output, np.array([np.arange(3), np.arange(3)]).T)

    def test_eval_with_output_algorithm_adds_dimension(self):
        class MyAlgorithm(Algorithm):
            def algorithm(self, inputs, coordinates):
                c = podpac.Coordinates(
                    [["2020-01-01", "2020-01-02"], coordinates["lat"], coordinates["lon"]], dims=["time", "lat", "lon"]
                )
                data = np.arange(c.size).reshape(c.shape)
                return self.create_output_array(c, data=data)

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])

        output = podpac.UnitsDataArray.create(coords)
        node = MyAlgorithm()
        with pytest.raises(podpac.NodeException, match="provided output is missing dims"):
            node.eval(coords, output=output)

    def test_eval_with_output_algorithm_transposes(self):
        class MyAlgorithm(Algorithm):
            def algorithm(self, inputs, coordinates):
                c = coordinates.transpose("lon", "lat")
                data = np.arange(c.size).reshape(c.shape)
                return self.create_output_array(c, data=data)

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        output = podpac.UnitsDataArray.create(coords)

        node = MyAlgorithm()
        node.eval(coords, output=output)
        assert isinstance(output, podpac.UnitsDataArray)
        assert output.dims == ("lat", "lon")
        np.testing.assert_array_equal(output, np.arange(6).reshape(2, 3).T)

    def test_eval_algorithm_inputs(self):
        class MyAlgorithm(Algorithm):
            x = tl.Instance(podpac.Node).tag(attr=True)

            def algorithm(self, inputs, coordinates):
                return inputs["x"]

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])

        a = podpac.data.Array(source=np.arange(6).reshape(coords.shape), coordinates=coords)
        node = MyAlgorithm(x=a)
        output = node.eval(coords)
        assert output.dims == ("lat", "lon")
        np.testing.assert_array_equal(output.data, a.source)

    def test_eval_multiple_outputs(self):
        class MyAlgorithm(Algorithm):
            x = tl.Instance(podpac.Node).tag(attr=True)
            y = tl.Instance(podpac.Node).tag(attr=True)
            outputs = ["sum", "prod", "diff"]

            def algorithm(self, inputs, coordinates):
                sum_ = inputs["x"] + inputs["y"]
                prod = inputs["x"] * inputs["y"]
                diff = inputs["x"] - inputs["y"]
                coords = podpac.Coordinates.from_xarray(prod)
                return self.create_output_array(coords, data=np.stack([sum_, prod, diff], -1))

        coords = podpac.Coordinates([[0, 1, 2], [10, 20]], dims=["lat", "lon"])
        x = podpac.data.Array(source=np.arange(coords.size).reshape(coords.shape), coordinates=coords)
        y = podpac.data.Array(source=np.full(coords.shape, 2), coordinates=coords)

        # all outputs
        node = MyAlgorithm(x=x, y=y)
        result = node.eval(coords)
        assert result.dims == ("lat", "lon", "output")
        np.testing.assert_array_equal(result["output"], ["sum", "prod", "diff"])
        np.testing.assert_array_equal(result.sel(output="sum"), x.source + y.source)
        np.testing.assert_array_equal(result.sel(output="prod"), x.source * y.source)
        np.testing.assert_array_equal(result.sel(output="diff"), x.source - y.source)

        # extract an output
        node = MyAlgorithm(x=x, y=y, output="prod")
        result = node.eval(coords)
        assert result.dims == ("lat", "lon")
        np.testing.assert_array_equal(result, x.source * y.source)

    def test_eval_multi_threading(self):
        class MySum(Algorithm):
            A = tl.Instance(podpac.Node).tag(attr=True)
            B = tl.Instance(podpac.Node).tag(attr=True)

            def algorithm(self, inputs, coordinates):
                return sum(o for o in inputs.values() if o is not None)

        coords = podpac.Coordinates([[1, 2, 3]], ["lat"])
        array_node = podpac.data.Array(source=np.ones(coords.shape), coordinates=coords)

        with podpac.settings:
            podpac.settings.set_unsafe_eval(True)
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["DEFAULT_CACHE"] = []
            podpac.settings["RAM_CACHE_ENABLED"] = False

            node1 = MySum(A=array_node, B=array_node)
            node2 = MySum(A=node1, B=array_node)

            # multithreaded
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 8
            omt = node2.eval(coords)

            # single threaded
            podpac.settings["MULTITHREADING"] = False
            ost = node2.eval(coords)

            np.testing.assert_array_equal(omt, ost)

    def test_eval_multi_threading_cache_race(self):
        class MyPow(Algorithm):
            source = tl.Instance(podpac.Node).tag(attr=True)
            exponent = tl.Float().tag(attr=True)

            def algorithm(self, inputs, coordinates):
                return inputs["source"] ** self.exponent

        class MySum(Algorithm):
            A = tl.Instance(podpac.Node).tag(attr=True)
            B = tl.Instance(podpac.Node).tag(attr=True)
            C = tl.Instance(podpac.Node).tag(attr=True)
            D = tl.Instance(podpac.Node).tag(attr=True)
            E = tl.Instance(podpac.Node).tag(attr=True)
            F = tl.Instance(podpac.Node).tag(attr=True)

            def algorithm(self, inputs, coordinates):
                return sum(o for o in inputs.values() if o is not None)

        coords = podpac.Coordinates([np.linspace(0, 1, 1024)], ["lat"])
        array_node = podpac.data.Array(source=np.ones(coords.shape), coordinates=coords)

        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 3
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = True
            podpac.settings["DEFAULT_CACHE"] = ["ram"]
            podpac.settings["RAM_CACHE_ENABLED"] = True
            podpac.settings.set_unsafe_eval(True)
            A = MyPow(source=array_node, exponent=2)
            B = MyPow(source=array_node, exponent=2)
            C = MyPow(source=array_node, exponent=2)
            D = MyPow(source=array_node, exponent=2)
            E = MyPow(source=array_node, exponent=2)
            F = MyPow(source=array_node, exponent=2)

            node2 = MySum(A=A, B=B, C=C, D=D, E=E, F=F)
            om = node2.eval(coords)
            assert sum(n._from_cache for n in node2.inputs.values()) > 0

    def test_eval_multi_threading_stress_nthreads(self):
        class MyPow(Algorithm):
            source = tl.Instance(podpac.Node).tag(attr=True)
            exponent = tl.Float().tag(attr=True)

            def algorithm(self, inputs, coordinates):
                return inputs["source"] ** self.exponent

        class MySum(Algorithm):
            A = tl.Instance(podpac.Node).tag(attr=True)
            B = tl.Instance(podpac.Node).tag(attr=True)
            C = tl.Instance(podpac.Node).tag(attr=True)
            D = tl.Instance(podpac.Node).tag(attr=True)
            E = tl.Instance(podpac.Node).tag(attr=True)
            F = tl.Instance(podpac.Node).tag(attr=True)
            G = tl.Instance(podpac.Node, allow_none=True).tag(attr=True)

            def algorithm(self, inputs, coordinates):
                return sum(o for o in inputs.values() if o is not None)

        coords = podpac.Coordinates([np.linspace(0, 1, 4)], ["lat"])
        array_node = podpac.data.Array(source=np.ones(coords.shape), coordinates=coords)

        A = MyPow(source=array_node, exponent=0)
        B = MyPow(source=array_node, exponent=1)
        C = MyPow(source=array_node, exponent=2)
        D = MyPow(source=array_node, exponent=3)
        E = MyPow(source=array_node, exponent=4)
        F = MyPow(source=array_node, exponent=5)

        node2 = MySum(A=A, B=B, C=C, D=D, E=E, F=F)
        node3 = MySum(A=A, B=B, C=C, D=D, E=E, F=F, G=node2)

        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 8
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["DEFAULT_CACHE"] = []
            podpac.settings["RAM_CACHE_ENABLED"] = False
            podpac.settings.set_unsafe_eval(True)

            omt = node3.eval(coords)

        assert node3._multi_threaded
        assert not node2._multi_threaded

        with podpac.settings:
            podpac.settings["MULTITHREADING"] = True
            podpac.settings["N_THREADS"] = 9  # 2 threads available after first 7
            podpac.settings["CACHE_NODE_OUTPUT_DEFAULT"] = False
            podpac.settings["DEFAULT_CACHE"] = []
            podpac.settings["RAM_CACHE_ENABLED"] = False
            podpac.settings.set_unsafe_eval(True)

            omt = node3.eval(coords)

        assert node3._multi_threaded
        assert node2._multi_threaded


class TestUnaryAlgorithm(object):
    def test_outputs(self):
        node = UnaryAlgorithm(source=podpac.data.Array())
        assert node.outputs == None

        node = UnaryAlgorithm(source=podpac.data.Array(outputs=["a", "b"]))
        assert node.outputs == ["a", "b"]
