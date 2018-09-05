from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np

from podpac.core.coordinates import Coordinates, UniformCoordinates1d
from podpac.core.algorithm.algorithm import Algorithm, Arange, CoordData, SinCoords, Arithmetic

class TestAlgorithm(object):
    def test_not_implemented(self):
        node = Algorithm()
        with pytest.raises(NotImplementedError):
            node.execute(Coordinates())

    def test_pipeline_definition(self):
        # note: any algorithm node with attrs and inputs would be fine here
        node = Arithmetic(A=Arange(), B=Arange(), eqn='A+B')
        d = node.definition
        
        assert isinstance(d, dict)
        
        # base (node, params)
        assert d['node'] == 'core.algorithm.algorithm.Arithmetic'
        assert d['attrs']['eqn'] == 'A+B'
        
        # inputs
        assert 'inputs' in d
        assert isinstance(d['inputs'], dict)
        assert 'A' in d['inputs']
        assert 'B' in d['inputs']

        #TODO value of d['inputs']['A'], etc

class TestArange(object):
    def test_Arange(self):
        coords = Coordinates([
            UniformCoordinates1d(0, 1, size=10, name='lat'),
            UniformCoordinates1d(0, 1, size=20, name='lon')
        ])
        node = Arange()
        output = node.execute(coords)
        assert output.shape == coords.shape

class TestCoordData(object):
    def test_CoordData(self):
        coords = Coordinates([
            UniformCoordinates1d(0, 1, size=10, name='lat'),
            UniformCoordinates1d(0, 1, size=20, name='lon')
        ])

        node = CoordData(coord_name='lat')
        np.testing.assert_array_equal(node.execute(coords), coords.coords['lat'])

        node = CoordData(coord_name='lon')
        np.testing.assert_array_equal(node.execute(coords), coords.coords['lon'])

    def test_invalid_dimension(self):
        coords = Coordinates([
            UniformCoordinates1d(0, 1, size=10, name='lat'),
            UniformCoordinates1d(0, 1, size=20, name='lon')
        ])

        node = CoordData(coord_name='time')
        with pytest.raises(ValueError):
            node.execute(coords)

class TestSinCoords(object):
    def test_SinCoords(self):
        coords = Coordinates([
            UniformCoordinates1d(-90, 90, 1.0, name='lat'),
            UniformCoordinates1d('2018-01-01', '2018-01-30', '1,D', name='time')
        ])
        node = SinCoords()
        output = node.execute(coords)
        assert output.shape == coords.shape
        
class TestArithmetic(object):
    @pytest.mark.skip("add_unique, plus algorithm execute weirdness")
    def test_Arithmetic(self):
        coords = Coordinates([
            UniformCoordinates1d(-90, 90, 1.0, name='lat'),
            UniformCoordinates1d(-180, 180, 1.0, name='lon')
        ])
        sine_node = SinCoords()
        node = Arithmetic(A=sine_node, B=sine_node, eqn='2*abs(A) - B + {offset}', params={'offset': 1})
        output = node.execute(coords)

        a = sine_node.execute(coords)
        b = sine_node.execute(coords)
        np.testing.assert_allclose(output, 2*abs(a) - b + 1)

    @pytest.mark.skip("add_unique, plus algorithm execute weirdness")
    def test_missing_equation(self):
        coords = Coordinates([
            UniformCoordinates1d(-90, 90, 1.0, name='lat'),
            UniformCoordinates1d(-180, 180, 1.0, name='lon')
        ])

        sine_node = SinCoords()
        node = Arithmetic(A=sine_node, B=sine_node)
        
        with pytest.raises(ValueError):
            node.execute(coords)