from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np

from podpac.core.coordinate import Coordinate
from podpac.core.algorithm.algorithm import Algorithm, Arange, CoordData, SinCoords, Arithmetic

class TestAlgorithm(object):
    def test_Algorithm(self):
        node = Algorithm()

    def test_not_implemented(self):
        c = Coordinate(lat=[0, 1, 10], lon=[0, 1, 20], order=['lat', 'lon'])
        
        node = Algorithm()
        with pytest.raises(NotImplementedError):
            node.execute(c)

    def test_pipeline_definition(self):
        # note: any algorithm node with params and inputs would be fine here
        node = Arithmetic(A=Arange(), B=Arange(), eqn='A+B')
        d = node.definition
        
        assert isinstance(d, dict)
        
        # base (node, params)
        assert d['node'] == 'core.algorithm.algorithm.Arithmetic'
        assert d['params']['eqn'] == 'A+B'
        
        # inputs
        assert 'inputs' in d
        assert isinstance(d['inputs'], dict)
        assert 'A' in d['inputs']
        assert 'B' in d['inputs']

        #TODO value of d['inputs']['A'], etc

class TestArange(object):
    def test_Arange(self):
        c = Coordinate(lat=(0, 1, 10), lon=(0, 1, 20), order=['lat', 'lon'])
        node = Arange()
        output = node.execute(c)
        assert output.shape == c.shape

class TestCoordData(object):
    def test_CoordData(self):
        c = Coordinate(lat=(0, 1, 10), lon=(0, 1, 20), order=['lat', 'lon'])
        node = CoordData(coord_name='lat')
        np.testing.assert_array_equal(node.execute(c), c.coords['lat'])

    def test_invalid_dimension(self):
        c = Coordinate(lat=(0, 1, 10), lon=(0, 1, 20), order=['lat', 'lon'])
        node = CoordData(coord_name='time')
        with pytest.raises(ValueError):
            node.execute(c)

class TestSinCoords(object):
    def test_SinCoords(self):
        c = Coordinate(lat=(-90, 90, 1.), time=('2018-01-01', '2018-01-30', '1,D'), order=['lat', 'time'])
        node = SinCoords()
        output = node.execute(c)
        assert output.shape == c.shape
        
class TestArithmetic(object):
    def test_Arithmetic(self):
        c = Coordinate(lat=(-90, 90, 1.), lon=(-180, 180, 1.), order=['lat', 'lon'])
        sine_node = SinCoords()
        node = Arithmetic(A=sine_node, B=sine_node)
        output = node.execute(c, params={'eqn': '2*abs(A) - B + {offset}', 'offset': 1})

        a = sine_node.execute(c)
        b = sine_node.execute(c)
        np.testing.assert_allclose(output, 2*abs(a) - b + 1)

    def test_missing_equation(self):
        c = Coordinate(lat=(0, 1, 10), lon=(0, 1, 20), order=['lat', 'lon'])
        sine_node = SinCoords()
        node = Arithmetic(A=sine_node, B=sine_node)
        
        with pytest.raises(ValueError):
            node.execute(c)