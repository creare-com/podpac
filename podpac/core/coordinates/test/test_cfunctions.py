
import pytest
import numpy as np

from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates1d import StackedCoordinates
from podpac.core.coordinates.cfunctions import crange, clinspace

def test_crange():
    c = crange(0, 1, 0.2)
    assert isinstance(c, UniformCoordinates1d)
    assert c.start == 0.0
    assert c.stop == 1.0
    assert c.step == 0.2

    c = crange('2018-01-01', '2018-01-05', '1,D')
    assert isinstance(c, UniformCoordinates1d)
    assert c.start == np.datetime64('2018-01-01')
    assert c.stop == np.datetime64('2018-01-05')
    assert c.step == np.timedelta64(1, 'D')

def test_clinspace():
    c = clinspace(0, 1, 6)
    assert isinstance(c, UniformCoordinates1d)
    assert c.start == 0.0
    assert c.stop == 1.0
    assert c.size == 6

    c = clinspace('2018-01-01', '2018-01-05', 5)
    assert isinstance(c, UniformCoordinates1d)
    assert c.start == np.datetime64('2018-01-01')
    assert c.stop == np.datetime64('2018-01-05')
    assert c.size == 5

def test_clinspace_stacked():
    c = clinspace((0, 10, '2018-01-01'), (1, 20, '2018-01-06'), 6)
    assert isinstance(c, StackedCoordinates)
    
    c1, c2, c3 = c
    assert isinstance(c1, UniformCoordinates1d)
    assert c1.start == 0.0
    assert c1.stop == 1.0
    assert c1.size == 6
    assert isinstance(c2, UniformCoordinates1d)
    assert c2.start == 10.0
    assert c2.stop == 20.0
    assert c2.size == 6
    assert isinstance(c3, UniformCoordinates1d)
    assert c3.start == np.datetime64('2018-01-01')
    assert c3.stop == np.datetime64('2018-01-06')
    assert c3.size == 6

    # size must be an integer
    with pytest.raises(TypeError):
        clinspace((0, 10), (1, 20), (6, 6))

    with pytest.raises(TypeError):
        clinspace((0, 10), (1, 20), 0.2)

    with pytest.raises(TypeError):
        clinspace((0, 10), (1, 20), (0.2, 1.0))