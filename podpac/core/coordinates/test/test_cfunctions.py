
import pytest
import numpy as np

from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates
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
    # numerical
    c = clinspace(0, 1, 6)
    assert isinstance(c, UniformCoordinates1d)
    assert c.start == 0.0
    assert c.stop == 1.0
    assert c.size == 6

    # datetime
    c = clinspace('2018-01-01', '2018-01-05', 5)
    assert isinstance(c, UniformCoordinates1d)
    assert c.start == np.datetime64('2018-01-01')
    assert c.stop == np.datetime64('2018-01-05')
    assert c.size == 5

    # named
    c = clinspace(0, 1, 6, name='lat')
    assert c.name == 'lat'

def test_clinspace_shape_mismatch():
    with pytest.raises(ValueError, match="Size mismatch, 'start' and 'stop' must have the same size"):
        clinspace(0, (0, 10), 6)
