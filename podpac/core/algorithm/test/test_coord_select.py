from __future__ import division, unicode_literals, print_function, absolute_import

import pytest

from podpac.core.coordinates import Coordinates
from podpac.core.coordinates import UniformCoordinates1d, ArrayCoordinates1d
from podpac.core.data.data import DataSource
from podpac.core.algorithm.algorithm import Arange
from podpac.core.algorithm.coord_select import ExpandCoordinates, SelectCoordinates

# TODO move to test setup
coords = Coordinates([
    ArrayCoordinates1d('2017-09-01', name='time'),
    UniformCoordinates1d(45, 66, size=4, name='lat'),
    UniformCoordinates1d(-80, -70, size=5, name='lon')
])

class MyDataSource(DataSource): # TODO better to use a NumpyArray
    def get_native_coordinates(self):
        return Coordinates([
            UniformCoordinates1d('2010-01-01', '2018-01-01', '4,h', name='time'),
            UniformCoordinates1d(180, 180, size=6, name='lat'),
            UniformCoordinates1d(80, -70, size=6, name='lon')
        ])

    def get_data(self, coordinates, slc):
        node = Arange()
        return node.execute(coordinates)

# TODO add assertions to tests

@pytest.mark.skip("TODO")
class TestExpandCoordinates(object):
    def test_no_expansion(self):
        node = ExpandCoordinates(source=Arange())
        o = node.execute(coords)

    def test_time_expansion(self):
        node = ExpandCoordinates(source=Arange(), time=('-5,D', '0,D', '1,D'))
        o = node.execute(coords)

    def test_spatial_exponsion(self):
        node = ExpandCoordinates(source=Arange(), lat=(-1, 1, 0.1))
        o = node.execute(coords)

    def test_time_expansion_native_coordinates(self):
        node = ExpandCoordinates(source=MyDataSource(), time=('-15,D', '0,D'))
        o = node.execute(coords)
        
        node = ExpandCoordinates(source=MyDataSource(), time=('-15,Y', '0,D', '1,Y'))
        o = node.execute(coords)
        node.get_expanded_coord('time') # TODO what are we checking here
    
        node = ExpandCoordinates(source=MyDataSource(), time=('-5,M', '0,D', '1,M'))
        o = node.execute(coords)
        
        # Behaviour a little strange on these?
        node = ExpandCoordinates(source=MyDataSource(), time=('-15,Y', '0,D', '4,Y'))
        o = node.execute(coords)
        node.get_expanded_coord('time') # TODO what are we checking here
        
        node = ExpandCoordinates(source=MyDataSource(), time=('-15,Y', '0,D', '13,M'))
        o = node.execute(coords)
        node.get_expanded_coord('time') # TODO what are we checking here
    
        node = ExpandCoordinates(source=MyDataSource(), time=('-144,M', '0,D', '13,M'))
        o = node.execute(coords)
        node.get_expanded_coord('time') # TODO what are we checking here
    
@pytest.mark.skip("TODO")
class TestSelectCoordinates(object):
    def test_no_expansion(self):
        node = SelectCoordinates(source=Arange())
        o = node.execute(coords)

    def test_time_selection(self):
        node = SelectCoordinates(source=Arange(), time=('2017-08-01', '2017-09-30', '1,D'))
        o = node.execute(coords)

    def test_spatial_selection(self):
        node = SelectCoordinates(source=Arange(), lat=(46, 56, 1))
        o = node.execute(coords)

    def test_time_selection_native_coordinates(self):
        node = SelectCoordinates(source=MyDataSource(), time=('2011-01-01', '2011-02-01'))
        o = node.execute(coords)
        
        node = SelectCoordinates(source=MyDataSource(), time=('2011-01-01', '2017-01-01', '1,Y'))
        o = node.execute(coords)
        node.get_expanded_coord('time') # TODO what are we checking here