from __future__ import division, unicode_literals, print_function, absolute_import

import pytest

import podpac
from podpac.core.data.datasource import DataSource
from podpac.core.algorithm.algorithm import Arange
from podpac.core.algorithm.coord_select import ExpandCoordinates, SelectCoordinates

# TODO move to test setup
coords = podpac.Coordinates(
    ['2017-09-01', podpac.clinspace(45, 66, 4), podpac.clinspace(-80, -70, 5)],
    dims=['time', 'lat', 'lon'])

class MyDataSource(DataSource):
    def get_native_coordinates(self):
        return podpac.Coordinates([
            podpac.crange('2010-01-01', '2018-01-01', '4,h'),
            podpac.clinspace(-180, 180, 6),
            podpac.clinspace(-80, -70, 6)],
            dims=['time', 'lat', 'lon'])

    def get_data(self, coordinates, slc):
        node = Arange()
        return node.execute(coordinates)

# TODO add assertions to tests
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
        node.get_expanded_coordinates1d('time') # TODO what are we checking here
    
        node = ExpandCoordinates(source=MyDataSource(), time=('-5,M', '0,D', '1,M'))
        o = node.execute(coords)
        
        # Behaviour a little strange on these?
        node = ExpandCoordinates(source=MyDataSource(), time=('-15,Y', '0,D', '4,Y'))
        o = node.execute(coords)
        node.get_expanded_coordinates1d('time') # TODO what are we checking here
        
        node = ExpandCoordinates(source=MyDataSource(), time=('-15,Y', '0,D', '13,M'))
        o = node.execute(coords)
        node.get_expanded_coordinates1d('time') # TODO what are we checking here
    
        node = ExpandCoordinates(source=MyDataSource(), time=('-144,M', '0,D', '13,M'))
        o = node.execute(coords)
        node.get_expanded_coordinates1d('time') # TODO what are we checking here
    
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
        node.get_expanded_coordinates1d('time') # TODO what are we checking here
