from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import xarray as xr
from pint.errors import DimensionalityError
from pint import UnitRegistry
ureg = UnitRegistry()

from podpac.core.coordinate import Coordinate
from podpac.core.algorithm.algorithm import Algorithm, SinCoords, Arithmetic
from podpac.core.algorithm.coord_select import ExpandCoordinates, SelectCoordinates

class TestCoordSelect(object):
    def test_coord_select(self):
        ### This was taken from the bottom of the coord_select file
        # This still needs to be broken out properly, and tested properly
        from podpac.core.algorithm.algorithm import Arange
        from podpac.core.data.data import DataSource
        
        coords = Coordinate(
            time='2017-09-01',
            lat=(45., 66., 4),
            lon=(-80., -70., 5),
            order=('time', 'lat', 'lon'))
    
        # source
        o = Arange().execute(coords)
        print(o.coords)
    
        # node
        node = ExpandCoordinates(source=Arange())
    
        # no expansion
        o = node.execute(coords)
        print(o.coords)
    
        # basic time expansion
        o = node.execute(coords, params={'time': ('-15,D', '0,D', '1,D') })
        print(o.coords)
    
        # basic spatial expansion
        o = node.execute(coords, params={'lat': (-1, 1, 0.1) })
        print(o.coords)
    
        # select node
        snode = SelectCoordinates(source=Arange())
    
        # no expansion of select 
        o = snode.execute(coords)
        print(o.coords)
    
        # basic time selection
        o = snode.execute(coords, params={'time': ('2017-08-01', '2017-09-30', '1,D') })
        print(o.coords)
    
        # basic spatial selection
        o = node.execute(coords, params={'lat': (46, 56, 1) })
        print(o.coords)
    
        # time expansion using native coordinates
        class Test(DataSource):
            def get_native_coordinates(self):
                return Coordinate(
                    time=('2010-01-01', '2018-01-01', '4,h'),
                    lat=(-180., 180., 1800),
                    lon=(-80., -70., 1800),
                    order=('time', 'lat', 'lon'))
    
            def get_data(self, coordinates, slc):
                node = Arange()
                return node.execute(coordinates)
        
        node = Test()
        o = node.execute(coords)
        print (o.coords)
        
        # node
        node = ExpandCoordinates(source=Test())
        o = node.execute(coords, params={'time': ('-15,D', '0,D')})
        print (o.coords)
    
        node._params={'time': ('-15,Y', '0,D', '1,Y')}
        print (node.get_expanded_coord('time'))
    
        o = node.execute(coords, params={'time': ('-5,M', '0,D', '1,M')})
        print (o.coords)
        
        node._params={'time': ('-15,Y', '0,D', '4,Y')}  # Behaviour a little strange
        print (node.get_expanded_coord('time'))
        
        node._params={'time': ('-15,Y', '0,D', '13,M')}  # Behaviour a little strange
        print (node.get_expanded_coord('time'))
    
        node._params={'time': ('-144,M', '0,D', '13,M')}  # Behaviour a little strange
        print (node.get_expanded_coord('time'))
    
        # select node
        node = SelectCoordinates(source=Test())
        o = node.execute(coords, params={'time': ('2011-01-01', '2011-02-01')})
        print (o.coords)
    
        node._params={'time': ('2011-01-01', '2017-01-01', '1,Y')}
        print (node.get_expanded_coord('time'))
    
        print ('Done')        