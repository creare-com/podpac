from __future__ import division, unicode_literals, print_function, absolute_import

import unittest
import numpy as np
import xarray as xr

from podpac.core.coordinate import Coord, Coordinate

class TestCoordCreation(unittest.TestCase):
    def test_single_coord(self):
        coord = Coord(coords=0.25)
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'single')
    
    def test_single_stacked_coord(self):
        coord = Coord(coords=[(0.25, 0.5, 1.2)])
        self.assertEqual(coord.stacked, 3)
        self.assertEqual(coord.regularity, 'single')    
    
    def test_unstacked_regular(self):
        coord = Coord(coords=(0, 1, 4))
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'regular')
        coord = Coord(coords=[0, 1, 4])
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'regular')
        coord = Coord(coords=(0, 1, 1/4))
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'regular')
        coord = Coord(coords=[0, 1, 1/4])
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'regular')
        
    def test_unstacked_irregular(self):
        coord = Coord(coords=np.linspace(0, 1, 4))
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'irregular')
        
    def test_unstacked_dependent(self):
        coord = Coord(coords=xr.DataArray(
            np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
                      dims=['lat', 'lon']))
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'dependent')
        
    def test_stacked_regular(self):
        coord = Coord(coords=((0, 0), (1, -1), 4))
        self.assertEqual(coord.stacked, 2)
        self.assertEqual(coord.regularity, 'regular')        
        coord = Coord(coords=[(0, 0), (1, -1), 4])
        self.assertEqual(coord.stacked, 2)
        self.assertEqual(coord.regularity, 'regular')                
        coord = Coord(coords=((0, 0), (1, -1), 1/4))
        self.assertEqual(coord.stacked, 2)
        self.assertEqual(coord.regularity, 'regular')                
        coord = Coord(coords=[(0, 0), (1, -1), 1/4])
        self.assertEqual(coord.stacked, 2)
        self.assertEqual(coord.regularity, 'regular')                
        
    def test_stacked_irregular(self):
        coord = Coord(coords=np.column_stack((np.linspace(0, 1, 4),
                                              np.linspace(0, -1, 4))))
        self.assertEqual(coord.stacked, 2)
        self.assertEqual(coord.regularity, 'irregular')
        
    def test_stacked_dependent(self):
        coord = Coord(coords=[
            xr.DataArray(
                         np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0],
                         dims=['lat-lon', 'time']), 
            xr.DataArray(
                    np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[1],
                             dims=['lat-lon', 'time'])        
        ])
        self.assertEqual(coord.stacked, 2)
        self.assertEqual(coord.regularity, 'dependent')        
        coord = Coord(coords=
            xr.DataArray(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5)),
                         dims=['stack', 'lat-lon', 'time']), 
        )
        self.assertEqual(coord.stacked, 2)
        self.assertEqual(coord.regularity, 'dependent')                