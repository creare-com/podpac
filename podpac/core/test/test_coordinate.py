from __future__ import division, unicode_literals, print_function, absolute_import

import unittest
import numpy as np
import xarray as xr

from podpac.core.coordinate import Coord, Coordinate

# TODO: Add tests that should fail
# TODO: Add tests to test the rest of the interface

class TestCoordCreation(unittest.TestCase):
    def test_single_coord(self):
        coord = Coord(coords=0.25)
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'single')
    
    def test_single_stacked_coord(self):
        coord = Coord(coords=[(0.25, 0.5, 1.2)])
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                      np.array(coord.bounds))
        self.assertEqual(coord.stacked, 3)
        self.assertEqual(coord.regularity, 'single')    
    
    def test_unstacked_regular(self):
        coord = Coord(coords=(0, 1, 4))
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'regular')
        coord = Coord(coords=[0, 1, 4])
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'regular')
        coord = Coord(coords=(0, 1, 1/4))
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'regular')
        coord = Coord(coords=[0, 1, 1/4])
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'regular')
        
    def test_unstacked_irregular(self):
        coord = Coord(coords=np.linspace(0, 1, 4))
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'irregular')
        
    def test_unstacked_dependent(self):
        coord = Coord(coords=xr.DataArray(
            np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
                      dims=['lat', 'lon']))
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
        self.assertEqual(coord.stacked, 1)
        self.assertEqual(coord.regularity, 'dependent')
        
    def test_stacked_regular(self):
        coord = Coord(coords=((0, 0), (1, -1), 4))
        self.assertEqual(coord.stacked, 2)
        self.assertEqual(coord.regularity, 'regular')
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
        coord = Coord(coords=[(0, 0), (1, -1), 4])
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
        self.assertEqual(coord.stacked, 2)
        self.assertEqual(coord.regularity, 'regular')                
        coord = Coord(coords=((0, 0), (1, -1), 1/4))
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
        self.assertEqual(coord.stacked, 2)
        self.assertEqual(coord.regularity, 'regular')                
        coord = Coord(coords=[(0, 0), (1, -1), 1/4])
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
        self.assertEqual(coord.stacked, 2)
        self.assertEqual(coord.regularity, 'regular')                
        
    def test_stacked_irregular(self):
        coord = Coord(coords=np.column_stack((np.linspace(0, 1, 4),
                                              np.linspace(0, -1, 4))))
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
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
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
        self.assertEqual(coord.stacked, 2)
        self.assertEqual(coord.regularity, 'dependent')        
        coord = Coord(coords=
            xr.DataArray(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5)),
                         dims=['stack', 'lat-lon', 'time']), 
        )
        np.testing.assert_array_equal(np.array(coord.intersect(coord).bounds),
                                          np.array(coord.bounds))        
        self.assertEqual(coord.stacked, 2)
        self.assertEqual(coord.regularity, 'dependent')                
        

class TestCoordIntersection(unittest.TestCase):
    def test_regular(self):
        coord = Coord(coords=(1, 10, 10))
        coord_left = Coord(coords=(-2, 7, 10))
        coord_right = Coord(coords=(4, 13, 10))
        coord_cent = Coord(coords=(4, 7, 4))
        coord_cover = Coord(coords=(-2, 13, 15))
        
        c = coord.intersect(coord).coordinates
        np.testing.assert_array_equal(c, coord.coordinates)
        c = coord.intersect(coord_cover).coordinates
        np.testing.assert_array_equal(c, coord.coordinates)        
        
        c = coord.intersect(coord_left).coordinates
        np.testing.assert_array_equal(c, coord.coordinates[:8])                
        c = coord.intersect(coord_right).coordinates
        np.testing.assert_array_equal(c, coord.coordinates[2:])
        c = coord.intersect(coord_cent).coordinates
        np.testing.assert_array_equal(c, coord.coordinates[2:8])
        
class TestCoordinateCreation(unittest.TestCase):
    def test_single_coord(self):
        coord = Coordinate(lat=0.25, lon=0.3)
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
    
    def test_single_stacked_coord(self):
        coord = Coordinate(lat=[(0.25, 0.5, 1.2)], lon=[(0.25, 0.5, 1.2)])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
    
    def test_unstacked_regular(self):
        coord = Coordinate(lat=(0, 1, 4), lon=(0, 1, 4))
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(lat=[0, 1, 4], lon=[0, 1, 4])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(lat=(0, 1, 1/4), lon=(0, 1, 1/4))
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(lat=[0, 1, 1/4], lon=[0, 1, 1/4])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        
    def test_unstacked_irregular(self):
        coord = Coordinate(lat=np.linspace(0, 1, 4), lon=np.linspace(0, 1, 4))
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        
    def test_unstacked_dependent(self):
        coord = Coordinate(lat=xr.DataArray(
            np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
                      dims=['lat', 'lon']),
                           lon=xr.DataArray(
            np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0], 
                    dims=['lat', 'lon'])                           )
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        
    def test_stacked_regular(self):
        coord = Coordinate(lat=((0, 0), (1, -1), 4), lon=((0, 0), (1, -1), 4))
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(lat=[(0, 0), (1, -1), 4], lon=[(0, 0), (1, -1), 4])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(lat=((0, 0), (1, -1), 1/4), lon=((0, 0), (1, -1), 1/4))
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(lat=[(0, 0), (1, -1), 1/4], lon=[(0, 0), (1, -1), 1/4])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        
    def test_stacked_irregular(self):
        coord = Coordinate(lat=np.column_stack((np.linspace(0, 1, 4),
                                              np.linspace(0, -1, 4))),
                           lon=np.column_stack((np.linspace(0, 1, 4),
                                              np.linspace(0, -1, 4))))
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        
    def test_stacked_dependent(self):
        coord = Coordinate(lat=[
            xr.DataArray(
                         np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0],
                         dims=['lat-lon', 'time']), 
            xr.DataArray(
                    np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[1],
                             dims=['lat-lon', 'time'])        
                               ], 
                           lon=[
            xr.DataArray(
                         np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[0],
                         dims=['lat-lon', 'time']), 
            xr.DataArray(
                    np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5))[1],
                             dims=['lat-lon', 'time'])        
        ])
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
        coord = Coordinate(lat=
            xr.DataArray(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5)),
                         dims=['stack', 'lat-lon', 'time']), 
            lon=xr.DataArray(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, -1, 5)),
                         dims=['stack', 'lat-lon', 'time']), 
        )
        np.testing.assert_array_equal(np.array(coord.intersect(coord)._coords['lat'].bounds),
                                          np.array(coord._coords['lat'].bounds))        
