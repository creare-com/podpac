from __future__ import division, unicode_literals, print_function, absolute_import

import unittest
import numpy as np
from pint.errors import DimensionalityError
from pint import UnitRegistry
ureg = UnitRegistry()

from podpac.core.units import UnitsDataArray

class TestUnitDataArray(unittest.TestCase):
    def test_no_units_coord(self):
        a1 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                           attrs={})
        a2 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                           attrs={})    
        a3 = a1 + a2
        a3b = a2 + a1
        a4 = a1 > a2
        a5 = a1 < a2
        a6 = a1 == a2
        a7 = a1 * a2
        a8 = a2 / a1
        a9 = a1 // a2
        a10 = a1 % a2        
        
    def test_first_units_coord(self):
        a1 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                           attrs={'units': ureg.meter})
        a2 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                           attrs={})    
        with self.assertRaises(DimensionalityError):
            a3 = a1 + a2
        with self.assertRaises(DimensionalityError):
            a3b = a2 + a1
        with self.assertRaises(DimensionalityError):
            a4 = a1 > a2
        with self.assertRaises(DimensionalityError):
            a5 = a1 < a2
        with self.assertRaises(DimensionalityError):
            a6 = a1 == a2
        a7 = a1 * a2
        a8 = a2 / a1
        with self.assertRaises(DimensionalityError):
            a9 = a1 // a2
        with self.assertRaises(DimensionalityError):
            a10 = a1 % a2        

    def test_second_units_coord(self):
        a1 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                           attrs={})
        a2 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                           attrs={'units': ureg.inch})    
        with self.assertRaises(DimensionalityError):
            a3 = a1 + a2
        with self.assertRaises(DimensionalityError):
            a3b = a2 + a1
        with self.assertRaises(DimensionalityError):
            a4 = a1 > a2
        with self.assertRaises(DimensionalityError):
            a5 = a1 < a2
        with self.assertRaises(DimensionalityError):
            a6 = a1 == a2
        a7 = a1 * a2
        a8 = a2 / a1
        with self.assertRaises(DimensionalityError):
            a9 = a1 // a2
        with self.assertRaises(DimensionalityError):
            a10 = a1 % a2        
        
    def test_units_allpass(self):
        a1 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                           attrs={'units': ureg.meter})
        a2 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                           attrs={'units': ureg.inch})    
        a3 = a1 + a2
        self.assertEqual(a3[0, 0].data[()],
                         (1*ureg.meter + 1*ureg.inch).to(ureg.meter).magnitude)
        a3b = a2 + a1
        self.assertEqual(a3b[0, 0].data[()],
                         (1*ureg.meter + 1*ureg.inch).to(ureg.inch).magnitude)        
        a4 = a1 > a2
        self.assertEqual(a4[0, 0].data[()], True)        
        a5 = a1 < a2
        self.assertEqual(a5[0, 0].data[()], False)        
        a6 = a1 == a2
        self.assertEqual(a6[0, 0].data[()], False)        
        a7 = a1 * a2
        self.assertEqual(a7[0, 0].to(ureg.m**2).data[()],
                         (1*ureg.meter*ureg.inch).to(ureg.meter**2).magnitude)
        a8 = a2 / a1
        self.assertEqual(a8[0, 0].to_base_units().data[()],
                         (1*ureg.inch/ureg.meter).to_base_units().magnitude)        
        a9 = a1 // a2
        self.assertEqual(a9[0, 0].to_base_units().data[()],
                         ((1*ureg.meter) // (1*ureg.inch)).to_base_units().magnitude)                
        a10 = a1 % a2        
        self.assertEqual(a10[0, 0].to_base_units().data[()],
                         ((1*ureg.meter) % (1*ureg.inch)).to_base_units().magnitude)                
    
    def test_units_somefail(self):
        a1 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                               attrs={'units': ureg.meter})
        a2 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                               attrs={'units': ureg.kelvin})    
        with self.assertRaises(DimensionalityError):
            a3 = a1 + a2
        with self.assertRaises(DimensionalityError):
            a3b = a2 + a1
        with self.assertRaises(DimensionalityError):
            a4 = a1 > a2
        with self.assertRaises(DimensionalityError):
            a5 = a1 < a2
        with self.assertRaises(DimensionalityError):
            a6 = a1 == a2
        
        a7 = a1 * a2
        self.assertEqual(a7[0, 0].to(ureg.meter * ureg.kelvin).data[()],
                             (1*ureg.meter*ureg.kelvin).magnitude)
        a8 = a1 / a2
        self.assertEqual(a8[0, 0].to(ureg.meter / ureg.kelvin).data[()],
                     (1*ureg.meter/ureg.kelvin).magnitude)
        with self.assertRaises(DimensionalityError):
            a9 = a1 // a2
        with self.assertRaises(DimensionalityError):
            a10 = a1 % a2        
            
    def test_ufuncs(self):
        a1 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                               attrs={'units': ureg.meter})
        a2 = UnitsDataArray(np.ones((4,3)), dims=['lat', 'lon'],
                               attrs={'units': ureg.kelvin}) 
        
        np.sqrt(a1)
        np.mean(a1)
        np.min(a1)
        np.max(a1)
        a1 ** 2
        
        # These don't have units!
        np.dot(a2.T, a1)
        np.std(a1)
        np.var(a1)        
