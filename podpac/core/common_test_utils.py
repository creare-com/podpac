"""
Utils Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
from collections import OrderedDict

import numpy as np

from podpac.core.coordinates import Coordinates, UniformCoordinates1d

def make_coordinate_combinations(lat=None, lon=None, alt=None, time=None):
    ''' Generates every combination of stacked and unstacked coordinates podpac expects to handle
    
    Parameters
    -----------
    lat: podpac.core.coordinates.Coordinates1d, optional
        1D coordinate object used to create the Coordinate objects that contain the latitude dimension. By default uses:
        UniformCoord(start=0, stop=2, delta=1.0)
    lon: podpac.core.coordinates.Coordinates1d, optional
        Same as above but for longitude. By default uses:
        UniformCoord(start=2, stop=6, delta=2.0)
    alt: podpac.core.coordinates.Coordinates1d, optional
        Same as above but for longitude. By default uses:
        UniformCoord(start=6, stop=12, delta=3.0)
    time: podpac.core.coordinates.Coordinates1d, optional
        Same as above but for longitude. By default uses:
        UniformCoord(start='2018-01-01T00:00:00', stop='2018-03-01T00:00:00', delta='1,M')

    Returns
    -------
    OrderedDict:
        Dictionary of all the podpac.Core.Coordinate objects podpac expects to handle. The dictionary keys is a tuple of
        coordinate dimensions, and the values are the actual Coordinate objects.
        
    Notes
    ------
    When custom lat, lon, alt, and time 1D coordinates are given, only those with the same number of coordinates are 
    stacked together. For example, if lat, lon, alt, and time have sizes 3, 4, 5, and 6, respectively, no stacked 
    coordinates are created. Also, no exception or warning is thrown for this case. 
    '''
    
    coord_collection = OrderedDict()
    d = OrderedDict((['lat', lat], ['lon', lon], ['alt', alt], ['time', time]))
    
    # Make all of the 1D coordinates
    if lat is None:
        d['lat'] = UniformCoordinates1d(0, 2, 1.0)
    if lon is None:
        d['lon'] = UniformCoordinates1d(2, 6, 2.0)
    if alt is None:
        d['alt'] = UniformCoordinates1d(6, 12, 3.0)
    if time is None:
        d['time'] = UniformCoordinates1d('2018-01-01T00:00:00', '2018-03-01T00:00:00', '1,M')
    
    
    # Make the singular and unstacked coordinate combinations
    # Create recursive helper function
    def recurse_coord(coord_collection, indep_coords, coords1d=OrderedDict(), depth=0):
        #print ("Depth", depth)
        if len(indep_coords) == depth:
            coord_collection[tuple(coords1d.keys())] = Coordinates(coords1d)
            coords1d = OrderedDict()
            return coord_collection
        else:
            for i in range(len(indep_coords)):
                crd = indep_coords[i]
                kwargs_plus = coords1d.copy()
                kwargs_plus.update(OrderedDict([[c, d[c]] for c in crd]))
                #print ("Depth", depth, ', i', i)
                coord_collection = recurse_coord(coord_collection, indep_coords, kwargs_plus, depth+1)
            return coord_collection


    # In general lat and lon should be grouped, so we have to swap their order manually. 
    indep_coords = [('lat', 'lon'), ('alt',), ('time',)]
    coord_collection = recurse_coord(coord_collection, indep_coords)
    indep_coords = [('lon', 'lat'), ('alt',), ('time',)]
    coord_collection = recurse_coord(coord_collection, indep_coords) # Some duplication here, but ok
    indep_coords = [('lat', ), ('alt',), ('time',)]
    coord_collection = recurse_coord(coord_collection, indep_coords) # Some duplication here, but ok
    indep_coords = [('lon', ), ('alt',), ('time',)]
    coord_collection = recurse_coord(coord_collection, indep_coords) # Some duplication here, but ok
    
    
    
    # Now add the stacked variants
    for key, val in coord_collection.copy().items():
        if len(key) == 1:
            continue
        # This is not general, but easy to handle all the cases
        elif len(key) == 2:
            try:
                stacked_dims = val.dims
                coord_collection[('_'.join(stacked_dims),)] = val.stack(stacked_dims)
            except: pass
        elif len(key) == 3:
            stacked_dims = val.dims[:2]
            full_stacked = False
            if (('lat' in stacked_dims) and ('lon' in val.dims) and ('lon' not in stacked_dims)) \
                    or (('lon' in stacked_dims) and ('lat' in val.dims) and ('lat' not in stacked_dims)):
                pass # case covered by full stacking
            else:
                k = ('_'.join(stacked_dims), ) + (val.dims[2],)
                try: 
                    coord_collection[k] = val.stack(stacked_dims)
                except: pass

            stacked_dims = val.dims[1:]
            if (('lat' in stacked_dims) and ('lon' in val.dims) and ('lon' not in stacked_dims)) \
                    or (('lon' in stacked_dims) and ('lat' in val.dims) and ('lat' not in stacked_dims)):
                pass # case covered by full stacking
            else:
                k =  (val.dims[0],) + ('_'.join(stacked_dims), )
                try: 
                    coord_collection[k] = val.stack(stacked_dims)
                except: pass
            try:
                coord_collection[('_'.join(val.dims))] = val.stack(val.dims)
            except: pass
            
        elif len(key) == 4: # actually 3 because lat-lon is a pair
            # Find lat, lon position
            lat_i = val.dims.index('lat')
            lon_i = val.dims.index('lon')
            ll_i = min(lat_i, lon_i)
            
            # Start stacking
            mid_i = 2 + (ll_i <= 1)
            stacked_dims = val.dims[:mid_i]
            k = (('_'.join(stacked_dims)), ) + tuple(val.dims[mid_i:])
            try: 
                coord_collection[k] = val.stack(stacked_dims)
            except:
                pass 
            
            mid_i = 1 + (ll_i == 0)
            stacked_dims = val.dims[mid_i:]
            k = tuple(val.dims[:mid_i]) + (('_'.join(stacked_dims)), )
            try: 
                coord_collection[k] = val.stack(stacked_dims)
            except: pass
            
            stacked_dims = val.dims[ll_i:ll_i + 2]
            k = tuple(val.dims[:ll_i]) + (('_'.join(stacked_dims)), ) + tuple(val.dims[ll_i+2:])
            try: 
                coord_collection[k] = val.stack(stacked_dims)            
            except: pass
            
            if ll_i == 0:
                try:
                    v = val.stack(stacked_dims)
                    v = v.stack(val.dims[2:])
                    k = (('_'.join(stacked_dims)), ) + ('_'.join(val.dims[2:]),)
                    coord_collection[k] = v
                except: pass
            elif ll_i == 2:
                try: 
                    v = val.stack(stacked_dims)
                    v = v.stack(val.dims[:2])
                    k = ('_'.join(val.dims[:2]),) + (('_'.join(stacked_dims)), ) 
                    coord_collection[k] = v
                except: pass
            
            try: 
                coord_collection[('_'.join(val.dims))] = val.stack(val.dims)
            except:
                pass
    return coord_collection

