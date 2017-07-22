import xarray as xr
import numpy as np
import traitlets as tl
from collections import OrderedDict




class CoordinateException(Exception):
    pass

class Coord(tl.HasTraits):
    area = tl.Enum(['segment', 'point', 'fence', 'post'], default_value='point')
    segment_position = tl.Integer(default_value=0.5)
    
    tl.validate('segment_position')
    def _segment_position_validate(self, proposal):
        if proposal['value'] <= 1 and proposal['value'] >= 0:
            return proposal["value"]
        else:
            raise CoordinateException("Coordinate dimension '" + self.dim + \
            "' must be in the segment position of [0, 1]")
    
    coords = tl.Instance([tuple, np.ndarray, xr.DataArray]) 
    tl.validate("coords")
    def _coords_validate(self, proposal):
        raise NotImplementedError
        
    _cached_coords = tl.Instance([xr.DataArray, np.ndarray], default_value=None, allow_none=True)
    @property
    def coordinates(self):
        if self._cached_coords is not None:
            return self._cached_coords
            
        if isinstance(coords, np.ndarray):  # independent coords, variable spacing
            self._cached_coords = coords
        elif isinstance(coords, xr.DataArray):  # dependent coord
            self._cached_coords = coords
        elif not isinstance(coords, tuple):  # Not sure what this is, probably a single number (need to test this)
            self._cached_coords = coords
        elif len(self.coords) == 1:  # again, probably a single number, not well tested
            self._cached_coords = np.array(self.coords)
        elif len(self.coords) == 2:  # dependent coord, from multiple points
            raise NotImplementedError("Need to implement case for 3 points or 4 points definition of box")
        elif len(self.coords) == 3:  # independent coords, uniform spacing
            if not isinstance(self.coords[2], int):  # delta specified
                N = (self.coords[1] - self.coords[2] // self.coords[3])
            else:
                N = self.coords[3]
            self._cached_coords = np.linspace(self.coords[0], self.coords[1], N)
            
        return self._cached_coords
        
    tl.observe('coords')
    def _clear_cache(self, change):
        if old != new:
            self._cached_coords = None
        
    def intersect(self, other_coord):
        raise NotImplementedError ("Determine if this overlaps, and then return overlapping coords")
    
class Coordinate(tl.HasTraits):
    """
    You can initialize a coordinate like this: 
    c = Coordinate(lat=1)  # Single number
    c = Coordinate(lat_lon=((1, 2)))  # Single number for stacked coordinate
    c = Coordinate(lat=(49.1, 50.2, 100)  # uniformly spaced range (start, stop, number)
    c = Coordinate(lat_lon=((49.1, -120), (50.2, -122), 100)  # uniform range for stacked coordinate
    c = Coordinate(lat=(49.1, 50.1, 0.1))  # uniformly spaced steps (start, stop, step)
    c = Coordinate(lat_lon=((49.1, -120), (50.2, -122), (0.1, 0.2))  # uniform steps for stacked coordinate
    c = Coordinate(lat=np.array([50, 50.1, 50.4, 50.8, 50.9]))  # specified coordinates
    c = Coordinate(lat_lon=(np.array([50, 50.1, 50.4, 50.8, 50.9]), 
                            np.array([-120, -125, -126, -127, -130])  # specified stacked coordinates
    c = Coordinate(lat=xr.DataArray([[50.1, 50.2, 50.3], [50.2, 50.3, 50.4]], dims=['lat', 'lon']),
                   lon=... ))  # Depended specified coordinates
    c = Coordinate(lat=((50.1, 51.4, 51.2), 100), lon=((120, 120.1, 121.1), 50))  # Dependent from 3 points
    """

    @property
    @staticmethod
    def _valid_dims():
        return ('time', 'lat', 'lon', 'alt')
    
    ctype = tl.Enum(['segment', 'point', 'fence', 'post'])  # default val set in constructor
    segment_position = tl.Integer()  # default val set in constructor
    
    def __init__(self, coords=None, projection="WGS84", ctype=point, segment_position=0.5, bounds=[], **kwargs):
        """
        bounds is for fence-specification with non-uniform coordinates
        """
        if coords is None:
            coords = OrderedDict(kwargs)
        for key, val in coords.iteritems()
            coords[key] = Coord(val)
        super(Coordinate, self).__init__(coords=coords, projection=projection)
    
    ctype = tl.CaselessStrEnum(['uniform', None, 'fence', 'post', )
    projection = tl.CUnicode
    coords = tl.Instance(OrderedDict)
    
    @tl.validate('coords')
    def _coords_validate(self, proposal):
        seen_dims = []
        for key in proposal['value']:
            self._validate_dim(key, seen_dims)
            val = proposal['value'][key]
            self._validate_val(val, key, proposal['value'].keys())
            
            
        return proposal['value']
        
    def _validate_dim(self, dim, seen_dims):
        parts = dim.replace('-', '_').split('_')
        for part in parts:
            if part not in self._valid_dims:
                raise CoordinateException("The '" + part + "' dimension of '" + key +\
                    "' is not a valid dimension " + str(self.valid_dims)
                )
            if part in seen_dims:
                raise CoordinateException("The dimensions '" + part + \
                "' cannot be repeated.")
            seen_dims.append(part)
    
    def _validate_val(self, val, dim='', dims=[]):
    # MOVE A LOT OF THIS INTO THE COORD
        if isinstance(val, tuple):
            # Regular, gridded equal value
            if len(val) != 3:
                raise CoordinateException("When specifying uniformly spaced" + \
                "coordinates, provide it in the format (start, stop, number)" +\
                "or (start, stop, step)")
            return
        # Dependent array, needs to be an xarray.DataArray
        elif isinstance(val, xr.DataArray):
            for key in val.coords: 
                    if key not in dims:
                        raise CoordinateException("Dimensions of dependent coordinate" +
                        " DatArray needs to be in " + str(dims))
        # Irregular spacing independent coordinates
        else: 
            v = np.array(val).squeeze()
            if len(v.shape) > 2:
                raise CoordinateException("Dependent coordinates need to be"
                " specified as an xarray.DataArray")
            elif np.isfinite(v):
                return # no problems
                
            raise CoordinateException("Coordinates '" + str(val) + "' for '" + dim +\
            "' not understood")
            
            
    def modify(self, copy=True, **kwargs):
        raise NotImplementedError("")
    def initialize_dataset(self, initial_value=0, dtype=np.float):
        pass
    def intersect(self, coords):
        pass