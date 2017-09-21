from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
import numpy as np
import xarray as xp
import traitlets as tl
from pint import UnitRegistry
ureg = UnitRegistry()

# Optional dependencies
try:
    import pydap.client
except:
    pydap = None
    
try:
    import rasterio
except:
    rasterio = None
# Internal dependencies
import podpac

class NumpyArray(podpac.DataSource):
    source = tl.Instance(np.ndarray)
    
    def get_data(self, coordinates, coordinates_slice):
        s = coordinates_slice
        d = self.initialize_coord_array(coordinates, 'data', 
                                        fillval=self.source[s])
        return d

class PyDAP(podpac.DataSource):
    dataset = tl.Instance('pydap.model.DatasetType', allow_none=True)
    @tl.default('dataset')
    def open_dataset(self, source=None):
        if source is None:
            source = self.source
        else:
            self.source = source
        return pydap.client.open_url(source)
    
    @tl.observe('source')
    def _update_dataset(self, change):
        if change['old'] == None:
            return
        if self.dataset is not None:
            self.dataset = self.open_dataset(change['new'])
        if self.native_coordinates is not None:
            self.native_coordinates = self.get_native_coordinates()
    
    datakey = tl.Unicode(allow_none=False)
    native_coordinates = tl.Instance('podpac.core.coordinate.Coordinate',
                                      allow_none=False)    
    @tl.default('native_coordinates')
    def get_native_coordinates(self):
        raise NotImplementedError("DAP has no mechanism for creating coordinates"
                                  ", so this is left up to child class "
                                  "implementations.")
    
    
    def get_data(self, coordinates, coordinates_slice):
        data = self.dataset[self.datakey][tuple(coordinates_slice)]
        d = self.initialize_coord_array(coordinates, 'data', 
                                        fillval=data.reshape(coordinates.shape))
        return d
    
class RasterioSource(podpac.DataSource):
    source = tl.Unicode(allow_none=False)
    
    
    dataset = tl.Instance('rasterio._io.RasterReader',
                          allow_none=True)
    @tl.default('dataset')
    def open_dataset(self, source=None):
        if source is None:
            source = self.source
        else:
            self.source = source
        return rasterio.open(source)
    
    @tl.observe('source')
    def _update_dataset(self, change):
        if self.dataset is not None:
            self.dataset = self.open_dataset(change['new'])
        self.native_coordinates = self.get_native_coordinates()
    
    native_coordinates = tl.Instance('podpac.core.coordinate.Coordinate',
                                      allow_none=False)    
    @tl.default('native_coordinates')
    def get_native_coordinates(self):
        dlon, dlat = self.dataset.res
        left, bottom, right, top = self.dataset.bounds
        if self.dataset.transform[1] != 0.0 or\
                self.dataset.transform[3] != 0.0:
            raise NotImplementedError("Have not implemented rotated coords")
        return podpac.Coordinate(lat=(top, bottom, dlat),
                                 lon=(left, right, dlon))
    
    def get_data(self, coordinates, coodinates_slice):
        return 

        
if __name__ == '__main__':
    coord_src = podpac.Coordinate(lat=(45, 0, 16), lon=(-70., -65., 16), time=(0, 1, 2))
    coord_dst = podpac.Coordinate(lat=(50., 0., 50), lon=(-71., -66., 100))
    LON, LAT, TIME = np.meshgrid(coord_src['lon'].coordinates,
                                  coord_src['lat'].coordinates,
                                  coord_src['time'].coordinates)
    #LAT, LON = np.mgrid[0:45+coord_src['lat'].delta/2:coord_src['lat'].delta,
                              #-70:-65+coord_src['lon'].delta/2:coord_src['lon'].delta]    
    source = LAT + 0*LON + 0*TIME
    nas = NumpyArray(source=source.astype(float), 
                     native_coordinates=coord_src, interpolation='bilinear')
    coord_pts = podpac.Coordinate(lat_lon=(coord_src.coords['lat'], coord_src.coords['lon']))
    o3 = nas.execute(coord_pts)
    o = nas.execute(coord_dst)
    coord_pt = podpac.Coordinate(lat=10., lon=-67.)
    o2 = nas.execute(coord_pt)
    print ("Done")
