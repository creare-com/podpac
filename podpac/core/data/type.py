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
        return pydap.client.open_url(source)
    
    @tl.observe('source')
    def _update_dataset(self, change):
        if self.dataset is not None:
            self.dataset = self.open_dataset(change['new'])
        self.native_coordinates = self.set_native_coordinates()
    
    datakey = tl.Unicode(allow_none=False)
    native_coordinates = tl.Instance('podpac.core.coordinate.Coordinate',
                                      allow_none=False)    
    @tl.default('native_coordinates')
    def set_native_coordinates(self):
        raise NotImplementedError("DAP has no mechanism for creating coordinates"
                                  ", so this is left up to child class "
                                  "implementations.")
    
    
    def get_data(self, coordinates, coordinates_slice):
        data = self.dataset[self.datakey][tuple(coordinates_slice)]
        d = self.initialize_coord_array(coordinates, 'data', 
                                        fillval=data.reshape(coordinates.shape))
        return d
        
if __name__ == '__main__':
    coord_src = podpac.Coordinate(lat=(45, 0, 15), lon=(-70., -65., 15), time=(0, 1, 2))
    coord_dst = podpac.Coordinate(lat=(50., 0., 50), lon=(-71., -66., 100))
    LAT, LON, TIME = np.mgrid[0:45+coord_src['lat'].delta/2:coord_src['lat'].delta,
                            -70:-65+coord_src['lon'].delta/2:coord_src['lon'].delta,
                            0:2:1]
    #LAT, LON = np.mgrid[0:45+coord_src['lat'].delta/2:coord_src['lat'].delta,
                              #-70:-65+coord_src['lon'].delta/2:coord_src['lon'].delta]    
    source = LAT[::-1, ...] + 0*LON + 0*TIME
    nas = NumpyArray(source=source, 
                     native_coordinates=coord_src, interpolation='nearest')
    o = nas.execute(coord_dst)
    coord_pt = podpac.Coordinate(lat=10., lon=-67.)
    o2 = nas.execute(coord_pt)
    print ("Done")