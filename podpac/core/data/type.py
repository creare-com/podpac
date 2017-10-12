from __future__ import division, unicode_literals, print_function, absolute_import

import requests
from io import BytesIO
import bs4
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


class WCS(podpac.DataSource):
    source = tl.Unicode()
    layer_name = tl.Unicode()
    version = tl.Unicode('1.0.0')
    crs = tl.Unicode('EPSG:4326')
    get_capabilities_qs = tl.Unicode('SERVICE=WCS&REQUEST=DescribeCoverage&'
                                     'VERSION={version}&COVERAGE={layer}')
    get_data_qs = tl.Unicode('SERVICE=WCS&VERSION={version}&REQUEST=GetCoverage&'
                             'FORMAT=GeoTIFF&COVERAGE={layer}&'
                             'BBOX={w},{s},{e},{n}&CRS={crs}&RESPONSE_CRS={crs}&'
                             'WIDTH={width}&HEIGHT={height}')
    
    @property
    def get_capabilities_url(self):
        return self.source + '?' + self.get_capabilities_qs.format(
            version=self.version, layer=self.layer_name)
    
    wcs_coordinates = tl.Instance(podpac.Coordinate)
    @tl.default('wcs_coordinates')
    def get_wcs_coordinates(self):
        capabilities = requests.get(self.get_capabilities_url)
        if capabilities.status_code != 200:
            raise Exception("Could not get capabilities from WCS server")
        capabilities = bs4.BeautifulSoup(capabilities.text, 'lxml')
        domain = capabilities.find('wcs:spatialdomain')
        pos = domain.find('gml:envelope').get_text().split()
        lonlat = np.array(pos, float).reshape(2, 2)
        grid_env = domain.find('gml:gridenvelope')
        low = np.array(grid_env.find('gml:low').text.split(), int)
        high = np.array(grid_env.find('gml:high').text.split(), int)
        size = high - low
        dlondlat = (lonlat[1, :] - lonlat[0, :]) / size
        bottom = lonlat[:, 1].min() + dlondlat[1] / 2
        top = lonlat[:, 1].max() - dlondlat[1] / 2
        left = lonlat[:, 1].min() + dlondlat[0] / 2
        right = lonlat[:, 1].max() - dlondlat[0] / 2        
        return podpac.Coordinate(lat=(top, bottom, size[1]),
                                 lon=(left, right, size[0]), order=['lat', 'lon'])    

    @tl.default('native_coordinates')
    def get_native_coordinates(self):
        if self.evaluated_coordinates:
            ev = self.evaluated_coordinates
            wcs_c = self.wcs_coordinates
            cs = OrderedDict()
            for c in wcs_c.dims:
                if c in ev.dims and ev[c].regularity in ['irregular', 'dependent']:
                    raise NotImplementedError
                if c in ev.dims:
                    cs[c] = [min(ev[c].coords), max(ev[c].coords), ev[c].delta]
                else:
                    cs.append(wcs_c[c])
            c = podpac.Coordinate(cs)
            return c
        else:
            return self.wcs_coordinates

    def get_data(self, coordinates, coodinates_slice):
        output = self.initialize_coord_array(coordinates)
        
        url = self.source + '?' + self.get_data_qs.format(
            version=self.version, layer=self.layer_name, 
            w=min(coordinates['lon'].area_bounds),
            e=max(coordinates['lon'].area_bounds),
            s=min(coordinates['lat'].area_bounds),
            n=max(coordinates['lat'].area_bounds),
            width=coordinates['lon'].size,
            height=coordinates['lat'].size,
            crs=self.crs
            )
        data = requests.get(url)
        if data.status_code != 200:
            raise Exception("Could not get data from WCS server")        
        io = BytesIO(bytearray(data.content))
        with rasterio.open(io) as dataset:
            output.data[:] = dataset.read()
            
        return output

# We mark this as an algorithm node for the sake of the pipeline, although
# the "algorithm" portion is not being used / is overwritten by the DataSource
class ReprojectedSource(podpac.DataSource, podpac.Algorithm):
    implicit_pipeline_evaluation = tl.Bool(False)
    source_interpolation = tl.Unicode('nearest_preview')
    source = tl.Instance(podpac.Node)
    # Specify either one of the next two
    coordinates_source = tl.Instance(podpac.Node, allow_none=True)
    reprojected_coordinates = tl.Instance(podpac.Coordinate)
    
    @tl.default('reprojected_coordinates')
    def get_reprojected_coordinates(self):
        try: 
            return self.coordinates_source.native_coordinates
        except AttributeError: 
            raise Exception("Either reprojected_coordinates or coordinates"
                            "_source must be specified")
            
    @tl.default('native_coordinates')
    def get_native_coordinates(self):
        coords = OrderedDict()
        sc = self.source.native_coordinates
        rc = self.reprojected_coordinates
        for d in sc.dims:
            if d in rc.dims:
                coords[d] = rc[d]
            else:
                coords[d] = sc[d]
        return podpac.Coordinate(coords)
    
    def get_data(self, coordinates, coordinates_slice):
        self.source.interpolation = self.source_interpolation
        return self.source.execute(coordinates, self.params)

if __name__ == '__main__':
    coord_src = podpac.Coordinate(lat=(45, 0, 16), lon=(-70., -65., 16), time=(0, 1, 2),
                                  order=['lat', 'lon', 'time'])
    coord_dst = podpac.Coordinate(lat=(50., 0., 50), lon=(-71., -66., 100),
                                  order=['lat', 'lon'])
    LON, LAT, TIME = np.meshgrid(coord_src['lon'].coordinates,
                                  coord_src['lat'].coordinates,
                                  coord_src['time'].coordinates)
    #LAT, LON = np.mgrid[0:45+coord_src['lat'].delta/2:coord_src['lat'].delta,
                              #-70:-65+coord_src['lon'].delta/2:coord_src['lon'].delta]    
    source = LAT + 0*LON + 0*TIME
    nas = NumpyArray(source=source.astype(float), 
                     native_coordinates=coord_src, interpolation='bilinear')
    #coord_pts = podpac.Coordinate(lat_lon=(coord_src.coords['lat'], coord_src.coords['lon']))
    #o3 = nas.execute(coord_pts)
    o = nas.execute(coord_dst)
    #coord_pt = podpac.Coordinate(lat=10., lon=-67.)
    #o2 = nas.execute(coord_pt)
    
    coordinates = podpac.Coordinate(lat=(45, 0, 16), lon=(-70., -65., 16),
                                    order=['lat', 'lon'])
    reprojected_coordinates = podpac.Coordinate(lat=(45, 0, 3), lon=(-70., -65., 3),
                                    order=['lat', 'lon'])    
                          'TopographicWetnessIndexComposited3090m'),
              )
    
    o = wcs.execute(coordinates)

    reprojected = ReprojectedSource(source=wcs,
                                    reprojected_coordinates=reprojected_coordinates,
                                    interpolation='bilinear')
    o2 = reprojected.execute(coordinates)
    
    print ("Done")
