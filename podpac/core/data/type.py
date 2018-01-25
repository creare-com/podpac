from __future__ import division, unicode_literals, print_function, absolute_import

import os
import re
from io import BytesIO
import tempfile
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
    try:
        import arcpy
    except:
        arcpy = None
    
try:
    import boto3
except:
    boto3 = None
    
try:
    import requests
except:
    requests = None
    try:
        import urllib3
    except:
        urllib3 = None
        
    try:
        import certifi
    except:
        certifi = None

# Not used directly, but used indirectly by bs4 so want to check if it's available
try:
    import lxml
except:
    lxml = None

# Internal dependencies
import podpac
from podpac.core import authentication

class NumpyArray(podpac.DataSource):
    source = tl.Instance(np.ndarray)

    def get_data(self, coordinates, coordinates_slice):
        s = coordinates_slice
        d = self.initialize_coord_array(coordinates, 'data', 
                                        fillval=self.source[s])
        return d

class PyDAP(podpac.DataSource):
    auth_session = tl.Instance(authentication.SessionWithHeaderRedirection,
                               allow_none=True)
    auth_class = tl.Type(authentication.SessionWithHeaderRedirection)
    username = tl.Unicode(None, allow_none=True)
    password = tl.Unicode(None, allow_none=True)
    @tl.default('auth_session')
    def _auth_session_default(self):
        if not self.username or not self.password:
            return None
        session = self.auth_class(
                username=self.username, password=self.password)
        # check url
        try:
            session.get(self.source + '.dds')
        except:
            return None
        return session
   
    dataset = tl.Instance('pydap.model.DatasetType', allow_none=True)
    @tl.default('dataset')
    def open_dataset(self, source=None):
        if source is None:
            source = self.source
        else:
            self.source = source
        
        try:
            dataset = pydap.client.open_url(source, session=self.auth_session)
        except:
            #Check Url (probably inefficient...)
            self.auth_session.get(self.source + '.dds')
            dataset = pydap.client.open_url(source, session=self.auth_session)
        
        return dataset
        

    @tl.observe('source')
    def _update_dataset(self, change):
        if change['old'] == None:
            return
        if self.dataset is not None:
            self.dataset = self.open_dataset(change['new'])
        if self.native_coordinates is not None:
            self.native_coordinates = self.get_native_coordinates()

    datakey = tl.Unicode(allow_none=False)
  
    def get_native_coordinates(self):
        raise NotImplementedError("DAP has no mechanism for creating coordinates"
                                  ", so this is left up to child class "
                                  "implementations.")


    def get_data(self, coordinates, coordinates_slice):
        data = self.dataset[self.datakey][tuple(coordinates_slice)]
        d = self.initialize_coord_array(coordinates, 'data', 
                                        fillval=data.reshape(coordinates.shape))
        return d
    
    @property
    def keys(self):
        return self.dataset.keys()

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

WCS_DEFAULT_VERSION = u'1.0.0'
WCS_DEFAULT_CRS = 'EPSG:4326'

class WCS(podpac.DataSource):
    source = tl.Unicode()
    layer_name = tl.Unicode()
    version = tl.Unicode(WCS_DEFAULT_VERSION)
    crs = tl.Unicode(WCS_DEFAULT_CRS)
    get_capabilities_qs = tl.Unicode('SERVICE=WCS&REQUEST=DescribeCoverage&'
                                     'VERSION={version}&COVERAGE={layer}')
    get_data_qs = tl.Unicode('SERVICE=WCS&VERSION={version}&REQUEST=GetCoverage&'
                             'FORMAT=GeoTIFF&COVERAGE={layer}&'
                             'BBOX={w},{s},{e},{n}&CRS={crs}&RESPONSE_CRS={crs}&'
                             'WIDTH={width}&HEIGHT={height}&TIME={time}')    

    @property
    def get_capabilities_url(self):
        return self.source + '?' + self.get_capabilities_qs.format(
            version=self.version, layer=self.layer_name)

    wcs_coordinates = tl.Instance(podpac.Coordinate)
    @tl.default('wcs_coordinates')
    def get_wcs_coordinates(self):
        if requests is not None:
            capabilities = requests.get(self.get_capabilities_url)
            if capabilities.status_code != 200:
                raise Exception("Could not get capabilities from WCS server")
            capabilities = capabilities.text
        elif urllib3 is not None:
            if certifi is not None:
                http = urllib3.PoolManager(ca_certs=certifi.where())
            else:
                http = urllib3.PoolManager()
            r = http.request('GET',self.get_capabilities_url)
            capabilities = r.data
            if r.status != 200:
                raise Exception("Could not get capabilities from WCS server")
        else:
            raise Exception("Do not have a URL request library to get WCS data.")
        if lxml is not None: # could skip using lxml and always use html.parser instead, which seems to work but lxml might be faster
            capabilities = bs4.BeautifulSoup(capabilities, 'lxml')
        else:
            capabilities = bs4.BeautifulSoup(capabilities, 'html.parser')
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

        timedomain = capabilities.find("wcs:temporaldomain")
        if timedomain is None:
            return podpac.Coordinate(lat=(top, bottom, size[1]),
                                         lon=(left, right, size[0]), order=['lat', 'lon'])    
        
        date_re = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}')
        times = str(timedomain).replace('<gml:timeposition>', '').replace('</gml:timeposition>', '').split('\n')
        times = np.array([t for t in times if date_re.match(t)], np.datetime64)
        
        return podpac.Coordinate(time=times,
                                 lat=(top, bottom, size[1]),
                                 lon=(left, right, size[0]),                          
                                 order=['time', 'lat', 'lon'])            
        

    @property
    def native_coordinates(self):
        if self.evaluated_coordinates:
            ev = self.evaluated_coordinates
            wcs_c = self.wcs_coordinates
            cs = OrderedDict()
            for c in wcs_c.dims:
                if c in ev.dims and ev[c].regularity in ['irregular', 'dependent']:
                    # This is rough, we have to use a regular grid for WCS calls, 
                    # Otherwise we have to do multiple WCS calls... 
                    # TODO: generalize/fix this
                    cs[c] = [min(ev[c].coords),
                             max(ev[c].coords), ev[c].delta]
                elif c in ev.dims and ev[c].regularity == 'regular':
                    cs[c] = [min(ev[c].coords[:2]),
                             max(ev[c].coords[:2]), ev[c].delta]
                elif c in ev.dims and ev[c].regularity == 'single':
                    cs[c] = ev[c].coords
                else:
                    cs.append(wcs_c[c])
            c = podpac.Coordinate(cs)
            return c
        else:
            return self.wcs_coordinates

    def get_data(self, coordinates, coodinates_slice):
        output = self.initialize_coord_array(coordinates)
        dotime = 'time' in self.wcs_coordinates.dims
        if 'time' in coordinates.dims and dotime:
            sd = np.timedelta64(0, 's')
            times = [str(t+sd) for t in coordinates['time'].coordinates]
        else:
            times = ['']
            
        if len(times) > 1:
            for i, time in enumerate(times): 
                url = self.source + '?' + self.get_data_qs.format(
                    version=self.version, layer=self.layer_name, 
                    w=min(coordinates['lon'].area_bounds),
                    e=max(coordinates['lon'].area_bounds),
                    s=min(coordinates['lat'].area_bounds),
                    n=max(coordinates['lat'].area_bounds),
                    width=coordinates['lon'].size,
                    height=coordinates['lat'].size,
                    time=time,
                    crs=self.crs
                )
                if not dotime:
                    url = url.replace('&TIME=', '')
                if requests is not None:
                    data = requests.get(url)
                    if data.status_code != 200:
                        raise Exception("Could not get data from WCS server")        
                    io = BytesIO(bytearray(data.content))
                    content = data.content
                elif urllib3 is not None:
                    if certifi is not None:
                        http = urllib3.PoolManager(ca_certs=certifi.where())
                    else:
                        http = urllib3.PoolManager()
                    r = http.request('GET',url)
                    if r.status != 200:
                        raise Exception("Could not get capabilities from WCS server") 
                    content = r.data
                    io = BytesIO(bytearray(r.data))
                else:
                    raise Exception("Do not have a URL request library to get WCS data.")
                
                if rasterio is not None:
                    try: # This works with rasterio v1.0a8 or greater, but not on python 2
                        with rasterio.open(io) as dataset:
                            output.data[i, ...] = dataset.read()
                    except Exception as e: # Probably python 2
                        print (e)
                        tmppath = os.path.join(
                            self.cache_dir, 'wcs_temp.tiff')                       
                        if not os.path.exists(os.path.split(tmppath)[0]):
                            os.makedirs(os.path.split(tmppath)[0]) 
                        open(tmppath,'wb').write(content)
                        with rasterio.open(tmppath) as dataset:
                            output.data[i, ...] = dataset.read()
                        os.remove(tmppath) # Clean up
                elif arcpy is not None:
                    # Writing the data to a temporary tiff and reading it from there is hacky
                    # However reading directly from r.data or io doesn't work
                    # Should improve in the future
                    open('temp.tiff','wb').write(r.data)
                    output.data[i, ...] = arcpy.RasterToNumPyArray('temp.tiff')
        else:
            time = times[0]
            
            url = self.source + '?' + self.get_data_qs.format(
                version=self.version, layer=self.layer_name, 
                w=min(coordinates['lon'].area_bounds),
                e=max(coordinates['lon'].area_bounds),
                s=min(coordinates['lat'].area_bounds),
                n=max(coordinates['lat'].area_bounds),
                width=coordinates['lon'].size,
                height=coordinates['lat'].size,
                time=time,
                crs=self.crs
            )
            if not dotime:
                url = url.replace('&TIME=', '')
            if requests is not None:
                data = requests.get(url)
                if data.status_code != 200:
                    raise Exception("Could not get data from WCS server")        
                io = BytesIO(bytearray(data.content))
                content = data.content
            elif urllib3 is not None:
                if certifi is not None:
                    http = urllib3.PoolManager(ca_certs=certifi.where())
                else:
                    http = urllib3.PoolManager()
                r = http.request('GET',url)
                if r.status != 200:
                    raise Exception("Could not get capabilities from WCS server") 
                content = r.data
                io = BytesIO(bytearray(r.data))
            else:
                raise Exception("Do not have a URL request library to get WCS data.")  
            
            if rasterio is not None:                          
                try: # This works with rasterio v1.0a8 or greater, but not on python 2
                    with rasterio.open(io) as dataset:
                        if dotime:
                            output.data[0, ...] = dataset.read()
                        else:
                            output.data[:] = dataset.read()
                except Exception as e: # Probably python 2
                    print (e)
                    tmppath = os.path.join(
                        self.cache_dir, 'wcs_temp.tiff')                        
                    if not os.path.exists(os.path.split(tmppath)[0]):
                        os.makedirs(os.path.split(tmppath)[0]) 
                    open(tmppath,'wb').write(content)
                    with rasterio.open(tmppath) as dataset:
                        output.data[:] = dataset.read()
                    os.remove(tmppath) # Clean up
            elif arcpy is not None:
                # Writing the data to a temporary tiff and reading it from there is hacky
                # However reading directly from r.data or io doesn't work
                # Should improve in the future
                open('temp.tiff','wb').write(r.data)
                output.data[:] = arcpy.RasterToNumPyArray('temp.tiff')            

        if not coordinates['lat'].is_max_to_min:
            if dotime:
                output.data[:] = output.data[:, ::-1, :]
            else:
                output.data[:] = output.data[::-1, :]

        return output

    @property
    def base_ref(self):
        return self.layer_name.rsplit('.', 1)[1]

    @property
    def definition(self):
        d = super(WCS, self).definition
        d['attrs'] = d.get('attrs', OrderedDict())
        d['attrs']['layer_name'] = self.layer_name
        if self.version != WCS_DEFAULT_VERSION:
            d['attrs']['version'] = self.version
        if self.crs != WCS_DEFAULT_CRS:
            d['attrs']['crs'] = self.crs
        return d

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

    @property
    def base_ref(self):
        return '%s_reprojected' % self.source.base_ref

    @property
    def definition(self):
        d = podpac.Algorithm.definition.fget(self)
        d['attrs'] = OrderedDict()
        if self.interpolation:
            d['attrs']['interpolation'] = self.interpolation
        if self.coordinates_source is None:
            # TODO serialize reprojected_coordinates
            raise NotImplementedError
        return d

class S3Source(podpac.DataSource):
    source = tl.Unicode()
    node = tl.Instance(podpac.Node)  
    node_class = tl.Type(podpac.DataSource)  # A class 
    node_kwargs = tl.Dict(default_value={})
    s3_bucket = tl.Unicode()
    s3_data = tl.Any()
    temp_file_cleanup = tl.List()
    return_type = tl.Enum(['file_handle', 'path'], default_value='path')
    
    @tl.default('node')
    def node_default(self):
        if 'source' in self.node_kwargs:
            raise Exception("'source' present in node_kwargs for S3Source")
        return self.node_class(source=self.s3_data, **self.node_kwargs)

    @tl.default('s3_bucket')
    def s3_bucket_default(self):
        return podpac.settings.S3_BUCKET_NAME

    @tl.default('s3_data')
    def s3_data_default(self):
        s3 = boto3.resource('s3').Bucket(self.s3_bucket)
        if self.return_type == 'file_handle':
            # download into memory
            io = BytesIO()
            s3.download_fileobj(self.source, io)
            io.seek(0)
            return io
        elif self.return_type == 'path':
            # Download the file to cache directory
            #tmppath = os.path.join(tempfile.gettempdir(),
                                   #self.source.replace('\\', '').replace(':','')\
                                   #.replace('/', ''))
            tmppath = os.path.join(
                self.cache_dir,
                self.source.replace('\\', '').replace(':','').replace('/', ''))
            
            rootpath = os.path.split(tmppath)[0]
            if not os.path.exists(rootpath):
                os.makedirs(rootpath)
            #i = 0
            #while os.path.exists(tmppath):
                #tmppath = os.path.join(tempfile.gettempdir(),
                                       #self.source + '.%d' % i)
            if not os.path.exists(tmppath):
                s3.download_file(self.source, tmppath)
            #self.temp_file_cleanup.append(tmppath)
            return tmppath

    def get_data(self, coordinates, coordinates_slice):
        self.no_data_vals = getattr(self.node, 'no_data_vals', [])
        return self.node.get_data(coordinates, coordinates_slice)

    @property
    def native_coordinates(self):
        return self.node.native_coordinates

    def __del__(self):
        if hasattr(super(S3Source), '__del__'):
            super(S3Source).__del__(self)
        for f in self.temp_file_cleanup:
            os.remove(f)

if __name__ == '__main__':
    #from podpac.core.data.type import S3Source
    #import podpac

    source = r'SMAPSentinel/SMAP_L2_SM_SP_1AIWDV_20170801T000000_20170731T114719_094E21N_T15110_002.h5'
    s3 = S3Source(source=source)
    
    s3.s3_data
    
    #coord_src = podpac.Coordinate(lat=(45, 0, 16), lon=(-70., -65., 16), time=(0, 1, 2),
                                    #order=['lat', 'lon', 'time'])
    #coord_dst = podpac.Coordinate(lat=(50., 0., 50), lon=(-71., -66., 100),
                                    #order=['lat', 'lon'])
    #LON, LAT, TIME = np.meshgrid(coord_src['lon'].coordinates,
                                    #coord_src['lat'].coordinates,
                                    #coord_src['time'].coordinates)
    ##LAT, LON = np.mgrid[0:45+coord_src['lat'].delta/2:coord_src['lat'].delta,
                                ##-70:-65+coord_src['lon'].delta/2:coord_src['lon'].delta]    
    #source = LAT + 0*LON + 0*TIME
    #nas = NumpyArray(source=source.astype(float), 
                        #native_coordinates=coord_src, interpolation='bilinear')
    ##coord_pts = podpac.Coordinate(lat_lon=(coord_src.coords['lat'], coord_src.coords['lon']))
    ##o3 = nas.execute(coord_pts)
    #o = nas.execute(coord_dst)
    ##coord_pt = podpac.Coordinate(lat=10., lon=-67.)
    ##o2 = nas.execute(coord_pt)
    from podpac.datalib.smap import SMAPSentinelSource
    s3.node_class = SMAPSentinelSource

    #coordinates = podpac.Coordinate(lat=(45, 0, 16), lon=(-70., -65., 16),
                                    #order=['lat', 'lon'])
    coordinates = podpac.Coordinate(lat=(39.3, 39., 64), lon=(-77.0, -76.7, 64), time='2017-09-03T12:00:00', 
                                    order=['lat', 'lon', 'time'])    
    reprojected_coordinates = podpac.Coordinate(lat=(45, 0, 3), lon=(-70., -65., 3),
                                                order=['lat', 'lon'])    
                          'TopographicWetnessIndexComposited3090m'),
              )

    o = wcs.execute(coordinates)
    reprojected = ReprojectedSource(source=wcs,
                                    reprojected_coordinates=reprojected_coordinates,
                                    interpolation='bilinear')

    from podpac.datalib.smap import SMAP
    smap = SMAP(product='SPL4SMAU.003')
    reprojected = ReprojectedSource(source=wcs,
                                    coordinates_source=smap,
                                    interpolation='nearest')    
    o2 = reprojected.execute(coordinates)

    coordinates_zoom = podpac.Coordinate(lat=(24.8, 30.6, 64), lon=(-85.0, -77.5, 64), time='2017-08-08T12:00:00', 
                                         order=['lat', 'lon', 'time'])
    o3 = wcs.execute(coordinates_zoom)


    print ("Done")
    
    # Rename files in s3 bucket
    s3 = boto3.resource('s3').Bucket(self.s3_bucket)
    s3.Bucket(name='podpac-s3')
    obs = list(s3.objects.all())
    obs2 = [o for o in obs if 'SMAP_L2_SM_SP' in o.key]
    
    rootpath = obs2[0].key.split('/')[0] + '/'
    for o in obs2:
        newkey = rootpath + os.path.split(o.key)[1]
        s3.Object(newkey).copy_from(CopySource=self.s3_bucket + '/' + o.key)
        
    obs3 = list(s3.objects.all())
    obsD = [o for o in obs3 if 'ASOwusu' in o.key]
    for o in obsD:
        o.delete()    
