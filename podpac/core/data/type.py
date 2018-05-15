"""
Type Summary

Attributes
----------
ureg : TYPE
    Description
WCS_DEFAULT_CRS : str
    Description
WCS_DEFAULT_VERSION : str
    Description
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import re
from io import BytesIO
from collections import OrderedDict, defaultdict

import bs4
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
        from arcpy import RasterToNumPyArray
    except:
        RasterToNumPyArray = None
    
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
from podpac.core.utils import cached_property, clear_cache


class NumpyArray(podpac.DataSource):
    """Summary
    
    Attributes
    ----------
    source : TYPE
        Description
    """
    
    source = tl.Instance(np.ndarray)

    def get_data(self, coordinates, coordinates_slice):
        """Summary
        
        Parameters
        ----------
        coordinates : TYPE
            Description
        coordinates_slice : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        s = coordinates_slice
        d = self.initialize_coord_array(coordinates, 'data',
                                        fillval=self.source[s])
        return d

class PyDAP(podpac.DataSource):
    """Summary
    
    Attributes
    ----------
    auth_class : TYPE
        Description
    auth_session : TYPE
        Description
    datakey : TYPE
        Description
    dataset : TYPE
        Description
    native_coordinates : TYPE
        Description
    password : TYPE
        Description
    source : TYPE
        Description
    username : TYPE
        Description
    """
    
    auth_session = tl.Instance(authentication.SessionWithHeaderRedirection,
                               allow_none=True)
    auth_class = tl.Type(authentication.SessionWithHeaderRedirection)
    username = tl.Unicode(None, allow_none=True)
    password = tl.Unicode(None, allow_none=True)

    @tl.default('auth_session')
    def _auth_session_default(self):
        if not self.username or not self.password:
            return None
        session = self.auth_class(username=self.username, password=self.password)

        # check url
        try:
            session.get(self.source + '.dds')
        except:
            return None
        return session
   
    dataset = tl.Instance('pydap.model.DatasetType', allow_none=True)

    @tl.default('dataset')
    def open_dataset(self, source=None):
        """Summary
        
        Parameters
        ----------
        source : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
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
        """Summary
        
        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError("DAP has no mechanism for creating coordinates"
                                  ", so this is left up to child class "
                                  "implementations.")


    def get_data(self, coordinates, coordinates_slice):
        """Summary
        
        Parameters
        ----------
        coordinates : TYPE
            Description
        coordinates_slice : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        data = self.dataset[self.datakey][tuple(coordinates_slice)]
        d = self.initialize_coord_array(coordinates, 'data',
                                        fillval=data.reshape(coordinates.shape))
        return d
    
    @property
    def keys(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self.dataset.keys()

class RasterioSource(podpac.DataSource):
    """Summary
    
    Attributes
    ----------
    band : TYPE
        Description
    dataset : TYPE
        Description
    native_coordinates : TYPE
        Description
    source : TYPE
        Description
    """
    
    source = tl.Unicode(allow_none=False)
    dataset = tl.Any(allow_none=True)
    band = tl.CInt(1)
    
    @tl.default('dataset')
    def open_dataset(self, source=None):
        """Summary
        
        Parameters
        ----------
        source : None, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        if source is None:
            source = self.source
        else:
            self.source = source
        return rasterio.open(source)
    
    def close_dataset(self):
        """Summary
        """
        self.dataset.close()

    @tl.observe('source')
    def _update_dataset(self, change):
        if self.dataset is not None:
            self.dataset = self.open_dataset(change['new'])
        self.native_coordinates = self.get_native_coordinates()

    def get_native_coordinates(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        NotImplementedError
            Description
        """
        dlon = self.dataset.width
        dlat = self.dataset.height
        if hasattr(self.dataset, 'affine'):
            affine = self.dataset.affine
        else:
            affine = self.dataset.transform
        left, bottom, right, top = self.dataset.bounds
        if affine[1] != 0.0 or\
           affine[3] != 0.0:
            raise NotImplementedError("Have not implemented rotated coords")

        return podpac.Coordinate(lat=(top, bottom, dlat),
                                 lon=(left, right, dlon),
                                 order=['lat', 'lon'])

    def get_data(self, coordinates, coodinates_slice):
        """Summary
        
        Parameters
        ----------
        coordinates : TYPE
            Description
        coodinates_slice : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        data = self.initialize_coord_array(coordinates)
        
        data.data.ravel()[:] = self.dataset.read(
            self.band, window=((slc[0].start, slc[0].stop),
                               (slc[1].start, slc[1].stop)),
            out_shape=tuple(coordinates.shape)
            ).ravel()
            
        return data
    
    @cached_property
    def band_count(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self.dataset.count
    
    @cached_property
    def band_descriptions(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        bands = OrderedDict()
        for i in range(self.dataset.count):
            bands[i] = self.dataset.tags(i + 1)
        return bands

    @cached_property
    def band_keys(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        keys = {}
        for i in range(self.band_count):
            for k in self.band_descriptions[i].keys():
                keys[k] = None
        keys = keys.keys()
        band_keys = defaultdict(lambda: [])
        for k in keys:
            for i in range(self.band_count):
                band_keys[k].append(self.band_descriptions[i].get(k, None))
        return band_keys
    
    @tl.observe('source')
    def _clear_band_description(self, change):
        clear_cache(self, change, ['band_descriptions', 'band_count',
                                   'band_keys'])

    def get_band_numbers(self, key, value):
        """Summary
        
        Parameters
        ----------
        key : TYPE
            Description
        value : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        if not hasattr(key, '__iter__') and not hasattr(value, '__iter__'):
            key = [key]
            value = [value]

        match = np.ones(self.band_count, bool)
        for k, v in zip(key, value):
            match = match & (np.array(self.band_keys[k]) == v)
        matches = np.where(match)[0] + 1

        return matches


WCS_DEFAULT_VERSION = u'1.0.0'
WCS_DEFAULT_CRS = 'EPSG:4326'

class WCS(podpac.DataSource):
    """Summary
    
    Attributes
    ----------
    crs : TYPE
        Description
    get_capabilities_qs : TYPE
        Description
    get_data_qs : TYPE
        Description
    layer_name : TYPE
        Description
    source : TYPE
        Description
    version : TYPE
        Description
    wcs_coordinates : TYPE
        Description
    """
    
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
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self.source + '?' + self.get_capabilities_qs.format(
            version=self.version, layer=self.layer_name)

    wcs_coordinates = tl.Instance(podpac.Coordinate)
    @tl.default('wcs_coordinates')
    def get_wcs_coordinates(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        Exception
            Description
        """
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

            r = http.request('GET', self.get_capabilities_url)
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
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        if self.evaluated_coordinates:
            ev = self.evaluated_coordinates
            wcs_c = self.wcs_coordinates
            cs = OrderedDict()
            for c in wcs_c.dims:
                if c in ev.dims and ev[c].size == 1:
                    cs[c] = ev[c].coords
                elif c in ev.dims and not isinstance(ev[c], podpac.UniformCoord):
                    # This is rough, we have to use a regular grid for WCS calls,
                    # Otherwise we have to do multiple WCS calls...
                    # TODO: generalize/fix this
                    cs[c] = (min(ev[c].coords),
                             max(ev[c].coords), abs(ev[c].delta))
                elif c in ev.dims and isinstance(ev[c], podpac.UniformCoord):
                    cs[c] = (min(ev[c].coords[:2]),
                             max(ev[c].coords[:2]), abs(ev[c].delta))
                else:
                    cs.append(wcs_c[c])
            c = podpac.Coordinate(cs)
            return c
        else:
            return self.wcs_coordinates

    def get_data(self, coordinates, coodinates_slice):
        """Summary
        
        Parameters
        ----------
        coordinates : TYPE
            Description
        coodinates_slice : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        Exception
            Description
        """
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
                    r = http.request('GET', url)
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
                        print(e)
                        tmppath = os.path.join(self.cache_dir, 'wcs_temp.tiff')
                        
                        if not os.path.exists(os.path.split(tmppath)[0]):
                            os.makedirs(os.path.split(tmppath)[0])
                        
                        # TODO: close tmppath? os does this on remove?
                        open(tmppath, 'wb').write(content)
                        
                        with rasterio.open(tmppath) as dataset:
                            output.data[i, ...] = dataset.read()

                        os.remove(tmppath) # Clean up

                elif RasterToNumPyArray is not None:
                    # Writing the data to a temporary tiff and reading it from there is hacky
                    # However reading directly from r.data or io doesn't work
                    # Should improve in the future
                    open('temp.tiff', 'wb').write(r.data)
                    output.data[i, ...] = RasterToNumPyArray('temp.tiff')
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
                r = http.request('GET', url)
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
                    print(e)
                    tmppath = os.path.join(
                        self.cache_dir, 'wcs_temp.tiff')
                    if not os.path.exists(os.path.split(tmppath)[0]):
                        os.makedirs(os.path.split(tmppath)[0])
                    open(tmppath, 'wb').write(content)
                    with rasterio.open(tmppath) as dataset:
                        output.data[:] = dataset.read()
                    os.remove(tmppath) # Clean up
            elif RasterToNumPyArray is not None:
                # Writing the data to a temporary tiff and reading it from there is hacky
                # However reading directly from r.data or io doesn't work
                # Should improve in the future
                open('temp.tiff', 'wb').write(r.data)
                output.data[:] = RasterToNumPyArray('temp.tiff')
            else:
                raise Exception('Rasterio or Arcpy not available to read WCS feed.')
        if not coordinates['lat'].is_descending:
            if dotime:
                output.data[:] = output.data[:, ::-1, :]
            else:
                output.data[:] = output.data[::-1, :]

        return output

    @property
    def base_ref(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return self.layer_name.rsplit('.', 1)[1]

    @property
    def definition(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
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
    """Summary
    
    Attributes
    ----------
    coordinates_source : TYPE
        Description
    implicit_pipeline_evaluation : TYPE
        Description
    reprojected_coordinates : TYPE
        Description
    source : TYPE
        Description
    source_interpolation : TYPE
        Description
    """
    
    implicit_pipeline_evaluation = tl.Bool(False)
    source_interpolation = tl.Unicode('nearest_preview')
    source = tl.Instance(podpac.Node)
    # Specify either one of the next two
    coordinates_source = tl.Instance(podpac.Node, allow_none=True)
    reprojected_coordinates = tl.Instance(podpac.Coordinate)

    @tl.default('reprojected_coordinates')
    def get_reprojected_coordinates(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        Exception
            Description
        """
        try:
            return self.coordinates_source.native_coordinates
        except AttributeError:
            raise Exception("Either reprojected_coordinates or coordinates"
                            "_source must be specified")

    def get_native_coordinates(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        coords = OrderedDict()
        if isinstance(self.source, podpac.DataSource):
            sc = self.source.native_coordinates
        else: # Otherwise we cannot guarantee that native_coordinates exist
            sc = self.reprojected_coordinates
        rc = self.reprojected_coordinates
        for d in sc.dims:
            if d in rc.dims:
                coords[d] = rc.stack_dict()[d]
            else:
                coords[d] = sc.stack_dict()[d]
        return podpac.Coordinate(coords)

    def get_data(self, coordinates, coordinates_slice):
        """Summary
        
        Parameters
        ----------
        coordinates : TYPE
            Description
        coordinates_slice : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        self.source.interpolation = self.source_interpolation
        data = self.source.execute(coordinates, self.params)
        
        # The following is needed in case the source is an algorithm
        # or compositor node that doesn't have all the dimensions of
        # the reprojected coordinates
        # TODO: What if data has coordinates that reprojected_coordinates
        #       doesn't have
        keep_dims = list(data.coords.keys())
        drop_dims = [d for d in coordinates.dims if d not in keep_dims]
        coordinates.drop_dims(*drop_dims)
        return data

    @property
    def base_ref(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return '{}_reprojected'.format(self.source.base_ref)

    @property
    def definition(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        NotImplementedError
            Description
        """
        d = podpac.Algorithm.definition.fget(self)
        d['attrs'] = OrderedDict()
        if self.interpolation:
            d['attrs']['interpolation'] = self.interpolation
        if self.coordinates_source is None:
            # TODO serialize reprojected_coordinates
            raise NotImplementedError
        return d

class S3Source(podpac.DataSource):
    """Summary
    
    Attributes
    ----------
    no_data_vals : TYPE
        Description
    node : TYPE
        Description
    node_class : TYPE
        Description
    node_kwargs : TYPE
        Description
    return_type : TYPE
        Description
    s3_bucket : TYPE
        Description
    s3_data : TYPE
        Description
    source : TYPE
        Description
    temp_file_cleanup : TYPE
        Description
    """
    
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
        """Summary
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        Exception
            Description
        """
        if 'source' in self.node_kwargs:
            raise Exception("'source' present in node_kwargs for S3Source")
        return self.node_class(source=self.s3_data, **self.node_kwargs)

    @tl.default('s3_bucket')
    def s3_bucket_default(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
        return podpac.settings.S3_BUCKET_NAME

    @tl.default('s3_data')
    def s3_data_default(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
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
                self.source.replace('\\', '').replace(':', '').replace('/', ''))
            
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
        """Summary
        
        Parameters
        ----------
        coordinates : TYPE
            Description
        coordinates_slice : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        self.no_data_vals = getattr(self.node, 'no_data_vals', [])
        return self.node.get_data(coordinates, coordinates_slice)

    @property
    def native_coordinates(self):
        """Summary
        
        Returns
        -------
        TYPE
            Description
        """
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
                                                order=['lat', 'lon']),
    #                                           'TopographicWetnessIndexComposited3090m'),
    #          )

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
