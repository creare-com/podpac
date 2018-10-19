"""
Type Summary

Attributes
----------
WCS_DEFAULT_CRS : str
    Description
WCS_DEFAULT_VERSION : str
    Description
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import warnings
import os
import re
from io import BytesIO
from collections import OrderedDict, defaultdict
from six import string_types

import bs4
import numpy as np
import traitlets as tl

# Optional dependencies
try:
    import pydap.client
except ImportError:
    pydap = None

try:
    import rasterio
except ImportError:
    rasterio = None

    try:
        from arcpy import RasterToNumPyArray
    except ImportError:
        RasterToNumPyArray = None
    
try:
    import boto3
except ImportError:
    boto3 = None
    
try:
    import requests
except ImportError:
    requests = None
    try:
        import urllib3
    except ImportError:
        urllib3 = None
        
    try:
        import certifi
    except ImportError:
        certifi = None

# Not used directly, but used indirectly by bs4 so want to check if it's available
try:
    import lxml
except ImportError:
    lxml = None

# Internal dependencies
from podpac import settings
from podpac.core import authentication
from podpac.core.node import Node
from podpac.core.utils import cached_property, clear_cache, common_doc
from podpac.core.data.datasource import COMMON_DATA_DOC, DataSource
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, ArrayCoordinates1d
from podpac.core.algorithm.algorithm import Algorithm

class Array(DataSource):
    """Create a DataSource from an array
    
    Attributes
    ----------
    source : np.ndarray
        Numpy array containing the source data
        
    Notes
    ------
    `native_coordinates` need to supplied by the user when instantiating this node.
    """
    
    source = tl.Instance(np.ndarray)
    
    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        s = coordinates_index
        d = self.create_output_array(coordinates, data=self.source[s])
        return d


class NumpyArray(Array):
    """Create a DataSource from a numpy array.

    .. deprecated:: 0.2.0
          `NumpyArray` will be removed in podpac 0.2.0, it is replaced by `Array`.
    """

    def init(self):
        warnings.warn('NumpyArray been renamed Array. ' +
                      'Backwards compatibility will be removed in future releases', DeprecationWarning)

@common_doc(COMMON_DATA_DOC)
class PyDAP(DataSource):
    """Create a DataSource from an OpenDAP server feed.
    
    Attributes
    ----------
    auth_class : authentication.SessionWithHeaderRedirection
        A request.Session-derived class that has header redirection. This is used to authenticate using an EarthData
        login. When username and password are provided, an auth_session is created using this class.
    auth_session : authentication.SessionWithHeaderRedirection
        Instance of the auth_class. This is created if username and password is supplied, but this object can also be
        supplied directly
    datakey : str
        Pydap 'key' for the data to be retrieved from the server. Datasource may have multiple keys, so this key
        determines which variable is returned from the source.
    dataset : pydap.model.DatasetType
        The open pydap dataset. This is provided for troubleshooting.
    native_coordinates : Coordinates
        {native_coordinates}
    password : str, optional
        Password used for authenticating against OpenDAP server. WARNING: this is stored as plain-text, provide
        auth_session instead if you have security concerns.
    source : str
        URL of the OpenDAP server.
    username : str, optional
        Username used for authenticating against OpenDAP server. WARNING: this is stored as plain-text, provide
        auth_session instead if you have security concerns.
    """
    
    # required inputs
    source = tl.Unicode(allow_none=False, default_value='').tag(attr=True)
    datakey = tl.Unicode(allow_none=False).tag(attr=True)

    # optional inputs and later defined traits
    auth_session = tl.Instance(authentication.SessionWithHeaderRedirection,
                               allow_none=True)
    auth_class = tl.Type(authentication.SessionWithHeaderRedirection)
    username = tl.Unicode(None, allow_none=True)
    password = tl.Unicode(None, allow_none=True)
    dataset = tl.Instance('pydap.model.DatasetType', allow_none=False)

    @tl.default('auth_session')
    def _auth_session_default(self):
        
        # requires username and password
        if not self.username or not self.password:
            return None

        # requires auth_class
        # TODO: default auth_class?
        if not self.auth_class:
            return None

        # instantiate and check utl
        try:
            session = self.auth_class(username=self.username, password=self.password)
            session.get(self.source + '.dds')
        except:
            # TODO: catch a 403 error
            return None


        return session
   

    @tl.default('dataset')
    def _open_dataset(self, source=None):
        """Summary
        
        Parameters
        ----------
        source : str, optional
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        # TODO: is source ever None?
        # TODO: enforce string source
        if source is None:
            source = self.source
        else:
            self.source = source
        
        # auth session
        # if self.auth_session:
        try:
            dataset = pydap.client.open_url(source, session=self.auth_session)
        except Exception:
            # TODO handle a 403 error
            # TODO: Check Url (probably inefficient...)
            try:
                self.auth_session.get(self.source + '.dds')
                dataset = pydap.client.open_url(source, session=self.auth_session)
            except Exception:
                # TODO: handle 403 error
                print ("Warning, dataset could not be opened. Check login credentials.")
                dataset = None

        return dataset
        

    @tl.observe('source')
    def _update_dataset(self, change=None):
        if change is None:
            return

        if change['old'] == None or change['old'] == '':
            return

        if self.dataset is not None and 'new' in change:
            self.dataset = self._open_dataset(source=change['new'])

        try:
            if self.native_coordinates is not None:
                self.native_coordinates = self.get_native_coordinates()
        except NotImplementedError:
            pass

  
    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        
        Raises
        ------
        NotImplementedError
            DAP has no mechanism for creating coordinates automatically, so this is left up to child classes.
        """
        raise NotImplementedError("DAP has no mechanism for creating coordinates" +
                                  ", so this is left up to child class " +
                                  "implementations.")

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        data = self.dataset[self.datakey][tuple(coordinates_index)].data
        d = self.create_output_array(coordinates, data=data.reshape(coordinates.shape))
        return d
    
    @property
    def keys(self):
        """The list of available keys from the OpenDAP dataset.
        
        Returns
        -------
        List
            The list of available keys from the OpenDAP dataset. Any of these keys can be set as self.datakey
        """
        return self.dataset.keys()

# TODO: rename "Rasterio" to be more consistent with other naming conventions
@common_doc(COMMON_DATA_DOC)
class Rasterio(DataSource):
    """Create a DataSource using Rasterio.
    
    Attributes
    ----------
    band : int
        The 'band' or index for the variable being accessed in files such as GeoTIFFs
    dataset : Any
        A reference to the datasource opened by rasterio
    native_coordinates : Coordinates
        {native_coordinates}
    source : str
        Path to the data source
    """
    
    source = tl.Unicode(allow_none=False)
    dataset = tl.Any(allow_none=True)
    band = tl.CInt(1).tag(attr=True)
    
    @tl.default('dataset')
    def open_dataset(self, source=None):
        """Opens the data source
        
        Parameters
        ----------
        source : str, optional
            Uses self.source by default. Path to the data source.
        
        Returns
        -------
        Any
            raster.open(source)
        """
        if source is None:
            source = self.source
        else:
            self.source = source

        # TODO: dataset should not open by default
        # prefer with as: syntax
        return rasterio.open(source)
    
    def close_dataset(self):
        """Closes the file for the datasource
        """
        self.dataset.close()

    @tl.observe('source')
    def _update_dataset(self, change):
        if self.dataset is not None:
            self.dataset = self.open_dataset(change['new'])
        self.native_coordinates = self.get_native_coordinates()
        
    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        
        The default implementation tries to find the lat/lon coordinates based on dataset.affine or dataset.transform
        (depending on the version of rasterio). It cannot determine the alt or time dimensions, so child classes may
        have to overload this method.
        """
        
        if hasattr(self.dataset, 'affine'):
            affine = self.dataset.affine
        else:
            affine = self.dataset.transform

        left, bottom, right, top = self.dataset.bounds

        if affine[1] != 0.0 or affine[3] != 0.0:
            raise NotImplementedError("Rotated coordinates are not yet supported")

        return Coordinates([
            UniformCoordinates1d(bottom, top, size=self.dataset.height, name='lat'),
            UniformCoordinates1d(left, right, size=self.dataset.width, name='lon')
        ])

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        data = self.create_output_array(coordinates)
        slc = coordinates_index
        window = window=((slc[0].start, slc[0].stop), (slc[1].start, slc[1].stop))
        a = self.dataset.read(self.band, out_shape=tuple(coordinates.shape))
        data.data.ravel()[:] = a.ravel()
        return data
    
    @cached_property
    def band_count(self):
        """The number of bands
        
        Returns
        -------
        int
            The number of bands in the dataset
        """
        return self.dataset.count
    
    @cached_property
    def band_descriptions(self):
        """A description of each band contained in dataset.tags
        
        Returns
        -------
        OrderedDict
            Dictionary of band_number: band_description pairs. The band_description values are a dictionary, each 
            containing a number of keys -- depending on the metadata
        """
        bands = OrderedDict()
        for i in range(self.dataset.count):
            bands[i] = self.dataset.tags(i + 1)
        return bands

    @cached_property
    def band_keys(self):
        """An alternative view of band_descriptions based on the keys present in the metadata
        
        Returns
        -------
        dict
            Dictionary of metadata keys, where the values are the value of the key for each band. 
            For example, band_keys['TIME'] = ['2015', '2016', '2017'] for a dataset with three bands.
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
        """Return the bands that have a key equal to a specified value.
        
        Parameters
        ----------
        key : str / list
            Key present in the metadata of the band. Can be a single key, or a list of keys.
        value : str / list
            Value of the key that should be returned. Can be a single value, or a list of values
        
        Returns
        -------
        np.ndarray
            An array of band numbers that match the criteria
        """
        if (not hasattr(key, '__iter__') or isinstance(key, string_types))\
                and (not hasattr(value, '__iter__') or isinstance(value, string_types)):
            key = [key]
            value = [value]

        match = np.ones(self.band_count, bool)
        for k, v in zip(key, value):
            match = match & (np.array(self.band_keys[k]) == v)
        matches = np.where(match)[0] + 1

        return matches

WCS_DEFAULT_VERSION = u'1.0.0'
WCS_DEFAULT_CRS = 'EPSG:4326'

class WCS(DataSource):
    """Create a DataSource from an OGC-complient WCS service
    
    Attributes
    ----------
    crs : 'str'
        Default is EPSG:4326 (WGS84 Geodic) EPSG number for the coordinate reference system that the data should
        be returned in.
    layer_name : str
        Name of the WCS layer that should be fetched from the server
    source : str
        URL of the WCS server endpoint
    version : str
        Default is 1.0.0. WCS version string.
    wcs_coordinates : Coordinates
        The coordinates of the WCS source
    """
    
    source = tl.Unicode()
    layer_name = tl.Unicode().tag(attr=True)
    version = tl.Unicode(WCS_DEFAULT_VERSION).tag(attr=True)
    crs = tl.Unicode(WCS_DEFAULT_CRS).tag(attr=True)
    wcs_coordinates = tl.Instance(Coordinates)   # default below

    _get_capabilities_qs = tl.Unicode('SERVICE=WCS&REQUEST=DescribeCoverage&'
                                      'VERSION={version}&COVERAGE={layer}')
    _get_data_qs = tl.Unicode('SERVICE=WCS&VERSION={version}&REQUEST=GetCoverage&'
                              'FORMAT=GeoTIFF&COVERAGE={layer}&'
                              'BBOX={w},{s},{e},{n}&CRS={crs}&RESPONSE_CRS={crs}&'
                              'WIDTH={width}&HEIGHT={height}&TIME={time}')

    # TODO: This should be capabilities_url, not get_
    @property
    def get_capabilities_url(self):
        """Constructs the url that requests the WCS capabilities
        
        Returns
        -------
        str
            The url that requests the WCS capabilities
        """
        return self.source + '?' + self._get_capabilities_qs.format(
            version=self.version, layer=self.layer_name)

    @tl.default('wcs_coordinates')
    def get_wcs_coordinates(self):
        """Retrieves the native coordinates reported by the WCS service.
        
        Returns
        -------
        Coordinates
            The native coordinates reported by the WCS service.
        
        Notes
        -------
        This assumes a `time`, `lat`, `lon` order for the coordinates, and currently doesn't handle `alt` coordinates
        
        Raises
        ------
        Exception
            Raises this if the required dependencies are not installed.
        """
        if requests is not None:
            capabilities = requests.get(self.get_capabilities_url)
            if capabilities.status_code != 200:
                raise Exception("Could not get capabilities from WCS server")
            capabilities = capabilities.text

        # TODO: remove support urllib3 - requests is sufficient
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
        left = lonlat[:, 0].min() + dlondlat[0] / 2
        right = lonlat[:, 0].max() - dlondlat[0] / 2

        timedomain = capabilities.find("wcs:temporaldomain")
        if timedomain is None:
            return Coordinates([UniformCoordinates1d(top, bottom, size=size[1], name='lat')])
        
        date_re = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}')
        times = str(timedomain).replace('<gml:timeposition>', '').replace('</gml:timeposition>', '').split('\n')
        times = np.array([t for t in times if date_re.match(t)], np.datetime64)
        
        return Coordinates([
            ArrayCoordinates1d(times, name='time'),
            UniformCoordinates1d(top, bottom, size=size[1], name='lat'),
            UniformCoordinates1d(left, right, size=size[0], name='lon')
        ])        

    @property
    @common_doc(COMMON_DATA_DOC)
    def native_coordinates(self):
        """{native_coordinates}
        
        Returns
        -------
        Coordinates
            {native_coordinates}
            
        Notes
        ------
        This is a little tricky and doesn't fit into the usual PODPAC method, as the service is actually doing the 
        data wrangling for us...
        """

        if not self.requested_coordinates:
            return self.wcs_coordinates

        cs = []
        for dim in self.wcs_coordinates.dims:
            if dim in self.requested_coordinates.dims:
                c = self.requested_coordinates[dim]
                if c.size == 1:
                    cs.append(ArrayCoordinates1d(c.coordinates[0], name=dim))
                elif isinstance(c, UniformCoordinates1d):
                    cs.append(UniformCoordinates1d(c.bounds[0], c.bounds[1], abs(c.step), name=dim))
                else:
                    # TODO: generalize/fix this
                    # WCS calls require a regular grid, could (otherwise we have to do multiple WCS calls)
                    cs.append(UniformCoordinates1d(c.bounds[0], c.bounds[1], size=c.size, name=dim))
            else:
                cs.append(self.wcs_coordinates[dim])
        c = Coordinates(cs)
        return c

    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        
        Raises
        ------
        Exception
            Raises this if there is a network error or required dependencies are not installed.
        """
        output = self.create_output_array(coordinates)
        dotime = 'time' in self.wcs_coordinates.dims

        if 'time' in coordinates.dims and dotime:
            sd = np.timedelta64(0, 's')
            times = [str(t+sd) for t in coordinates['time'].coordinates]
        else:
            times = ['']
        
        if len(times) > 1:
            for i, time in enumerate(times):
                url = self.source + '?' + self._get_data_qs.format(
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

                # TODO: remove support urllib3 - requests is sufficient
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
            
            url = self.source + '?' + self._get_data_qs.format(
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

            # TODO: remove support urllib3 - requests is sufficient
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

# We mark this as an algorithm node for the sake of the pipeline, although
# the "algorithm" portion is not being used / is overwritten by the DataSource
# In particular, this is required for providing coordinates_source
# We should be able to to remove this requirement of attributes in the pipeline 
# can have nodes specified... 
class ReprojectedSource(DataSource, Algorithm):
    """Create a DataSource with a different resolution from another Node. This can be used to bilinearly interpolated a
    dataset after averaging over a larger area.
    
    Attributes
    ----------
    coordinates_source : Node
        Node which is used as the source
    reprojected_coordinates : Coordinates
        Coordinates where the source node should be evaluated. 
    source : Node
        The source node
    source_interpolation : str
        Type of interpolation method to use for the source node
    """
    
    source_interpolation = tl.Unicode('nearest_preview').tag(attr=True)
    source = tl.Instance(Node)
    # Specify either one of the next two
    coordinates_source = tl.Instance(Node, allow_none=True).tag(attr=True)
    reprojected_coordinates = tl.Instance(Coordinates).tag(attr=True)

    @tl.default('reprojected_coordinates')
    def get_reprojected_coordinates(self):
        """Retrieves the reprojected coordinates in case coordinates_source is specified
        
        Returns
        -------
        reprojected_coordinates : Coordinates
            Coordinates where the source node should be evaluated. 
        
        Raises
        ------
        Exception
            If neither coordinates_source or reproject_coordinates are specified
        """
        if not hasattr(self, 'coordinates_source'):
            raise Exception("Either reprojected_coordinates or coordinates_source must be specified")
        
        return self.coordinates_source.native_coordinates

    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        """
        if isinstance(self.source, DataSource):
            sc = self.source.native_coordinates
        else: # Otherwise we cannot guarantee that native_coordinates exist
            sc = self.reprojected_coordinates
        rc = self.reprojected_coordinates
        coords = [rc[dim] if dim in rc.dims else sc[dim] for dim in sc.dims]
        return Coordinates(coords)

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        self.source.interpolation = self.source_interpolation
        data = self.source.eval(coordinates)
        
        # The following is needed in case the source is an algorithm
        # or compositor node that doesn't have all the dimensions of
        # the reprojected coordinates
        # TODO: What if data has coordinates that reprojected_coordinates doesn't have
        keep_dims = list(data.coords.keys())
        drop_dims = [d for d in coordinates.dims if d not in keep_dims]
        coordinates.drop(drop_dims)
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
    def base_definition(self):
        """ Base node definition. 
        
        Returns
        -------
        OrderedDict
            Base node definition. 
        
        Raises
        ------
        NotImplementedError
            If coordinates_source is None, this raises an error because serialization of reprojected_coordinates 
            is not implemented
        """
        
        d = Algorithm.base_definition.fget(self)
        d['attrs'] = OrderedDict()
        if self.interpolation:
            d['attrs']['interpolation'] = self.interpolation
        if self.coordinates_source is None:
            # TODO serialize reprojected_coordinates
            raise NotImplementedError
        return d

class S3(DataSource):
    """Create a DataSource from a file on an S3 Bucket. 
    
    Attributes
    ----------
    node : Node, optional
        The DataSource node used to interpret the S3 file
    node_class : DataSource, optional
        The class type of self.node. This is used to create self.node if self.node is not specified
    node_kwargs : dict, optional
        Keyword arguments passed to `node_class` when automatically creating `node`
    return_type : str, optional
        Either: 'file_handle' (for files downloaded to RAM); or
        the default option 'path' (for files downloaded to disk)
    s3_bucket : str, optional
        Name of the S3 bucket. Uses settings.S3_BUCKET_NAME by default.
    s3_data : file/str
        If return_type == 'file_handle' returns a file pointer object
        If return_type == 'path' returns a string to the data
    source : str
        Path to the file residing in the S3 bucket that will be loaded
    """
    
    source = tl.Unicode()
    node = tl.Instance(Node)
    node_class = tl.Type(DataSource)  # A class
    node_kwargs = tl.Dict(default_value={})
    s3_bucket = tl.Unicode(allow_none=True)
    s3_data = tl.Any(allow_none=True)
    _temp_file_cleanup = tl.List()
    return_type = tl.Enum(['file_handle', 'path'], default_value='path')
    # TODO: handle s3 auth setup
    
    @tl.default('node')
    def node_default(self):
        """Creates the default node using the node_class and node_kwargs
        
        Returns
        -------
        self.node_class
            Instance of self.node_class
        
        Raises
        ------
        Exception
            This function sets the source in the node, so 'source' cannot be present in node_kwargs
        """
        if 'source' in self.node_kwargs:
            raise Exception("'source' present in node_kwargs for S3")

        return self.node_class(source=self.s3_data, **self.node_kwargs)

    @tl.default('s3_bucket')
    def s3_bucket_default(self):
        """Retrieves default S3 Bucket from settings
        
        Returns
        -------
        Str
            Name of the S3 bucket
        """
        return settings.S3_BUCKET_NAME

    @tl.default('s3_data')
    def s3_data_default(self):
        """Returns the file handle or path to the S3 bucket
        
        Returns
        -------
        str/file
            Either a string to the downloaded file path, or a file handle
        """
        if self.s3_bucket is None:
            raise ValueError('No s3 bucket set')

        s3 = boto3.resource('s3').Bucket(self.s3_bucket)

        if self.return_type == 'file_handle':
            # TODO: should this use the with/as syntax
            # https://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.download_fileobj
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

            # TODO: should we handle temp files here?
            #self._temp_file_cleanup.append(tmppath)
            return tmppath

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        self.nan_vals = getattr(self.node, 'nan_vals', [])
        return self.node.get_data(coordinates, coordinates_index)

    @property
    @common_doc(COMMON_DATA_DOC)
    def native_coordinates(self):
        """{native_coordinates}
        """
        return self.node.native_coordinates

    def __del__(self):
        if hasattr(super(S3), '__del__'):
            super(S3).__del__(self)
        for f in self._temp_file_cleanup:
            os.remove(f)
