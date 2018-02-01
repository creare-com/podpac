from __future__ import division, print_function, absolute_import

import os
import glob
import shutil
import inspect
from collections import OrderedDict
from io import BytesIO
import base64
import json
import xarray as xr
import numpy as np
import traitlets as tl
import matplotlib.colors, matplotlib.cm

try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

try:
    import boto3
except:
    boto3 = None

from podpac import settings
from podpac import Units, UnitsDataArray
from podpac import Coordinate

class NodeException(Exception):
    pass

class Style(tl.HasTraits):
    
    def __init__(self, node=None, *args, **kwargs):
        if node:
            self.name = self.node.__class.__name__
            self.units = self.node.units
        super(Style, self).__init__(*args, **kwargs)
        
    name = tl.Unicode()
    units = Units(allow_none=True)
    
    is_enumerated = tl.Bool(default_value=False)
    enumeration_legend = tl.Tuple(trait=tl.Unicode)
    enumeration_colors = tl.Tuple(trait=tl.Tuple)
    
    clim = tl.List(default_value=[None, None])
    cmap = tl.Instance(matplotlib.colors.Colormap)
    tl.default('cmap')
    def _cmap_default(self):
        return matplotlib.cm.get_cmap('viridis')

class Node(tl.HasTraits):
    output = tl.Instance(UnitsDataArray, allow_none=True, default_value=None)
    @tl.default('output')
    def _output_default(self):
        return self.initialize_output_array('nan')
    
    native_coordinates = tl.Instance('podpac.core.coordinate.Coordinate',
                                     allow_none=True)
    evaluted = tl.Bool(default_value=False)
    implicit_pipeline_evaluation = tl.Bool(default_value=True,
        help="Evaluate the pipeline implicitly (True, Default)")
    evaluated_coordinates = tl.Instance('podpac.core.coordinate.Coordinate', 
                                        allow_none=True)
    params = tl.Dict(default_value=None, allow_none=True)
    units = Units(default_value=None, allow_none=True)
    dtype = tl.Any(default_value=float)
    cache_type = tl.Enum([None, 'disk', 'ram'], allow_none=True)    
    
    style = tl.Instance(Style)
    @tl.default('style')
    def _style_default(self):
        return Style()
    
    @property
    def shape(self):
        # Changes here likely will also require changes in initialize_output_array
        ev = self.evaluated_coordinates
        #nv = self._trait_values.get('native_coordinates',  None)
        # Switching from _trait_values to hasattr because "native_coordinates"
        # not showing up in _trait_values
        if hasattr(self,'native_coordinates'):
            nv = self.native_coordinates
        else:
            nv = None
        if ev is not None and nv is not None:
            return nv.get_shape(ev)
        elif ev is not None and nv is None:
            return ev.shape
        elif nv is not None:
            return nv.shape    
        else:
            raise NodeException("Cannot determine shape if "
                                "evaluated_coordinates and native_coordinates"
                                " are both None.")
    
    def __init__(self, **kwargs):
        """ Do not overwrite me """
        tkwargs = self._first_init(**kwargs)
        super(Node, self).__init__(**tkwargs)
        self.init()

    def _first_init(self, **kwargs):
        """ Only overwrite me if absolutely necessary """
        return kwargs

    def init(self):
        pass

    def execute(self, coordinates, params=None, output=None):
        """ This is the common interface used for ALL nodes. Pipelines only
        understand this and get_description. 
        """
        raise NotImplementedError
    
    def initialize_output_array(self, init_type='nan', fillval=0, style=None,
                              no_style=False, shape=None, coords=None,
                              dims=None, units=None, dtype=np.float, **kwargs):
        # Changes here likely will also require changes in shape
        if coords is None: 
            coords = self.evaluated_coordinates
        if not isinstance(coords, (Coordinate)):
            coords = Coordinate(coords)
        #if self._trait_values.get("native_coordinates", None) is not None:
        # Switching from _trait_values to hasattr because "native_coordinates"
        # not showing up in _trait_values        
        if hasattr(self, "native_coordinates") and self.native_coordinates is not None:
            crds = self.native_coordinates.replace_coords(coords).coords
        else:
            crds = coords.coords        
        dims = list(crds.keys())
        return self.initialize_array(init_type, fillval, style, no_style, shape,
                                     crds, dims, units, dtype, **kwargs)
    
    def initialize_coord_array(self, coords, init_type='nan', fillval=0, 
                               style=None, no_style=False, units=None,
                               dtype=np.float, **kwargs):
        return self.initialize_array(init_type, fillval, style, no_style, 
                                     coords.shape, coords.coords, coords.dims,
                                     units, dtype, **kwargs)

    def initialize_array(self, init_type='nan', fillval=0, style=None,
                              no_style=False, shape=None, coords=None,
                              dims=None, units=None, dtype=np.float,  **kwargs):
        """Initialize output data array

        Parameters
        -----------
        init_type : str, optional
            How to initialize the array. Options are:
                nan: uses np.full(..., np.nan) (Default option)
                empty: uses np.empty 
                zeros: uses np.zeros()
                ones: uses np.ones
                full: uses np.full(..., fillval)
                data: uses the fillval as the input array
        fillval : number, optional
            used if init_type=='full' or 'data', default = 0
        style : Style, optional
            The style to use for plotting. Uses self.style by default
        no_style : bool, optional
            Default is False. If True, self.style will not be assigned to 
            arr.attr['layer_style']
        shape : tuple, optional
            Shape of array. Uses self.shape by default.
        coords : dict/list, optional
            input to UnitsDataArray, uses self.coords by default
        dims : list(str), optional
            input to UnitsDataArray, uses self.native_coords.dims by default
        units : pint.unit.Unit, optional
            Default is self.units The Units for the data contained in the 
            DataArray
        dtype : type, optional
            Default is np.float. Datatype used by default
        kwargs : kwargs
            other keyword arguments passed to UnitsDataArray

        Returns
        -------
        arr : UnitsDataArray
            Unit-aware xarray DataArray of the desired size initialized using 
            the method specified
        """
        if style is None: style = self.style
        if shape is None: shape = self.shape
        if units is None: units = self.units
        if not isinstance(coords, (dict, OrderedDict)): coords = dict(coords)

        if init_type == 'empty':
            data = np.empty(shape)
        elif init_type == 'nan':
            data = np.full(shape, np.nan)
        elif init_type == 'zeros':
            data = np.zeros(shape)
        elif init_type == 'ones':
            data = np.ones(shape)
        elif init_type == 'full':
            data = np.full(shape, fillval)
        elif init_type == 'data':
            data = fillval
        else:
            raise ValueError("Unknown init_type=%" % init_type)
        
        x = UnitsDataArray(data, coords=coords, dims=dims, **kwargs)
        
        if not no_style:
            x.attrs['layer_style'] = style
        if units is not None:
            x.attrs['units'] = units
        x.attrs['params'] = self.params
        return x

    def plot(self, show=True, interpolation='none', **kwargs):
        """
        Plot function to display the output
        
        TODO: Improve this substantially please
        """
        
        import matplotlib.pyplot as plt

        if kwargs:
            plt.imshow(self.output.data, cmap=self.style.cmap,
                       interpolation=interpolation, **kwargs)
        else:
            self.output.plot()
        if show:
            plt.show()

    @property
    def base_ref(self):
        """
        Default pipeline node reference/name in pipeline node definitions
        """
        return self.__class__.__name__
    
    def _base_definition(self):
        """ populates 'node' and 'plugin', if necessary """
        d = OrderedDict()
        
        if self.__module__ == 'podpac':
            d['node'] = self.__class__.__name__
        elif self.__module__.startswith('podpac.'):
            _, module = self.__module__.split('.', 1)
            d['node'] = '%s.%s' % (module, self.__class__.__name__)
        else:
            d['plugin'] = self.__module__
            d['node'] = self.__class__.__name__

        return d

    @property
    def definition(self):
        """
        Pipeline node definition. Implemented in primary base nodes, with
        custom implementations or extensions necessary for specific nodes.

        Should be an OrderedDict with at least a 'node' attribute.
        """
        parents = inspect.getmro(self.__class__)
        podpac_parents = [
            '%s.%s' % (p.__module__.split('.', 1)[1:], p.__name__)
            for p in parents
            if p.__module__.startswith('podpac.')]
        raise NotImplementedError('See %s' % ', '.join(podpac_parents))

    @property
    def pipeline_definition(self):
        """
        Full pipeline definition for this node.
        """

        from podpac.core.pipeline import make_pipeline_definition
        return make_pipeline_definition(self)

    @property
    def pipeline_json(self):
        return json.dumps(self.pipeline_definition, indent=4)

    @property
    def pipeline(self):
        from pipeline import Pipeline
        return Pipeline(self.pipeline_definition)

    def get_hash(self, coordinates=None, params=None):
        if params is not None:
            # convert to OrderedDict with consistent keys
            params = OrderedDict(sorted(params.items()))
            
            # convert dict values to OrderedDict with consistent keys
            for key, value in params.items():
                if type(value) is dict:
                    params[key] = OrderedDict(sorted(value.items()))

        return hash((str(coordinates), str(params)))

    @property
    def evaluated_hash(self):
        if self.evaluated_coordinates is None:
            raise Exception("node not evaluated")
            
        return self.get_hash(self.evaluated_coordinates, self.params)

    @property
    def latlon_bounds_str(self):
        return self.evaluated_coordinates.latlon_bounds_str
    
    def get_output_path(self, filename, outdir=None):
        if outdir is None:
            outdir = settings.OUT_DIR

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        return os.path.join(outdir, filename)

    def write(self, name, outdir=None, format='pickle'):
        filename = '%s_%s_%s.pkl' % (
            name,
            self.evaluated_hash,
            self.latlon_bounds_str)
        path = self.get_output_path(filename, outdir=outdir)

        if format == 'pickle':
            with open(path, 'wb') as f:
                cPickle.dump(self.output, f)
        else:
            raise NotImplementedError

    def load(self, name, coordinates, params, outdir=None):
        filename = '%s_%s_%s.pkl' % (
            name,
            self.get_hash(coordinates, params),
            coordinates.latlon_bounds_str)
        path = self.get_output_path(filename, outdir=outdir)
        
        with open(path, 'rb') as f:
            self.output = cPickle.load(f)

    def load_from_file(self, path):
        with open(path, 'rb') as f:
            output = cPickle.load(f)

        self.output = output
        self.evaluated_coordinates = self.output.coordinates
        self.params = self.output.attrs['params']

    def get_image(self, format='png', vmin=None, vmax=None):
        import matplotlib
        matplotlib.use('agg')
        from matplotlib import cm
        from matplotlib.image import imsave

        data = self.output.data.squeeze()

        if np.isnan(vmin):

            vmin = np.nanmin(data)
        if np.isnan(vmax):
            vmax = np.nanmax(data)      
        if vmax == vmin:
            vmax +=  1e-16
            
        c = (data - vmin) / (vmax - vmin)
        i = cm.viridis(c, bytes=True)
        im_data = BytesIO()
        imsave(im_data, i, format='png')
        im_data.seek(0)
        return base64.b64encode(im_data.getvalue())

    @property
    def cache_dir(self):
        basedir = settings.CACHE_DIR
        subdir = str(self.__class__)[8:-2].split('.')
        dirs = [basedir] + subdir
        return os.path.join(*dirs)
    
    def cache_path(self, filename):
        pre = str(self.source).replace('/', '_').replace('\\', '_').replace(':', '_')
        return os.path.join(self.cache_dir, pre  + '_' + filename)
    
    def cache_obj(self, obj, filename):
        path = self.cache_path(filename)
        if settings.S3_BUCKET_NAME is None or settings.CACHE_TO_S3 == False:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            with open(path, 'wb') as fid:
                cPickle.dump(obj, fid)#, protocol=cPickle.HIGHEST_PROTOCOL)
        else:
            s3 = boto3.resource('s3').Bucket(settings.S3_BUCKET_NAME)
            io = BytesIO(cPickle.dumps(obj))
            s3.upload_fileobj(io, path) 
            
    def load_cached_obj(self, filename):
        path = self.cache_path(filename)
        if settings.S3_BUCKET_NAME is None or settings.CACHE_TO_S3 == False:
            with open(path, 'rb') as fid:
                obj = cPickle.load(fid)
        else:
            s3 = boto3.resource('s3').Bucket(settings.S3_BUCKET_NAME)
            io = BytesIO()
            s3.download_fileobj(path, io)
            io.seek(0)
            obj = cPickle.loads(io.read()) 
        return obj
    
    def clear_disk_cache(self, attr='*', node_cache=False, all_cache=False):
        """ Helper function to clear disk cache. 
        
        WARNING: This function will permanently delete cached values
        
        Parameters
        ------------
        attr: str, optional
            Default '*'. Specific attribute to be cleared for specific 
            instance of this Node. By default all attributes are cleared.
        node_cache: bool, optional
            Default False. If True, will ignore `attr` and clear all attributes
            for all variants/instances of this Node. 
        all_cache: bool, optional
            Default False. If True, will clear the entire podpac cache. 
        """
        if all_cache:
            shutil.rmtree(settings.CACHE_DIR)
        elif node_cache:
            shutil.rmtree(self.cache_dir)
        else: 
            for f in glob.glob(self.cache_path(attr)):
                os.remove(f)
            

if __name__ == "__main__":
    # checking creation of output node
    c1 = Coordinate(lat_lon=((0, 1, 10), (0, 1, 10)), time=(0, 1, 2))
    c2 = Coordinate(lat_lon=((0.5, 1.5, 15), (0.1, 1.1, 15)))
    
    n = Node(native_coordinates=c1)
    print (n.initialize_output_array().shape)
    n.evaluated_coordinates = c2
    print (n.initialize_output_array().shape)
    
    n = Node(native_coordinates=c1.unstack())
    print (n.initialize_output_array().shape)
    n.evaluated_coordinates = c2
    print (n.initialize_output_array().shape)
    
    n = Node(native_coordinates=c1)
    print (n.initialize_output_array().shape)
    n.evaluated_coordinates = c2.unstack()
    print (n.initialize_output_array().shape)
    print ("Nothing to do")
