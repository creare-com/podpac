"""
Node Summary
"""

from __future__ import division, print_function, absolute_import

import os
import glob
import shutil
import inspect
from collections import OrderedDict
from io import BytesIO
import base64
import json
import numpy as np
import traitlets as tl
import matplotlib
import matplotlib.cm


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
from podpac.core.utils import common_doc

COMMON_NODE_DOC = {
    'native_coordinates': 
        '''The native set of coordinates for a node. This attribute may be `None` for some nodes.''',
    'evaluated_coordinates': 
        '''The set of coordinates requested by a user. The Node will be executed using these coordinates.''',
    'execute_params': 'Default is None. Runtime parameters that modify any default node parameters.',
    'execute_out': 
        '''Default is None. Optional input array used to store the output data. When supplied, the node will not 
            allocate its own memory for the output array. This array needs to have the correct dimensions and 
            coordinates.''',
    'execute_method': 
        '''Default is None. How the node will be executed: serial, parallel, on aws, locally, etc. Currently only local
            execution is supported.''',
    'execute_return': 
        '''UnitsDataArray
            Unit-aware xarray DataArray containing the results of the node execution.''',
    'hash_return': 'A unique hash capturing the coordinates and parameters used to execute the node. ',
    'outdir': 'Optional output directory. Uses settings.CACHE_DIR by default',
    'definition_return': '''OrderedDict
            Dictionary containing the location of the Node, the name of the plugin (if required), as well as any 
            parameters and attributes that were tagged by children.''',
    'arr_init_type': 
        '''How to initialize the array. Options are:
                nan: uses np.full(..., np.nan) (Default option)
                empty: uses np.empty
                zeros: uses np.zeros()
                ones: uses np.ones
                full: uses np.full(..., fillval)
                data: uses the fillval as the input array''',
    'arr_fillval' : "used if init_type=='full' or 'data', default = 0",
    'arr_style' : "The style to use for plotting. Uses self.style by default",
    'arr_no_style' : "Default is False. If True, self.style will not be assigned to arr.attr['layer_style']",
    'arr_shape': 'Shape of array. Uses self.shape by default.',
    'arr_coords' : "Input to UnitsDataArray (i.e. an xarray coords dictionary/list)",
    'arr_dims' : "Input to UnitsDataArray (i.e. an xarray dims list of strings)",
    'arr_units' : "Default is self.units The Units for the data contained in the DataArray.",
    'arr_dtype' :"Default is np.float. Datatype used by default",
    'arr_kwargs' : "Dictioary of any additional keyword arguments that will be passed to UnitsDataArray.",
    'arr_return' : 
        """UnitsDataArray
            Unit-aware xarray DataArray of the desired size initialized using the method specified.
            """
    }

COMMON_DOC = COMMON_NODE_DOC.copy()

class NodeException(Exception):
    """Summary
    """
    pass

class Style(tl.HasTraits):
    """Summary

    Attributes
    ----------
    clim : TYPE
        Description
    cmap : TYPE
        Description
    enumeration_colors : TYPE
        Description
    enumeration_legend : TYPE
        Description
    is_enumerated : TYPE
        Description
    name : TYPE
        Description
    units : TYPE
        Description
    """

    def __init__(self, node=None, *args, **kwargs):
        if node:
            self.name = node.__class__.__name__
            self.units = node.units
        super(Style, self).__init__(*args, **kwargs)

    name = tl.Unicode()
    units = Units(allow_none=True)

    is_enumerated = tl.Bool(default_value=False)
    enumeration_legend = tl.Tuple(trait=tl.Unicode)
    enumeration_colors = tl.Tuple(trait=tl.Tuple)

    clim = tl.List(default_value=[None, None])
    cmap = tl.Instance(matplotlib.colors.Colormap)
    
    @tl.default('cmap') 
    def _cmap_default(self):
        return matplotlib.cm.get_cmap('viridis')

@common_doc(COMMON_DOC)
class Node(tl.HasTraits):
    """The base class for all Nodes, which defines the common interface for everything.

    Attributes
    ----------
    cache_type : [None, 'disk', 'ram']
        How the output of the nodes should be cached. By default, outputs are not cached.
    dtype : type
        The numpy datatype of the output. Currently only `float` is supported.
    evaluated : Bool
        Flag indicating if the node has been evaluated.
    evaluated_coordinates : podpac.Coordinate
        {evaluated_coordinates}
        This attribute stores the coordinates that were last used to evaluate the node.
    implicit_pipeline_evaluation : Bool
        Flag indicating if nodes as part of a pipeline should be automatically evaluated when
        the root node is evaluated. This attribute is planned for deprecation in the future.
    native_coordinates : podpac.Coordinate, optional
        {native_coordinates} 
    node_defaults : dict
        Dictionary of defaults values for attributes of a Node. 
    output : podpac.UnitsDataArray
        Output data from the last evaluation of the node. 
    params : dict
        Dictionary of parameters that control the output of a node. For example, these can be coefficients in an 
        equation, or the interpolation type. This attribute is planned for deprecation in the future.
    style : podpac.Style
        Object discribing how the output of a node should be displayed. This attribute is planned for deprecation in the
        future.
    units : podpac.Units
        The units of the output data, defined using the pint unit registry `podpac.units.ureg`.
    interpolation : str, optional
        The interpolation type to use for the node. Not all nodes use this attribute.
    """

    output = tl.Instance(UnitsDataArray, allow_none=True, default_value=None)
    @tl.default('output')
    def _output_default(self):
        return self.initialize_output_array('nan')

    native_coordinates = tl.Instance('podpac.core.coordinate.Coordinate',
                                     allow_none=True, default=None)
    evaluated = tl.Bool(default_value=False)
    implicit_pipeline_evaluation = tl.Bool(default_value=True, help="Evaluate the pipeline implicitly (True, Default)")
    evaluated_coordinates = tl.Instance('podpac.core.coordinate.Coordinate',
                                        allow_none=True)
    _params = tl.Dict(default=None, allow_none=True)
    units = Units(default_value=None, allow_none=True)
    dtype = tl.Any(default_value=float)
    cache_type = tl.Enum([None, 'disk', 'ram'], allow_none=True)

    node_defaults = tl.Dict(allow_none=True)
    
    interpolation = tl.Unicode('')

    style = tl.Instance(Style)
    @tl.default('style')
    def _style_default(self):
        return Style()

    @property
    def shape(self):
        """See `get_output_shape`
        """
        return self.get_output_shape()

    def __init__(self, **kwargs):
        """ Do not overwrite me """
        tkwargs = self._first_init(**kwargs)

        # Add default values listed in dictionary
        # self.node_defaults.update(tkwargs) <-- could almost do this...
        #                                        but don't want to overwrite
        #                                        node_defaults and want to
        #                                        ignore 'node_defaults'
        for key, val in self.node_defaults.items():
            if key == 'node_defaults':
                continue  # ignore this entry
            if key not in tkwargs:  # Only add value if not in input
                tkwargs[key] = val

        # Call traitlest constructor
        super(Node, self).__init__(**tkwargs)
        self.init()

    def _first_init(self, **kwargs):
        """Only overwrite me if absolutely necessary

        Parameters
        ----------
        **kwargs
            Keyword arguments provided by user when class was instantiated.

        Returns
        -------
        dict
            Keyword arguments that will be passed to the standard intialization function.
        """
        return kwargs

    def init(self):
        """Overwrite this method if a node needs to do any additional initialization after the standard initialization.
        """
        pass

    @common_doc(COMMON_DOC)
    def execute(self, coordinates, params=None, output=None, method=None):
        """This is the common interface used for ALL nodes. Pipelines only
        understand this method and get_description.

        Parameters
        ----------
        coordinates : podpac.Coordinate
            {evaluated_coordinates}
        params : dict, optional
            {execute_params} 
        output : podpac.UnitsDataArray, optional
            {execute_out}
        method : str, optional
            {execute_method}

        Raises
        ------
        NotImplementedError
            Children need to implement this method, otherwise this error is raised. 
        """
        raise NotImplementedError
    
    def get_output_shape(self, coords=None):
        """Returns the shape of the output based on the evaluated coordinates. This shape is jointly determined by the
        input or evaluated coordinates and the native_coordinates (if present).

        Parameters
        -----------
        coords: podpac.Coordinates, optional
            Requested coordinates that help determine the shape of the output. Uses self.evaluated_coordinates if not 
            supplied. 

        Returns
        -------
        tuple/list
            Size of the dimensions of the output

        Raises
        ------
        NodeException
            If the shape cannot be automatically determined, this exception is raised. 
        """

        # Changes here likely will also require changes in initialize_output_array
        if coords is None: 
            ev = self.evaluated_coordinates
        else: 
            ev = coords
        #nv = self._trait_values.get('native_coordinates',  None)
        # Switching from _trait_values to hasattr because "native_coordinates"
        # not showing up in _trait_values
        if hasattr(self, 'native_coordinates'):
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

    def get_output_dims(self, coords=None):
        """Returns the dimensions of the output coordinates based on the user-requested coordinates. This is jointly 
        determined by the input or evaluated coordinates and the native_coordinates (if present).

        Parameters
        ----------
        coords : podpac.Coordinates, optional
            Requested coordinates that help determine the coordinates of the output. Uses self.evaluated_coordinates if 
            not supplied.

        Returns
        -------
        list
            A list of the coordinates
        """
        # Changes here likely will also require changes in shape
        if coords is None:
            coords = self.evaluated_coordinates
        if not isinstance(coords, (Coordinate)):
            coords = Coordinate(coords)

        #if self._trait_values.get("native_coordinates", None) is not None:
        # Switching from _trait_values to hasattr because "native_coordinates"
        # not showing up in _trait_values
        if hasattr(self, "native_coordinates") and self.native_coordinates is not None:
            dims = self.native_coordinates.dims
        else:
            dims = coords.dims
        return dims

    def get_output_coords(self, coords=None):
        """Returns the output coordinates based on the user-requested coordinates. This is jointly determined by 
        the input or evaluated coordinates and the native_coordinates (if present).

        Parameters
        ----------
        coords : podpac.Coordinates, optional
            Requested coordinates that help determine the coordinates of the output. Uses self.evaluated_coordinates if 
            not supplied.

        Returns
        -------
        podpac.Coordinate
            The coordinates of the output if the node is executed with `coords`
        """

        # Changes here likely will also require changes in shape
        if coords is None:
            coords = self.evaluated_coordinates
        if not isinstance(coords, (Coordinate)):
            coords = Coordinate(coords)

        #if self._trait_values.get("native_coordinates", None) is not None:
        # Switching from _trait_values to hasattr because "native_coordinates"
        # not showing up in _trait_values
        if hasattr(self, "native_coordinates") and self.native_coordinates is not None:
            crds = self.native_coordinates.replace_coords(coords)
        else:
            crds = coords
        return crds

    @common_doc(COMMON_DOC)
    def initialize_output_array(self, init_type='nan', fillval=0, style=None,
                                no_style=False, shape=None, coords=None,
                                dims=None, units=None, dtype=np.float, **kwargs):
        """Initializes the UnitsDataArray with the expected output size.

        Parameters
        ----------
        init_type : str, optional
            {arr_init_type}
        fillval : number/np.ndarray, optional
            {arr_fillval}
        style : podpac.Style, optional
            {arr_style}
        no_style : bool, optional
            {arr_no_style}
        shape : list/tuple, optional
            {arr_shape}
        coords : Coordinate, optional
            {arr_coords}
        dims : None, optional
            {arr_dims}
        units : Unit, optional
            {arr_units}
        dtype : TYPE, optional
            {arr_dtype}
        **kwargs
            {arr_kwargs}

        Returns
        -------
        {arr_return}
        """
        crds = self.get_output_coords(coords).coords
        dims = list(crds.keys())
        return self.initialize_array(init_type, fillval, style, no_style, shape,
                                     crds, dims, units, dtype, **kwargs)
    
    @common_doc(COMMON_DOC)
    def copy_output_array(self, init_type='nan'):
        """Create a copy of the output array, initialized using the specified method.

        Parameters
        ----------
        init_type : str, optional
            {arr_init_type}

        Returns
        -------
        UnitsDataArray
            A copy of the `self.output` array, initialized with the specified method.

        Raises
        ------
        ValueError
            Raises this error if the init_type specified is not supported.
        """
        x = self.output.copy(True)
        shape = x.data.shape

        if init_type == 'empty':
            x.data = np.empty(shape)
        elif init_type == 'nan':
            x.data = np.full(shape, np.nan)
        elif init_type == 'zeros':
            x.data = np.zeros(shape)
        elif init_type == 'ones':
            x.data = np.ones(shape)
        else:
            raise ValueError('Unknown init_type={}'.format(init_type))

        return x

    @common_doc(COMMON_DOC)
    def initialize_coord_array(self, coords, init_type='nan', fillval=0,
                               style=None, no_style=False, units=None,
                               dtype=np.float, **kwargs):
        """Initialize an output array using a podpac.Coordinate object.

        Parameters
        ----------
        coords : podpac.Coordinate
            Coordinates descriping the size of the data array that will be initialized
        init_type : str, optional
            {arr_init_type}
        fillval : number/np.ndarray, optional
            {arr_fillval}
        style : podpac.Style, optional
            {arr_style}
        no_style : bool, optional
            {arr_no_style}
        units : podpac.Unit, optional
            {arr_units}
        dtype : TYPE, optional
            {arr_dtype}
        **kwargs
            {arr_kwargs}

        Returns
        -------
        {arr_return}
        """
        return self.initialize_array(init_type, fillval, style, no_style,
                                     coords.shape, coords.coords, coords.dims,
                                     units, dtype, **kwargs)

    @common_doc(COMMON_DOC)
    def initialize_array(self, init_type='nan', fillval=0, style=None,
                         no_style=False, shape=None, coords=None,
                         dims=None, units=None, dtype=np.float, **kwargs):
        """Initialize output data array

        Parameters
        ----------
        init_type : str, optional
            {arr_init_type}
        fillval : number, optional
            {arr_fillval}
        style : Style, optional
            {arr_style}
        no_style : bool, optional
            {arr_no_style}
        shape : tuple
            {arr_shape}
        coords : dict/list
            {arr_coords}
        dims : list(str)
            {arr_dims}
        units : pint.unit.Unit, optional
            {arr_units}
        dtype : type, optional
            {arr_dtype}
        **kwargs
            {arr_kwargs}

        Returns
        -------
        {arr_return}
            

        Raises
        ------
        ValueError
            Raises this error if the init_type specified is not supported.
        """

        if style is None: style = self.style
        if shape is None: shape = self.shape
        if units is None: units = self.units
        if not isinstance(coords, (dict, OrderedDict, list)): coords = dict(coords)

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
            raise ValueError('Unknown init_type={}'.format(init_type))

        x = UnitsDataArray(data, coords=coords, dims=dims, **kwargs)

        if not no_style:
            x.attrs['layer_style'] = style
        if units is not None:
            x.attrs['units'] = units
        x.attrs['params'] = self._params
        return x

    @property
    def base_ref(self):
        """
        Default pipeline node reference/name in pipeline node definitions

        Returns
        -------
        str
            Name of the node in pipeline definitions
        """
        return self.__class__.__name__


    def base_definition(self):
        """Get the base pipeline definition.

        Returns
        -------
        {definition_return}
        """
        d = OrderedDict()

        if self.__module__ == 'podpac':
            d['node'] = self.__class__.__name__
        elif self.__module__.startswith('podpac.'):
            _, module = self.__module__.split('.', 1)
            d['node'] = '%s.%s' % (module, self.__class__.__name__)
        else:
            d['plugin'] = self.__module__
            d['node'] = self.__class__.__name__
        params = {}
        attrs = {}
        for key, value in self.traits().items():
            if value.metadata.get('param', False):
                params[key] = getattr(self, key)
            if value.metadata.get('attr', False):
                attrs[key] = getattr(self, key)
        if params:
            d['params'] = params
        if attrs:
            d['attrs'] = attrs
        return d

    def get_params(self, params=None):
        """Helper function to update default parameters with runtime parameters. 
        
        Parameters
        -----------
        params: dict
            {execute_params}
            
        Returns
        -------
        dict
            The set of parameters that will be used for the execution of the node.
        """
        p = {}
        for key, value in self.traits().items():
            if value.metadata.get('param', False):
                p[key] = getattr(self, key)
        if params: 
            p.update(params)
        return p

    @property
    def definition(self):
        """
        Pipeline node definition.

        This property is implemented in the primary base nodes (DataSource, Algorithm, and Compositor). Node
        subclasses with additional params or attrs will need to extend this property.

        Returns
        -------
        definition : OrderedDct
            full pipeline definition, including the base_defition and any additional properties

        Raises
        ------
        NotImplementedError
            This needs to be implemented by derived classes

        See Also
        --------
        base_definition
        """

        raise NotImplementedError

    @property
    def pipeline_definition(self):
        """
        Full pipeline definition for this node.

        Returns
        -------
        OrderedDict
            Dictionary-formatted definition of a PODPAC pipeline. 
        """

        from podpac.core.pipeline import make_pipeline_definition
        return make_pipeline_definition(self)

    @property
    def pipeline_json(self):
        """Full pipeline definition for this node in json format

        Returns
        -------
        str
            JSON-formatted definition of a PODPAC pipeline.
            
        Notes
        ------
        This definition can be used to create Pipeline Nodes. It also serves as a light-weight transport mechanism to 
        share algorithms and pipelines, or run code on cloud services. 
        """
        return json.dumps(self.pipeline_definition, indent=4)

    @property
    def pipeline(self):
        """Create a pipeline node from this node

        Returns
        -------
        podpac.Pipeline
            A pipeline node created using the self.pipeline_definition
        """
        from podpac.core.pipeline import Pipeline
        return Pipeline(self.pipeline_definition)

    def get_hash(self, coordinates=None, params=None):
        """Hash used for caching node outputs.

        Parameters
        ----------
        coordinates : None, optional
            {evaluated_coordinates}
        params : None, optional
            {params}

        Returns
        -------
        str
            {hash_return}
        """
        if params is not None:
            # convert to OrderedDict with consistent keys
            if not isinstance(params, OrderedDict):
                params = OrderedDict(sorted(params.items()))

            # convert dict values to OrderedDict with consistent keys
            for key, value in params.items():
                if isinstance(value, dict) and not isinstance(value, OrderedDict):
                    params[key] = OrderedDict(sorted(value.items()))

        return hash((str(coordinates), str(params)))

    @property
    def evaluated_hash(self):
        """Get hash for node after being evaluated

        Returns
        -------
        str
            {hash_return}

        Raises
        ------
        NodeException
            Gets raised if node has not been evaluated
        """
        if self.evaluated_coordinates is None:
            raise NodeException("node not evaluated")

        return self.get_hash(self.evaluated_coordinates, self._params)

    @property
    def latlon_bounds_str(self):
        """Helper property used for naming cached files

        Returns
        -------
        str
            String containing the latitude/longitude bounds of a set of coordinates
        """
        return self.evaluated_coordinates.latlon_bounds_str


    def get_output_path(self, filename, outdir=None):
        """Get the output path where data is cached to disk

        Parameters
        ----------
        filename : str
            Name of the file in the cache.
        outdir : None, optional
            {out_dir}

        Returns
        -------
        str
            Path to location where data is cached
            
        Notes
        ------
        If the output directory doesn't exist, it will be created.
        """
        if outdir is None:
            outdir = settings.CACHE_DIR

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        return os.path.join(outdir, filename)


    def write(self, name, outdir=None, format='pickle'):
        """Write self.output to disk using the specified format

        Parameters
        ----------
        name : str
            Name of the file prefix. The final filename will have <name>_<hash>_<latlon_bounds_str>.<format>
        outdir : None, optional
            {outdir}
        format : str, optional
            The file format. Currently only `pickle` is supported. 
            
        Returns
        --------
        str
            The path of the loaded file

        Raises
        ------
        NotImplementedError
            Raised if an unsupported format is specified
        """
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
        return path

    def load(self, name, coordinates, params, outdir=None):
        """Retrieves a cached output from disk and assigns it to self.output

        Parameters
        ----------
        name : str
            Name of the file prefix.
        coordinates : podpac.Coordinate
            {evaluated_coordinates}
        params : dict
            {params}
        outdir : str, optional
            {outdir}
            
        Returns
        --------
        str
            The path of the loaded file
        """
        filename = '%s_%s_%s.pkl' % (
            name,
            self.get_hash(coordinates, params),
            coordinates.latlon_bounds_str)
        path = self.get_output_path(filename, outdir=outdir)

        with open(path, 'rb') as f:
            self.output = cPickle.load(f)
        return path

    def get_image(self, format='png', vmin=None, vmax=None):
        """Return a base64-encoded image of the output

        Parameters
        ----------
        format : str, optional
            Default is 'png'. Type of image. 
        vmin : number, optional
            Minimum value of colormap
        vmax : vmax, optional
            Maximum value of colormap

        Returns
        -------
        str
            Base64 encoded image. 
        """
        matplotlib.use('agg')
        from matplotlib.image import imsave

        data = self.output.data.squeeze()

        if vmin is None or np.isnan(vmin):
            vmin = np.nanmin(data)
        if vmax is None or np.isnan(vmax):
            vmax = np.nanmax(data)
        if vmax == vmin:
            vmax += 1e-16

        c = (data - vmin) / (vmax - vmin)
        i = matplotlib.cm.viridis(c, bytes=True)
        im_data = BytesIO()
        imsave(im_data, i, format=format)
        im_data.seek(0)
        return base64.b64encode(im_data.getvalue())

    @property
    def cache_dir(self):
        """Return the directory used for caching

        Returns
        -------
        str
            Path to the default cache directory
        """
        basedir = settings.CACHE_DIR
        subdir = str(self.__class__)[8:-2].split('.')
        dirs = [basedir] + subdir
        return os.path.join(*dirs)

    def cache_path(self, filename):
        """Return the cache path for the file

        Parameters
        ----------
        filename : str
            Name of the cached file

        Returns
        -------
        str
            Path to the cached file
        """
        pre = str(self.source).replace('/', '_').replace('\\', '_').replace(':', '_')
        return os.path.join(self.cache_dir, pre  + '_' + filename)

    def cache_obj(self, obj, filename):
        """Cache the input object using the given filename

        Parameters
        ----------
        obj : object
            Object to be cached to disk
        filename : str
            File name for the object to be cached
        """
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
        """Retreive an object from cache

        Parameters
        ----------
        filename : str
            File name of object to be retrieved from cache

        Returns
        -------
        object
            Object loaded from cache
        """
        path = self.cache_path(filename)
        if settings.S3_BUCKET_NAME is None or not settings.CACHE_TO_S3:
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
        """Helper function to clear disk cache.

        WARNING: This function will permanently delete cached values
        
        Parameters
        ----------
        attr : str, optional
            Default '*'. Specific attribute to be cleared for specific
            instance of this Node. By default all attributes are cleared.
        node_cache : bool, optional
            Default False. If True, will ignore `attr` and clear all attributes
            for all variants/instances of this Node.
        all_cache : bool, optional
            Default False. If True, will clear the entire podpac cache.
        """
        if all_cache:
            shutil.rmtree(settings.CACHE_DIR)
        elif node_cache:
            shutil.rmtree(self.cache_dir)
        else:
            for f in glob.glob(self.cache_path(attr)):
                os.remove(f)

