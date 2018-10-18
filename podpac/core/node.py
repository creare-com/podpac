"""
Node Summary
"""

from __future__ import division, print_function, absolute_import

import os
import glob
import shutil
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
from podpac.core.units import Units, UnitsDataArray, create_data_array
from podpac.core.utils import common_doc
from podpac.core.coordinates import Coordinates
from podpac.core.style import Style

COMMON_NODE_DOC = {
    'requested_coordinates': 
        '''The set of coordinates requested by a user. The Node will be evaluated using these coordinates.''',
    'eval_output': 
        '''Default is None. Optional input array used to store the output data. When supplied, the node will not 
            allocate its own memory for the output array. This array needs to have the correct dimensions and 
            coordinates.''',
    'eval_method': 
        '''Default is None. How the node will be evaluated: serial, parallel, on aws, locally, etc. Currently only
           local evaluation is supported.''',
    'eval_return': 
        '''UnitsDataArray
            Unit-aware xarray DataArray containing the results of the node evaluation.''',
    'hash_return': 'A unique hash capturing the coordinates and parameters used to evaluate the node. ',
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
    """ Summary """
    pass

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
    requested_coordinates : podpac.Coordinates
        {requested_coordinates}
        This attribute stores the coordinates that were last used to evaluate the node.
    implicit_pipeline_evaluation : Bool
        Flag indicating if nodes as part of a pipeline should be automatically evaluated when
        the root node is evaluated. This attribute is planned for deprecation in the future.
    node_defaults : dict
        Dictionary of defaults values for attributes of a Node. 
    output : podpac.UnitsDataArray
        Output data from the last evaluation of the node. 
    style : podpac.Style
        Object discribing how the output of a node should be displayed. This attribute is planned for deprecation in the
        future.
    units : podpac.Units
        The units of the output data, defined using the pint unit registry `podpac.units.ureg`.
    """

    units = Units(default_value=None, allow_none=True)
    dtype = tl.Any(default_value=float)
    cache_type = tl.Enum([None, 'disk', 'ram'], allow_none=True)
    node_defaults = tl.Dict(allow_none=True)
    style = tl.Instance(Style)

    @tl.default('style')
    def _style_default(self):
        return Style()
    
    # TODO remove these (at least from public api)
    output = tl.Instance(UnitsDataArray, allow_none=True)
    evaluated = tl.Bool(default_value=False)
    implicit_pipeline_evaluation = tl.Bool(default_value=True)
    requested_coordinates = tl.Instance(Coordinates, allow_none=True)

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
    def eval(self, coordinates, output=None, method=None):
        """
        Evaluate the node at the given coordinates.

        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}
        method : str, optional
            {eval_method}
        
        Returns
        -------
        output : {eval_return}
        """

        raise NotImplementedError

    def eval_group(self, group, method=None):
        """
        Evaluate the node for each of the coordinates in the group.
        
        Parameters
        ----------
        group : podpac.CoordinatesGroup
            Group of coordinates to evaluate.
        method : str, optional
            {eval_method}

        Returns
        -------
        outputs : list
            evaluation output, list of UnitsDataArray objects
        """

        return [self.eval(coords, method=method) for coords in group]

    def find_coordinates(self):
        """
        Get all available native coordinates for the Node. Implemented in child classes.

        Returns
        -------
        coord_list : list
            list of available coordinates (Coordinates objects)
        """

        raise NotImplementedError

    @common_doc(COMMON_DOC)
    def create_output_array(self, coords, data=np.nan, **kwargs):
        """
        Initialize an output data array

        Parameters
        ----------
        coords : podpac.Coordinates
            {arr_coords}
        data : None, number, or array-like (optional)
            {arr_init_type}
        **kwargs
            {arr_kwargs}

        Returns
        -------
        {arr_return}
        """

        attrs = {'layer_style': self.style, 'units': self.units}
        return create_data_array(coords, data=data, dtype=self.dtype, attrs=attrs, **kwargs)

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
        attrs = {}
        for key, value in self.traits().items():
            if value.metadata.get('attr', False):
                attrs[key] = getattr(self, key)
        if attrs:
            d['attrs'] = attrs
        return d

    @property
    def definition(self):
        """
        Pipeline node definition.

        This property is implemented in the primary base nodes (DataSource, Algorithm, and Compositor). Node
        subclasses with additional attrs will need to extend this property.

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

    def get_hash(self, coordinates=None):
        """Hash used for caching node outputs.

        Parameters
        ----------
        coordinates : None, optional
            {requested_coordinates}

        Returns
        -------
        str
            {hash_return}
        """
        
        # TODO this needs to include the tagged node attrs
        return hash(str(coordinates))

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
        if self.requested_coordinates is None:
            raise NodeException("node not evaluated")

        return self.get_hash(self.requested_coordinates)

    @property
    def latlon_bounds_str(self):
        """Helper property used for naming cached files

        Returns
        -------
        str
            String containing the latitude/longitude bounds of a set of coordinates
        """
        return self.requested_coordinates.latlon_bounds_str


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

    def load(self, name, coordinates, outdir=None):
        """Retrieves a cached output from disk and assigns it to self.output

        Parameters
        ----------
        name : str
            Name of the file prefix.
        coordinates : podpac.Coordinates
            {requested_coordinates}
        outdir : str, optional
            {outdir}
            
        Returns
        --------
        str
            The path of the loaded file
        """
        filename = '%s_%s_%s.pkl' % (name, self.get_hash(coordinates), coordinates.latlon_bounds_str)
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

