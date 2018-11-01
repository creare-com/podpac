"""
Node Summary
"""

from __future__ import division, print_function, absolute_import

import os
import re
from collections import OrderedDict
import json
import numpy as np
import traitlets as tl

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
    node_defaults : dict
        Dictionary of defaults values for attributes of a Node.
    style : podpac.Style
        Object discribing how the output of a node should be displayed. This attribute is planned for deprecation in the
        future.
    units : podpac.Units
        The units of the output data, defined using the pint unit registry `podpac.units.ureg`.

    Notes
    -----
    Additional attributes are available for debugging after evaluation, including::
     * _requested_coordinates: the requested coordinates of the most recent call to eval
     * _output: the output of the most recent call to eval
    """

    units = Units(default_value=None, allow_none=True)
    dtype = tl.Any(default_value=float)
    cache_type = tl.Enum([None, 'disk', 'ram'], allow_none=True)
    node_defaults = tl.Dict(allow_none=True)
    style = tl.Instance(Style)

    @tl.default('style')
    def _style_default(self):
        return Style()

    # debugging
    _requested_coordinates = tl.Instance(Coordinates, allow_none=True)
    _output = tl.Instance(UnitsDataArray, allow_none=True)

    # temporary messages
    @property
    def requested_coordinates(self):
        raise AttributeError("The 'requested_coordinates' attribute has been removed"
                             "(_requested_coordinates may be available for debugging)")
    @requested_coordinates.setter
    def requested_coordinates(self, value):
        raise AttributeError("The 'requested_coordinates' attribute has been removed")

    @property
    def output(self):
        raise AttributeError("The 'output' attribute has been removed; use the output returned by eval instead."
                             "('_output' may be available for debugging)")
    @output.setter
    def output(self, value):
        raise AttributeError("The 'output' attribute has been removed.")

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
    def eval(self, coordinates, output=None):
        """
        Evaluate the node at the given coordinates.

        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        output : podpac.UnitsDataArray, optional
            {eval_output}

        Returns
        -------
        output : {eval_return}
        """

        raise NotImplementedError

    def eval_group(self, group):
        """
        Evaluate the node for each of the coordinates in the group.

        Parameters
        ----------
        group : podpac.CoordinatesGroup
            Group of coordinates to evaluate.

        Returns
        -------
        outputs : list
            evaluation output, list of UnitsDataArray objects
        """

        return [self.eval(coords) for coords in group]

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

    # -----------------------------------------------------------------------------------------------------------------
    # Serialization properties
    # -----------------------------------------------------------------------------------------------------------------

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

    @property
    def base_definition(self):
        """
        Pipeline node definition.

        This property is implemented in the primary base nodes (DataSource, Algorithm, and Compositor). Node
        subclasses with additional attrs will need to extend this property.

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
        lookup_attrs = {}

        for key, value in self.traits().items():
            if not value.metadata.get('attr', False):
                continue

            attr = getattr(self, key)

            if isinstance(attr, Node):
                lookup_attrs[key] = attr
            elif isinstance(attr, np.ndarray):
                attrs[key] = attr.tolist()
            elif isinstance(attr, Coordinates):
                attrs[key] = attr.definition
            else:
                try:
                    json.dumps(attr)
                except:
                    raise NodeException("Cannot serialize attr '%s' with type '%s'" % (key, type(attr)))
                else:
                    attrs[key] = attr

        if attrs:
            d['attrs'] = OrderedDict([(key, attrs[key]) for key in sorted(attrs.keys())])

        if lookup_attrs:
            d['lookup_attrs'] = OrderedDict([(key, lookup_attrs[key]) for key in sorted(lookup_attrs.keys())])

        return d

    @property
    def definition(self):
        """
        Full pipeline definition for this node.

        Returns
        -------
        OrderedDict
            Dictionary-formatted definition of a PODPAC pipeline.
        """

        nodes = []
        refs = []
        definitions = []

        def add_node(node):
            if node in nodes:
                return refs[nodes.index(node)]

            # get base definition and then replace nodes with references, adding nodes depth first
            d = node.base_definition
            if 'source' in d:
                if isinstance(d['source'], Node):
                    d['source'] = add_node(d['source'])
                elif isinstance(d['source'], np.ndarray):
                    d['source'] = d['source'].tolist()
            if 'inputs' in d:
                for key, input_node in d['inputs'].items():
                    if input_node is not None:
                        d['inputs'][key] = add_node(input_node)
            if 'sources' in d:
                for i, source_node in enumerate(d['sources']):
                    d['sources'][i] = add_node(source_node)

            # get base ref and then ensure it is unique
            ref = node.base_ref
            while ref in refs:
                if re.search('_[1-9][0-9]*$', ref):
                    ref, i = ref.rsplit('_', 1)
                    i = int(i)
                else:
                    i = 0
                ref = '%s_%d' % (ref, i+1)

            nodes.append(node)
            refs.append(ref)
            definitions.append(d)

            return ref

        add_node(self)

        d = OrderedDict()
        d['nodes'] = OrderedDict(zip(refs, definitions))
        return d

    @property
    def pipeline(self):
        """Create a pipeline node from this node

        Returns
        -------
        podpac.Pipeline
            A pipeline node that wraps this node
        """
        from podpac.core.pipeline import Pipeline
        return Pipeline(definition=self.definition)

    @property
    def json(self):
        """definition for this node in json format

        Returns
        -------
        str
            JSON-formatted definition of a PODPAC pipeline.

        Notes
        ------
        This definition can be used to create Pipeline Nodes. It also serves as a light-weight transport mechanism to
        share algorithms and pipelines, or run code on cloud services.
        """
        return json.dumps(self.definition)

    @property
    def json_pretty(self):
        return json.dumps(self.definition, indent=4)

    @property
    def hash(self):
        return hash(self.json)

    # -----------------------------------------------------------------------------------------------------------------
    # Caching Interface
    # -----------------------------------------------------------------------------------------------------------------

    def get_cache(self, key, coordinates=None):
        """
        Get cached data for this node.

        Parameters
        ----------
        key : str
            Key for the cached data, e.g. 'output'
        coordinates : podpac.Coordinates, optional
            Coordinates for which the cached data should be retrieved. Omit for coordinate-independent data.

        Returns
        -------
        data : any
            The cached data.

        Raises
        ------
        NodeException
            Cached data not found.
        """

        if not self.has_cache(key, coordinates=coordinates):
            raise NodeException("cached data not found for key '%s' and cooordinates %s" % (key, coordinates))

        # return cache.get(self, data, key, coordinates=coordinates)

    def put_cache(self, data, key, coordinates=None, overwrite=False):
        """
        Cache data for this node.

        Parameters
        ----------
        data : any
            The data to cache.
        key : str
            Unique key for the data, e.g. 'output'
        coordinates : podpac.Coordinates, optional
            Coordinates that the cached data depends on. Omit for coordinate-independent data.
        overwrite : bool
            Overwrite existing data, default False

        Raises
        ------
        NodeException
            Cached data already exists (and overwrite is False)
        """

        if not overwrite and self.has_cache(key, coordinates=coordinates):
            raise NodeException("Cached data already exists for key '%s' and coordinates %s" % (key, coordinates))

        # cache.put(self, data, key, coordinates=coordinates, overwrite=overwrite)

    def has_cache(self, key, coordinates=None):
        """
        Check for cached data for this node.

        Parameters
        ----------
        key : str
            Key for the cached data, e.g. 'output'
        coordinates : podpac.Coordinates, optional
            Coordinates for which the cached data should be retrieved. Omit for coordinate-independent data.

        Returns
        -------
        bool
            True if there is cached data for this node, key, and coordinates.
        """

        return False
        # return cache.has(self, data, key, coordinates=coordinates)

    def del_cache(self, key=None, coordinates=None):
        """
        Clear cached data for this node.

        Parameters
        ----------
        key : str, optional
            Delete cached objects with this key. If None, cached data is deleted for all keys.
        coordinates : podpac.Coordinates, optional
            Delete cached objects for these coordinates. If None, cached data is deleted for all coordinates, including
            coordinate-independent data.
        """

        pass
        # return cache.rem(self, data, key, coordinates=coordinates)

    # -----------------------------------------------------------------------------------------------------------------
    # Deprecated methods
    # -----------------------------------------------------------------------------------------------------------------

    def _get_filename(self, name, coordinates):
        return '%s_%s_%s' % (name, self.hash, coordinates.hash)

    def _get_output_path(self, outdir=None):
        if outdir is None:
            outdir = settings.CACHE_DIR
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        return outdir

    def write(self, name, outdir=None, fmt='pickle'):
        """Write the most recent evaluation output to disk using the specified format

        Parameters
        ----------
        name : str
            Name of the file prefix. The final filename will have <name>_<node_hash>_<coordinates_hash>.<ext>
        outdir : None, optional
            {outdir}
        fmt : str
            Output format, default 'pickle'

        Returns
        --------
        str
            The path of the loaded file

        Raises
        ------
        NotImplementedError
            format not yet implemented
        ValueError
            invalid format

        .. deprecated:: 0.2.0
            This method will be removed and replaced by the caching module by version 0.2.0.
        """

        import warnings
        warnings.warn('Node.write will be removed in a later release', DeprecationWarning)

        try:
            import cPickle  # Python 2.7
        except:
            import _pickle as cPickle

        coordinates = self._requested_coordinates
        path = os.path.join(self._get_output_path(outdir=outdir), self._get_filename(name, coordinates=coordinates))

        if fmt == 'pickle':
            path = '%s.pkl' % path
            with open(path, 'wb') as f:
                cPickle.dump(self._output, f)
        elif fmt == 'png':
            raise NotImplementedError("format '%s' not yet implemented" % fmt)
        elif fmt == 'geotif':
            raise NotImplementedError("format '%s' not yet implemented" % fmt)
        else:
            raise ValueError("invalid format, '%s' not recognized" % fmt)

        return path

    def load(self, name, coordinates, outdir=None):
        """Retrieves cached output from disk as though the node has been evaluated

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

        .. deprecated:: 0.2.0
            This method will be removed and replaced by the caching module by version 0.2.0.
        """

        import warnings
        warnings.warn('Node.load will be removed in a later release', DeprecationWarning)

        try:
            import cPickle  # Python 2.7
        except:
            import _pickle as cPickle

        path = os.path.join(self._get_output_path(outdir=outdir), self._get_filename(name, coordinates=coordinates))
        path = '%s.pkl' % path # assumes pickle
        with open(path, 'rb') as f:
            self._output = cPickle.load(f)
        return path

    @property
    def cache_dir(self):
        """Return the directory used for caching

        Returns
        -------
        str
            Path to the default cache directory

        .. deprecated:: 0.2.0
            This method will be removed and replaced by the caching module by version 0.2.0.
        """

        import warnings
        warnings.warn('Node.cache_dir will be removed in a later release', DeprecationWarning)

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

        .. deprecated:: 0.2.0
            This method will be removed and replaced by the caching module by version 0.2.0.
        """

        import warnings
        warnings.warn('Node.cache_path will be removed in a later release', DeprecationWarning)

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

        .. deprecated:: 0.2.0
            This method will be removed and replaced by the caching module by version 0.2.0.
        """

        import warnings
        warnings.warn('Node.cache_obj will be replaced by put_cache in a later release', DeprecationWarning)

        try:
            import cPickle  # Python 2.7
        except:
            import _pickle as cPickle

        try:
            import boto3
        except:
            boto3 = None

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

        .. deprecated:: 0.2.0
            This method will be removed and replaced by the caching module by version 0.2.0.
        """

        import warnings
        warnings.warn('Node.load_cached_obj will be replaced by get_cache in a later release', DeprecationWarning)

        try:
            import cPickle  # Python 2.7
        except:
            import _pickle as cPickle

        try:
            import boto3
        except:
            boto3 = None

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

        .. deprecated:: 0.2.0
            This method will be removed and replaced by the caching module by version 0.2.0.
        """

        import warnings
        warnings.warn('Node.clear_disk_cache will be replaced by del_cache in a later release', DeprecationWarning)

        import glob
        import shutil

        if all_cache:
            shutil.rmtree(settings.CACHE_DIR)
        elif node_cache:
            shutil.rmtree(self.cache_dir)
        else:
            for f in glob.glob(self.cache_path(attr)):
                os.remove(f)
