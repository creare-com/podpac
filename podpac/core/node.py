"""
Node Summary
"""

from __future__ import division, print_function, absolute_import

import os
import re
from collections import OrderedDict
import functools
from hashlib import md5 as hash_alg
import json
import numpy as np
import traitlets as tl

from podpac.core.settings import settings
from podpac.core.units import Units, UnitsDataArray, create_data_array
from podpac.core.utils import common_doc
from podpac.core.coordinates import Coordinates
from podpac.core.style import Style
from podpac.core.cache import cache

COMMON_NODE_DOC = {
    'requested_coordinates':
        """The set of coordinates requested by a user. The Node will be evaluated using these coordinates.""",
    'eval_output':
        """Default is None. Optional input array used to store the output data. When supplied, the node will not
            allocate its own memory for the output array. This array needs to have the correct dimensions and
            coordinates.""",
    'eval_return':
        """
        :class:`podpac.UnitsDataArray`
            Unit-aware xarray DataArray containing the results of the node evaluation.
        """,
    'hash_return': 'A unique hash capturing the coordinates and parameters used to evaluate the node. ',
    'outdir': "Optional output directory. Uses :attr:`podpac.settings['CACHE_DIR']` by default",
    'definition_return':
        """
        OrderedDict
            Dictionary containing the location of the Node, the name of the plugin (if required), as well as any
            parameters and attributes that were tagged by children.
        """,
    'arr_init_type':
        """How to initialize the array. Options are:
                nan: uses np.full(..., np.nan) (Default option)
                empty: uses np.empty
                zeros: uses np.zeros()
                ones: uses np.ones
                full: uses np.full(..., fillval)
                data: uses the fillval as the input array
        """,
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
        """
        :class:`podpac.UnitsDataArray`
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
    cache_output: bool
        Default if True. Should the node's output be cached? 
    cache_update: bool
        Default is False. Should the node's cached output be updated from the source data? 
    cache_type : [None, 'disk', 'ram']
        How the output of the nodes should be cached. By default, outputs are not cached.
    cache_ctrl: :class:`podpac.core.cache.cache.CacheCtrl`
        Class that controls caching. If not provided, uses default based on cache_type.
    dtype : type
        The numpy datatype of the output. Currently only ``float`` is supported.
    node_defaults : dict
        Dictionary of defaults values for attributes of a Node.
    style : :class:`podpac.Style`
        Object discribing how the output of a node should be displayed. This attribute is planned for deprecation in the
        future.
    units : :class:`podpac.Units`
        The units of the output data, defined using the pint unit registry `podpac.units.ureg`.

    Notes
    -----
    Additional attributes are available for debugging after evaluation, including:
     * ``_requested_coordinates``: the requested coordinates of the most recent call to eval
     * ``_output``: the output of the most recent call to eval
    """

    units = Units(default_value=None, allow_none=True)
    dtype = tl.Any(default_value=float)
    cache_output = tl.Bool(True)
    cache_update = tl.Bool(False)
    cache_type = tl.Enum([None, 'disk', 'ram'], allow_none=True)
    cache_ctrl = tl.Instance(cache.CacheCtrl, allow_none=True)

    @tl.default('cache_ctrl')
    def _cache_ctrl_default(self):
        if self.cache_type is None:
            return None
        elif self.cache_type == 'ram':
            raise NotImplementedError('Cachetype RAM has not been implemented')
        elif self.cache_type == 'disk':
            store = cache.DiskCacheStore(root_cache_dir_path=settings['CACHE_DIR'])
            ctrl = cache.CacheCtrl(cache_stores=[store])

        return ctrl
    @tl.observe('cache_type')
    def _cache_type_changed(self, change):
        self.cache_ctrl = self._cache_ctrl_default()

    node_defaults = tl.Dict(allow_none=True)
    style = tl.Instance(Style)

    @tl.default('style')
    def _style_default(self):
        return Style()

    # debugging
    _requested_coordinates = tl.Instance(Coordinates, allow_none=True)
    _output = tl.Instance(UnitsDataArray, allow_none=True)
    _from_cache = tl.Bool(allow_none=True, default_value=None)

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
            if 'lookup_source' in d:
                d['lookup_source'] = add_node(d['lookup_source'])
            if 'lookup_attrs' in d:
                for key, attr_node in d['lookup_attrs'].items():
                    d['lookup_attrs'][key] = add_ode(input_node)
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
        return hash_alg(self.json.encode('utf-8')).hexdigest()

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

        return self.cache_ctrl.get(self, key, coordinates=coordinates)

    def put_cache(self, data, key, coordinates=None, overwrite=False, raise_no_cache_exception=True):
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
        overwrite : bool, optional
            Overwrite existing data, default False
        raise_no_cache_exception: bool, optional
            Raises a NodeException if trying to put data to the cache, but no cache is available.

        Raises
        ------
        NodeException
            Cached data already exists (and overwrite is False)
        NodeException
            No cache_ctrl available and raise_no_cache_exception is True
        """

        if not overwrite and self.has_cache(key, coordinates=coordinates):
            raise NodeException("Cached data already exists for key '%s' and coordinates %s" % (key, coordinates))
        if self.cache_ctrl is None:
            if raise_no_cache_exception:
                raise NodeException('Trying to cache data but no cache_ctrl available. Specify cache_type.')
            else:
                return
        
        self.cache_ctrl.put(self, data, key, coordinates=coordinates, update=overwrite)

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
        if self.cache_ctrl is None:
            return False
        return self.cache_ctrl.has(self, key, coordinates=coordinates)

    def rem_cache(self, key, coordinates=None, mode=None, all_cache=False):
        """
        Clear cached data for this node.

        Parameters
        ----------
        key : str, optional
            Delete cached objects with this key. If `'*'`, cached data is deleted for all keys.
        coordinates : podpac.Coordinates, str, optional
            Delete cached objects for these coordinates. If `'*'`, cached data is deleted for all coordinates, including
            coordinate-independent data.
        mode: str, optional
            Specify which cache stores are affected. 
        all_cache: bool, optional
            Default is False. If True, deletes all of the cache.

        See Also
        ---------
        `podpac.core.cache.cache.CacheCtrl.rem`
        """
        if self.cache_ctrl is None:
            return 
        if all_cache:
            self.cache_ctrl.rem()
        else:
            self.cache_ctrl.rem(self, key=key, coordinates=coordinates, mode=mode)


#--------------------------------------------------------#
#  Decorators
#--------------------------------------------------------#

def node_eval(fn):
    """
    Decorator for Node eval methods that handles caching and a user provided output argument.

    fn : function
        Node eval method to wrap

    Returns
    -------
    wrapper : function
        Wrapped node eval method
    """

    cache_key = 'output'

    @functools.wraps(fn)
    def wrapper(self, coordinates, output=None):
        if settings['debug']:
            self._requested_coordinates = coordinates
        key = cache_key
        cache_coordinates = coordinates.transpose(*sorted(coordinates.dims)) # order agnostic caching
        if self.has_cache(key, cache_coordinates) and not self.cache_update:
            data = self.get_cache(key, cache_coordinates)
            if output is not None:
                order = [dim for dim in output.dims if dim not in data.dims] + list(data.dims)
                output.transpose(*order)[:] = data
            self._from_cache = True
        else:
            data = fn(self, coordinates, output=output,)
            if self.cache_output:
                self.put_cache(data, key, cache_coordinates, overwrite=self.cache_update,
                               raise_no_cache_exception=False)
            self._from_cache = False

        # transpose data to match the dims order of the requested coordinates
        order = [dim for dim in coordinates.dims if dim in data.dims]
        data = data.transpose(*order)

        if settings['debug']:
            self._output = data

        return data
    return wrapper

def cache_func(key, depends=None, raise_no_cache_exception=False):
    """
    Decorating for caching a function's output based on a key.

    Parameters
    -----------
    key: str
        Key used for caching.
    depends: str, list, traitlets.All (optional)
        Default is None. Any traits that the cached property depends on. The cached function may NOT
        change the value of any of these dependencies (this will result in a RecursionError)
    raise_no_cache_exception: bool, optional
            Raises a NodeException if trying to put data to the cache, but no cache is available.


    Notes
    -----
    This decorator cannot handle function input parameters.
    
    If the function uses any tagged attributes, these will essentially operate like dependencies
    because the cache key changes based on the node definition, which is affected by tagged attributes.

    Examples
    ----------
    >>> from podpac import Node
    >>> from podpac.core.node import cache_func
    >>> import traitlets as tl
    >>> class MyClass(Node):
           value = tl.Int(0)
           @cache_func('add')
           def add_value(self):
               self.value += 1
               return self.value
           @cache_func('square', depends='value')
           def square_value_depends(self):
               return self.value

    >>> n = MyClass()
    >>> n.add_value()  # The function as defined is called
    1
    >>> n.add_value()  # The function as defined is called again, since we have no caching specified
    2
    >>> n.cache_type = 'disk'
    >>> n.add_value()  # The function as defined is called again, and the value is stored to disk
    3
    >>> n.add_value()  # The value is retrieved from disk, note the change in n.value is not captured
    3
    >>> n.square_value_depends()  # The function as defined is called, and the value is stored to disk
    16
    >>> n.square_value_depends()  # The value is retrieved from disk
    16
    >>> n.value += 1
    >>> n.square_value_depends()  # The function as defined is called, and the value is stored to disk. Note the change in n.value is captured.
    25
    """
    # This is the actual decorator which will be evaluated and returns the wrapped function
    def cache_decorator(func):
        # This is the initial wrapper that sets up the observations
        @functools.wraps(func)
        def cache_wrapper(self):
            # This is the function that updates the cached based on observed traits
            def cache_updator(change):
                # print("Updating value on self:", id(self))
                out = func(self)
                self.put_cache(out, key, overwrite=True, raise_no_cache_exception=raise_no_cache_exception)
            
            if depends:
                # This sets up the observer on the dependent traits
                # print ("setting up observer on self: ", id(self))
                self.observe(cache_updator, depends)
                # Since attributes could change on instantiation, anything we previously 
                # stored is likely out of date. So, force and update to the cache.
                cache_updator(None)

            # This is the final wrapper the continues to fetch data from cache
            # after the observer has been set up.
            @functools.wraps(func)
            def cached_function():
                try: 
                    out = self.get_cache(key)
                except NodeException:
                    out = func(self)
                    self.put_cache(out, key, raise_no_cache_exception=raise_no_cache_exception)
                return out
            
            # Since this is the first time the function is run, set the new wrapper 
            # on the class instance so that the current function won't be called again
            # (which would set up an additional observer)
            setattr(self, func.__name__, cached_function)
            
            # Return the value on the first run
            return cached_function()
        return cache_wrapper
    return cache_decorator
