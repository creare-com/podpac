"""
Node Summary
"""

from __future__ import division, print_function, absolute_import

import re
import functools
import json
import inspect
import importlib
from collections import OrderedDict
from copy import deepcopy
from hashlib import md5 as hash_alg

import numpy as np
import traitlets as tl

from podpac.core.settings import settings
from podpac.core.units import ureg, UnitsDataArray
from podpac.core.utils import common_doc
from podpac.core.utils import JSONEncoder
from podpac.core.utils import trait_is_defined, trait_is_default
from podpac.core.utils import _get_query_params_from_url, _get_from_url, _get_param
from podpac.core.coordinates import Coordinates
from podpac.core.style import Style
from podpac.core.cache import CacheCtrl, get_default_cache_ctrl, S3CacheStore, make_cache_ctrl
from podpac.core.managers.multi_threading import thread_manager


COMMON_NODE_DOC = {
    "requested_coordinates": """The set of coordinates requested by a user. The Node will be evaluated using these coordinates.""",
    "eval_output": """Default is None. Optional input array used to store the output data. When supplied, the node will not
            allocate its own memory for the output array. This array needs to have the correct dimensions,
            coordinates, and coordinate reference system.""",
    "eval_return": """
        :class:`podpac.UnitsDataArray`
            Unit-aware xarray DataArray containing the results of the node evaluation.
        """,
    "hash_return": "A unique hash capturing the coordinates and parameters used to evaluate the node. ",
    "outdir": "Optional output directory. Uses :attr:`podpac.settings['DISK_CACHE_DIR']` by default",
    "definition_return": """
        OrderedDict
            Dictionary containing the location of the Node, the name of the plugin (if required), as well as any
            parameters and attributes that were tagged by children.
        """,
    "arr_init_type": """How to initialize the array. Options are:
                nan: uses np.full(..., np.nan) (Default option)
                empty: uses np.empty
                zeros: uses np.zeros()
                ones: uses np.ones
                full: uses np.full(..., fillval)
                data: uses the fillval as the input array
        """,
    "arr_fillval": "used if init_type=='full' or 'data', default = 0",
    "arr_style": "The style to use for plotting. Uses self.style by default",
    "arr_no_style": "Default is False. If True, self.style will not be assigned to arr.attr['layer_style']",
    "arr_shape": "Shape of array. Uses self.shape by default.",
    "arr_coords": "Input to UnitsDataArray (i.e. an xarray coords dictionary/list)",
    "arr_dims": "Input to UnitsDataArray (i.e. an xarray dims list of strings)",
    "arr_units": "Default is self.units The Units for the data contained in the DataArray.",
    "arr_dtype": "Default is np.float. Datatype used by default",
    "arr_kwargs": "Dictioary of any additional keyword arguments that will be passed to UnitsDataArray.",
    "arr_return": """
        :class:`podpac.UnitsDataArray`
            Unit-aware xarray DataArray of the desired size initialized using the method specified.
        """,
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
        Should the node's output be cached? If not provided or None, uses default based on settings.
    cache_update: bool
        Default is False. Should the node's cached output be updated from the source data?
    cache_ctrl: :class:`podpac.core.cache.cache.CacheCtrl`
        Class that controls caching. If not provided, uses default based on settings.
    dtype : type
        The numpy datatype of the output. Currently only ``float`` is supported.
    style : :class:`podpac.Style`
        Object discribing how the output of a node should be displayed. This attribute is planned for deprecation in the
        future.
    units : str
        The units of the output data. Must be pint compatible.
    outputs : list
        For multiple-output nodes, the names of the outputs. Default is ``None`` for standard nodes.
    output : str
        For multiple-output nodes only, specifies a particular output to evaluate, if desired. Must be one of ``outputs``.

    Notes
    -----
    Additional attributes are available for debugging after evaluation, including:
     * ``_requested_coordinates``: the requested coordinates of the most recent call to eval
     * ``_output``: the output of the most recent call to eval
     * ``_from_cache``: whether the most recent call to eval used the cache
     * ``_multi_threaded``: whether the most recent call to eval was executed using multiple threads
    """

    outputs = tl.List(tl.Unicode, allow_none=True).tag(attr=True)
    output = tl.Unicode(default_value=None, allow_none=True).tag(attr=True)
    units = tl.Unicode(default_value=None, allow_none=True).tag(attr=True)
    style = tl.Instance(Style)

    dtype = tl.Any(default_value=float)
    cache_output = tl.Bool()
    cache_update = tl.Bool(False)
    cache_ctrl = tl.Instance(CacheCtrl, allow_none=True)

    # tl.List does not honor default_value
    outputs.default_value = None

    @tl.default("outputs")
    def _default_outputs(self):
        return self.traits()["outputs"].default_value

    @tl.validate("output")
    def _validate_output(self, d):
        if d["value"] is not None:
            if self.outputs is None:
                raise TypeError("Invalid output '%s' (output must be None for single-output nodes)." % self.output)
            if d["value"] not in self.outputs:
                raise ValueError("Invalid output '%s' (available outputs are %s)" % (self.output, self.outputs))
        return d["value"]

    @tl.default("style")
    def _default_style(self):
        return Style()

    @tl.validate("units")
    def _validate_units(self, d):
        ureg.Unit(d["value"])  # will throw an exception if this is not a valid pint Unit
        return d["value"]

    @tl.default("cache_output")
    def _cache_output_default(self):
        return settings["CACHE_OUTPUT_DEFAULT"]

    @tl.default("cache_ctrl")
    def _cache_ctrl_default(self):
        return get_default_cache_ctrl()

    # debugging
    _requested_coordinates = tl.Instance(Coordinates, allow_none=True)
    _output = tl.Instance(UnitsDataArray, allow_none=True)
    _from_cache = tl.Bool(allow_none=True, default_value=None)
    # Flag that is True if the Node was run multi-threaded, or None if the question doesn't apply
    _multi_threaded = tl.Bool(allow_none=True, default_value=None)

    def __init__(self, **kwargs):
        """ Do not overwrite me """

        # Shortcut for users to make setting the cache_ctrl simpler:
        if "cache_ctrl" in kwargs and isinstance(kwargs["cache_ctrl"], list):
            kwargs["cache_ctrl"] = make_cache_ctrl(kwargs["cache_ctrl"])

        tkwargs = self._first_init(**kwargs)

        # make tagged "readonly" and "attr" traits read_only, and set them using set_trait
        # NOTE: The set_trait is required because this sets the traits read_only at the *class* level;
        #       on subsequent initializations, they will already be read_only.
        with self.hold_trait_notifications():
            for name, trait in self.traits().items():
                if settings["DEBUG"]:
                    trait.read_only = False
                elif trait.metadata.get("readonly") or trait.metadata.get("attr"):
                    if name in tkwargs:
                        self.set_trait(name, tkwargs.pop(name))
                    trait.read_only = True

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

        attrs = {}
        attrs["layer_style"] = self.style
        attrs["crs"] = coords.crs
        if self.units is not None:
            attrs["units"] = ureg.Unit(self.units)
        try:
            attrs["geotransform"] = coords.geotransform
        except (TypeError, AttributeError):
            pass

        return UnitsDataArray.create(coords, data=data, outputs=self.outputs, dtype=self.dtype, attrs=attrs, **kwargs)

    def trait_is_defined(self, name):
        return trait_is_defined(self, name)

    def trait_is_default(self, name):
        return trait_is_default(self, name)

    # -----------------------------------------------------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------------------------------------------------

    @property
    def base_ref(self):
        """
        Default reference/name in node definitions

        Returns
        -------
        str
            Name of the node in node definitions
        """
        return self.__class__.__name__

    @property
    def base_definition(self):
        d = OrderedDict()

        if self.__module__ == "podpac":
            d["node"] = self.__class__.__name__
        elif self.__module__.startswith("podpac."):
            _, module = self.__module__.split(".", 1)
            d["node"] = "%s.%s" % (module, self.__class__.__name__)
        else:
            d["plugin"] = self.__module__
            d["node"] = self.__class__.__name__

        attrs = {}
        lookup_attrs = {}

        for name, trait in self.traits().items():
            if not trait.metadata.get("attr", False) or self.trait_is_default(name):
                continue

            value = getattr(self, name)

            # check serializable
            json.dumps(value, cls=JSONEncoder)

            # use lookup_attrs for nodes
            if (
                isinstance(value, Node)
                or isinstance(value, (list, tuple, np.ndarray))
                and all(isinstance(elem, Node) for elem in value)
                or isinstance(value, dict)
                and all(isinstance(elem, Node) for elem in value.values())
            ):
                lookup_attrs[name] = value
            else:
                attrs[name] = value

        if attrs:
            d["attrs"] = OrderedDict([(key, attrs[key]) for key in sorted(attrs.keys())])

        if lookup_attrs:
            d["lookup_attrs"] = OrderedDict([(key, lookup_attrs[key]) for key in sorted(lookup_attrs.keys())])

        if self.style != Style() and self.style.definition:
            d["style"] = self.style.definition

        return d

    @property
    def definition(self):
        """
        Full node definition.

        Returns
        -------
        OrderedDict
            Dictionary-formatted node definition.
        """

        nodes = []
        refs = []
        definitions = []

        def add_node(node):
            for ref, n in zip(refs, nodes):
                if node.hash == n.hash:
                    return ref

            # get base definition and then replace nodes with references, adding nodes depth first
            d = node.base_definition
            if "lookup_attrs" in d:
                for key, value in d["lookup_attrs"].items():
                    if isinstance(value, Node):
                        d["lookup_attrs"][key] = add_node(value)
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        d["lookup_attrs"][key] = [add_node(item) for item in value]
                    elif isinstance(value, dict):
                        d["lookup_attrs"][key] = {k: add_node(v) for k, v in value.items()}
                    else:
                        raise ValueError("TODO")

            # get base ref and then ensure it is unique
            ref = node.base_ref
            while ref in refs:
                if re.search("_[1-9][0-9]*$", ref):
                    ref, i = ref.rsplit("_", 1)
                    i = int(i)
                else:
                    i = 0
                ref = "%s_%d" % (ref, i + 1)

            nodes.append(node)
            refs.append(ref)
            definitions.append(d)

            return ref

        add_node(self)

        return OrderedDict(zip(refs, definitions))

    @property
    def json(self):
        """definition for this node in json format

        Returns
        -------
        str
            JSON-formatted node definition.
        """
        return json.dumps(self.definition, separators=(",", ":"), cls=JSONEncoder)

    @property
    def json_pretty(self):
        return json.dumps(self.definition, indent=4, cls=JSONEncoder)

    @property
    def hash(self):
        # Style should not be part of the hash
        defn = self.json

        # Note: this ONLY works because the Style node has NO dictionaries as part
        # of its attributes
        hashstr = re.sub(r'"style":\{.*?\},?', "", defn)

        return hash_alg(hashstr.encode("utf-8")).hexdigest()

    def save(self, path):
        """
        Write node to file.

        Arguments
        ---------
        path : str
            path to write to

        See Also
        --------
        load : load podpac Node from file.
        """

        with open(path, "w") as f:
            json.dump(self.definition, f, separators=(",", ":"), cls=JSONEncoder)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.hash == other.hash

    def __ne__(self, other):
        if not isinstance(other, Node):
            return True
        return self.hash != other.hash

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

        if self.cache_ctrl is None or not self.has_cache(key, coordinates=coordinates):
            raise NodeException("cached data not found for key '%s' and coordinates %s" % (key, coordinates))

        return self.cache_ctrl.get(self, key, coordinates=coordinates)

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
        overwrite : bool, optional
            Overwrite existing data, default False

        Raises
        ------
        NodeException
            Cached data already exists (and overwrite is False)
        """

        if self.cache_ctrl is None:
            return

        if not overwrite and self.has_cache(key, coordinates=coordinates):
            raise NodeException("Cached data already exists for key '%s' and coordinates %s" % (key, coordinates))

        with thread_manager.cache_lock:
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

        with thread_manager.cache_lock:
            return self.cache_ctrl.has(self, key, coordinates=coordinates)

    def rem_cache(self, key, coordinates=None, mode=None):
        """
        Clear cached data for this node.

        Parameters
        ----------
        key : str
            Delete cached objects with this key. If `'*'`, cached data is deleted for all keys.
        coordinates : podpac.Coordinates, str, optional
            Default is None. Delete cached objects for these coordinates. If `'*'`, cached data is deleted for all
            coordinates, including coordinate-independent data. If None, will only affect coordinate-independent data.
        mode: str, optional
            Specify which cache stores are affected.


        See Also
        ---------
        `podpac.core.cache.cache.CacheCtrl.clear` to remove ALL cache for ALL nodes.
        """
        if self.cache_ctrl is None:
            return
        self.cache_ctrl.rem(self, key=key, coordinates=coordinates, mode=mode)

    # --------------------------------------------------------#
    #  Class Methods (Deserialization)
    # --------------------------------------------------------#

    @classmethod
    def from_definition(cls, definition):
        """
        Create podpac Node from a dictionary definition.

        Arguments
        ---------
        d : dict
            node definition

        Returns
        -------
        :class:`Node`
            podpac Node

        See Also
        --------
        definition : node definition as a dictionary
        from_json : create podpac node from a JSON definition
        load : create a node from file
        """

        if len(definition) == 0:
            raise ValueError("Invalid definition: definition cannot be empty.")

        # parse node definitions in order
        nodes = OrderedDict()
        for name, d in definition.items():
            if "node" not in d:
                raise ValueError("Invalid definition for node '%s': 'node' property required" % name)

            # get node class
            module_root = d.get("plugin", "podpac")
            node_string = "%s.%s" % (module_root, d["node"])
            module_name, node_name = node_string.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                raise ValueError("Invalid definition for node '%s': no module found '%s'" % (name, module_name))
            try:
                node_class = getattr(module, node_name)
            except AttributeError:
                raise ValueError(
                    "Invalid definition for node '%s': class '%s' not found in module '%s'"
                    % (name, node_name, module_name)
                )

            # parse and configure kwargs
            kwargs = {}
            for k, v in d.get("attrs", {}).items():
                kwargs[k] = v

            for k, v in d.get("lookup_attrs", {}).items():
                if isinstance(v, str):
                    kwargs[k] = _get_subattr(nodes, name, v)
                elif isinstance(v, list):
                    kwargs[k] = [_get_subattr(nodes, name, e) for e in v]
                elif isinstance(v, dict):
                    kwargs[k] = {_k: _get_subattr(nodes, name, _v) for _k, _v in v.items()}
                else:
                    raise ValueError("TODO")

            if "style" in d:
                kwargs["style"] = Style.from_definition(d["style"])

            for k in d:
                if k not in ["node", "attrs", "lookup_attrs", "plugin", "style"]:
                    raise ValueError("Invalid definition for node '%s': unexpected property '%s'" % (name, k))

            nodes[name] = node_class(**kwargs)

        return list(nodes.values())[-1]

    @classmethod
    def from_json(cls, s):
        """
        Create podpac Node from a JSON definition.

        Arguments
        ---------
        s : str
            JSON-formatted node definition

        Returns
        -------
        :class:`Node`
            podpac Node

        See Also
        --------
        json : node definition as a JSON string
        load : create a node from file
        """

        d = json.loads(s, object_pairs_hook=OrderedDict)
        return cls.from_definition(d)

    @classmethod
    def load(cls, path):
        """
        Create podpac Node from file.

        Arguments
        ---------
        path : str
            path to text file containing a JSON-formatted node definition

        Returns
        -------
        :class:`Node`
            podpac Node

        See Also
        --------
        save : save a node to file
        """

        with open(path) as f:
            d = json.load(f, object_pairs_hook=OrderedDict)
        return cls.from_definition(d)

    @classmethod
    def from_url(cls, url):
        """
        Create podpac Node from a WMS/WCS request.

        Arguments
        ---------
        url : str, dict
            The raw WMS/WCS request url, or a dictionary of query parameters

        Returns
        -------
        :class:`Node`
            A full Node with sub-nodes based on the definition of the node from the URL

        Notes
        -------
        The request can specify the PODPAC node by four different mechanism:
        * Direct node name: PODPAC will look for an appropriate node in podpac.datalib
        * JSON definition passed using the 'PARAMS' query string: Need to specify the special LAYER/COVERAGE value of
          "%PARAMS%"
        * By pointing at the JSON definition retrievable with a http GET request:
          e.g. by setting LAYER/COVERAGE value to https://my-site.org/pipeline_definition.json
        * By pointing at the JSON definition retrievable from an S3 bucket that the user has access to:
          e.g by setting LAYER/COVERAGE value to s3://my-bucket-name/pipeline_definition.json
        """
        params = _get_query_params_from_url(url)

        if _get_param(params, "SERVICE") == "WMS":
            layer = _get_param(params, "LAYERS")
        elif _get_param(params, "SERVICE") == "WCS":
            layer = _get_param(params, "COVERAGE")

        d = None
        if layer.startswith("https://"):
            s = _get_from_url(layer)
        elif layer.startswith("s3://"):
            parts = layer.split("/")
            bucket = parts[2]
            key = "/".join(parts[3:])
            s3 = S3CacheStore(s3_bucket=bucket)
            s = s3._load(key)
        elif layer == "%PARAMS%":
            s = _get_param(params, "PARAMS")
        else:
            p = _get_param(params, "PARAMS")
            if p is None:
                p = "{}"
            d = OrderedDict({layer.replace(".", "-"): {"node": layer, "attrs": json.loads(p)}})

        if d is None:
            d = json.loads(s, object_pairs_hook=OrderedDict)

        return cls.from_definition(d)


def _get_subattr(nodes, name, ref):
    refs = ref.split(".")
    try:
        attr = nodes[refs[0]]
        for _name in refs[1:]:
            attr = getattr(attr, _name)
    except (KeyError, AttributeError):
        raise ValueError("Invalid definition for node '%s': reference to nonexistent node/attribute '%s'" % (name, ref))
    if settings["DEBUG"]:
        attr = deepcopy(attr)
    return attr


# --------------------------------------------------------#
#  Decorators
# --------------------------------------------------------#


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

    cache_key = "output"

    @functools.wraps(fn)
    def wrapper(self, coordinates, output=None):
        if settings["DEBUG"]:
            self._requested_coordinates = coordinates
        key = cache_key
        cache_coordinates = coordinates.transpose(*sorted(coordinates.dims))  # order agnostic caching

        if not self.cache_update and self.has_cache(key, cache_coordinates):
            data = self.get_cache(key, cache_coordinates)
            if output is not None:
                order = [dim for dim in output.dims if dim not in data.dims] + list(data.dims)
                output.transpose(*order)[:] = data
            self._from_cache = True
        else:
            data = fn(self, coordinates, output=output)

            # We need to check if the cache now has the key because it is possible that
            # the previous function call added the key with the coordinates to the cache
            if self.cache_output and not (self.has_cache(key, cache_coordinates) and not self.cache_update):
                self.put_cache(data, key, cache_coordinates, overwrite=self.cache_update)
            self._from_cache = False

        # extract single output, if necessary
        # subclasses should extract single outputs themselves if possible, but this provides a backup
        if "output" in data.dims and self.output is not None:
            data = data.sel(output=self.output)

        # transpose data to match the dims order of the requested coordinates
        order = [dim for dim in coordinates.idims if dim in data.dims]
        if "output" in data.dims:
            order.append("output")
        data = data.transpose(*order)

        if settings["DEBUG"]:
            self._output = data

        # Add style information
        data.attrs["layer_style"] = self.style

        return data

    return wrapper


def cache_func(key, depends=None):
    """
    Decorating for caching a function's output based on a key.

    Parameters
    -----------
    key: str
        Key used for caching.
    depends: str, list, traitlets.All (optional)
        Default is None. Any traits that the cached property depends on. The cached function may NOT
        change the value of any of these dependencies (this will result in a RecursionError)


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

    >>> n = MyClass(cache_ctrl=None)
    >>> n.add_value()  # The function as defined is called
    1
    >>> n.add_value()  # The function as defined is called again, since we have specified no caching
    2
    >>> n.cache_ctrl = CacheCtrl([RamCacheStore()])
    >>> n.add_value()  # The function as defined is called again, and the value is stored in memory
    3
    >>> n.add_value()  # The value is retrieved from disk, note the change in n.value is not captured
    3
    >>> n.square_value_depends()  # The function as defined is called, and the value is stored in memory
    16
    >>> n.square_value_depends()  # The value is retrieved from memory
    16
    >>> n.value += 1
    >>> n.square_value_depends()  # The function as defined is called, and the value is stored in memory. Note the change in n.value is captured.
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
                self.put_cache(out, key, overwrite=True)

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
                    self.put_cache(out, key)
                return out

            # Since this is the first time the function is run, set the new wrapper
            # on the class instance so that the current function won't be called again
            # (which would set up an additional observer)
            setattr(self, func.__name__, cached_function)

            # Return the value on the first run
            return cached_function()

        return cache_wrapper

    return cache_decorator
