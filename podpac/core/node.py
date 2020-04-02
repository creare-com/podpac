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
from podpac.core.utils import cached_property
from podpac.core.utils import trait_is_defined
from podpac.core.utils import _get_query_params_from_url, _get_from_url, _get_param
from podpac.core.coordinates import Coordinates
from podpac.core.style import Style
from podpac.core.cache import CacheCtrl, get_default_cache_ctrl, make_cache_ctrl, S3CacheStore, DiskCacheStore
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

    # list of attribute names, used by __repr__ and __str__ to display minimal info about the node
    # e.g. data sources use ['source']
    _repr_keys = []

    @tl.default("outputs")
    def _default_outputs(self):
        return None

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

        # Call traitlets constructor
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

    @property
    def attrs(self):
        """List of node attributes"""
        return [name for name in self.traits() if self.trait_metadata(name, "attr")]

    @property
    def _repr_info(self):
        keys = self._repr_keys.copy()
        if self.trait_is_defined("output") and self.output is not None:
            if "output" not in keys:
                keys.append("output")
        elif self.trait_is_defined("outputs") and self.outputs is not None:
            if "outputs" not in keys:
                keys.append("outputs")
        return ", ".join("%s=%s" % (key, repr(getattr(self, key))) for key in keys)

    def __repr__(self):
        return "<%s(%s)>" % (self.__class__.__name__, self._repr_info)

    def __str__(self):
        return "<%s(%s) attrs: %s>" % (self.__class__.__name__, self._repr_info, ", ".join(self.attrs))

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
    def _base_definition(self):
        d = OrderedDict()

        # node and plugin
        if self.__module__ == "podpac":
            d["node"] = self.__class__.__name__
        elif self.__module__.startswith("podpac."):
            _, module = self.__module__.split(".", 1)
            d["node"] = "%s.%s" % (module, self.__class__.__name__)
        else:
            d["plugin"] = self.__module__
            d["node"] = self.__class__.__name__

        # attrs/inputs
        attrs = {}
        inputs = {}
        for name in self.attrs:
            value = getattr(self, name)

            if (
                isinstance(value, Node)
                or (isinstance(value, (list, tuple, np.ndarray)) and all(isinstance(elem, Node) for elem in value))
                or (isinstance(value, dict) and all(isinstance(elem, Node) for elem in value.values()))
            ):
                inputs[name] = value
            else:
                attrs[name] = value

        if "units" in attrs and attrs["units"] is None:
            del attrs["units"]

        if "outputs" in attrs and attrs["outputs"] is None:
            del attrs["outputs"]

        if "output" in attrs and attrs["output"] is None:
            del attrs["output"]

        if attrs:
            d["attrs"] = attrs

        if inputs:
            d["inputs"] = inputs

        # style
        if self.style != Style() and self.style.definition:
            d["style"] = self.style.definition

        return d

    @cached_property
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
                if node == n:
                    return ref

            # get base definition
            d = node._base_definition

            if "inputs" in d:
                # sort and shallow copy
                d["inputs"] = OrderedDict([(key, d["inputs"][key]) for key in sorted(d["inputs"].keys())])

                # replace nodes with references, adding nodes depth first
                for key, value in d["inputs"].items():
                    if isinstance(value, Node):
                        d["inputs"][key] = add_node(value)
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        d["inputs"][key] = [add_node(item) for item in value]
                    elif isinstance(value, dict):
                        d["inputs"][key] = {k: add_node(v) for k, v in value.items()}
                    else:
                        raise TypeError("Invalid input '%s' of type '%s': %s" % (key, type(value)))

            if "attrs" in d:
                # sort and shallow copy
                d["attrs"] = OrderedDict([(key, d["attrs"][key]) for key in sorted(d["attrs"].keys())])

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

        # add top level node
        add_node(self)

        # finalize, verify serializable, and return
        definition = OrderedDict(zip(refs, definitions))
        json.dumps(definition, cls=JSONEncoder)
        return definition

    @cached_property
    def json(self):
        """definition for this node in json format

        Returns
        -------
        str
            JSON-formatted node definition.
        """
        return json.dumps(self.definition, separators=(",", ":"), cls=JSONEncoder)

    @cached_property
    def json_pretty(self):
        return json.dumps(self.definition, indent=4, cls=JSONEncoder)

    @cached_property
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

            for k, v in d.get("inputs", {}).items():
                kwargs[k] = _lookup_input(nodes, name, v)

            for k, v in d.get("lookup_attrs", {}).items():
                kwargs[k] = _lookup_attr(nodes, name, v)

            if "style" in d:
                kwargs["style"] = Style.from_definition(d["style"])

            for k in d:
                if k not in ["node", "inputs", "attrs", "lookup_attrs", "plugin", "style"]:
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


def _lookup_input(nodes, name, value):
    # containers
    if isinstance(value, list):
        return [_lookup_input(nodes, name, elem) for elem in value]

    if isinstance(value, dict):
        return {k: _lookup_input(nodes, name, v) for k, v in value.items()}

    # node reference
    if not isinstance(value, str):
        raise ValueError(
            "Invalid definition for node '%s': invalid reference '%s' of type '%s' in inputs"
            % (name, value, type(value))
        )

    if not value in nodes:
        raise ValueError(
            "Invalid definition for node '%s': reference to nonexistent node '%s' in inputs" % (name, value)
        )

    node = nodes[value]

    # copy in debug mode
    if settings["DEBUG"]:
        node = deepcopy(node)

    return node


def _lookup_attr(nodes, name, value):
    # containers
    if isinstance(value, list):
        return [_lookup_attr(nodes, name, elem) for elem in value]

    if isinstance(value, dict):
        return {_k: _lookup_attr(nodes, name, v) for k, v in value.items()}

    if not isinstance(value, str):
        raise ValueError(
            "Invalid definition for node '%s': invalid reference '%s' of type '%s' in lookup_attrs"
            % (name, value, type(value))
        )

    # node
    elems = value.split(".")
    if elems[0] not in nodes:
        raise ValueError(
            "Invalid definition for node '%s': reference to nonexistent node '%s' in lookup_attrs" % (name, elems[0])
        )

    # subattrs
    attr = nodes[elems[0]]
    for n in elems[1:]:
        if not hasattr(attr, n):
            raise ValueError(
                "Invalid definition for node '%s': reference to nonexistent attribute '%s' in lookup_attrs value '%s"
                % (name, n, value)
            )
        attr = getattr(attr, n)

    # copy in debug mode
    if settings["DEBUG"]:
        attr = deepcopy(attr)

    return attr


# --------------------------------------------------------#
#  Mixins
# --------------------------------------------------------#


class NoCacheMixin(tl.HasTraits):
    """ Mixin to use no cache by default. """

    cache_ctrl = tl.Instance(CacheCtrl, allow_none=True)

    @tl.default("cache_ctrl")
    def _cache_ctrl_default(self):
        return CacheCtrl([])


class DiskCacheMixin(tl.HasTraits):
    """ Mixin to add disk caching to the Node by default. """

    cache_ctrl = tl.Instance(CacheCtrl, allow_none=True)

    @tl.default("cache_ctrl")
    def _cache_ctrl_default(self):
        # get the default cache_ctrl and addd a disk cache store if necessary
        default_ctrl = get_default_cache_ctrl()
        stores = default_ctrl._cache_stores
        if not any(isinstance(store, DiskCacheStore) for store in default_ctrl._cache_stores):
            stores.append(DiskCacheStore())
        return CacheCtrl(stores)


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
