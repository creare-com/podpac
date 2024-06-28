"""
Node Summary
"""

from __future__ import division, print_function, absolute_import

import re
import functools
import json
import inspect
import importlib
import warnings
from collections import OrderedDict
from copy import deepcopy
import logging

import numpy as np
import traitlets as tl
import six

import podpac
from podpac.core.settings import settings
from podpac.core.units import ureg, UnitsDataArray
from podpac.core.utils import common_doc
from podpac.core.utils import JSONEncoder
from podpac.core.utils import cached_property
from podpac.core.utils import trait_is_defined
from podpac.core.utils import _get_query_params_from_url, _get_from_url, _get_param
from podpac.core.utils import probe_node
from podpac.core.utils import NodeTrait
from podpac.core.utils import hash_alg
from podpac.core.coordinates import Coordinates
from podpac.core.style import Style
from podpac.core.cache import CacheCtrl, get_default_cache_ctrl, make_cache_ctrl, S3CacheStore, DiskCacheStore
from podpac.core.managers.multi_threading import thread_manager

_logger = logging.getLogger(__name__)

COMMON_NODE_DOC = {
    "requested_coordinates": """The set of coordinates requested by a user. The Node will be evaluated using these coordinates.""",
    "eval_output": """Default is None. Optional input array used to store the output data. When supplied, the node will not
            allocate its own memory for the output array. This array needs to have the correct dimensions,
            coordinates, and coordinate reference system.""",
    "eval_selector": """The selector function is an optimization that enables nodes to only select data needed by an interpolator.
            It returns a new Coordinates object, and an index object that indexes into the `coordinates` parameter
            If not provided, the Coordinates.intersect() method will be used instead.""",
    "eval_return": """
        :class:`podpac.UnitsDataArray`
            Unit-aware xarray DataArray containing the results of the node evaluation.
        """,
    "hash_return": "A unique hash capturing the coordinates and parameters used to evaluate the node. ",
    "outdir": "Optional output directory. Uses :attr:`podpac.settings[.cache_path` by default",
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
    """Base class for exceptions when using podpac nodes"""

    pass


class NodeDefinitionError(NodeException):
    """Raised node definition errors, such as when the definition is circular or is not yet unavailable."""

    pass


@common_doc(COMMON_DOC)
class Node(tl.HasTraits):
    """The base class for all Nodes, which defines the common interface for everything.

    Attributes
    ----------
    cache_output: bool
        Should the node's output be cached? If not provided or None, uses default based on settings
        (CACHE_NODE_OUTPUT_DEFAULT for general Nodes, and CACHE_DATASOURCE_OUTPUT_DEFAULT  for DataSource nodes).
        If True, outputs will be cached and retrieved from cache. If False, outputs will not be cached OR retrieved from cache (even if
        they exist in cache).
    force_eval: bool
        Default is False. Should the node's cached output be updated from the source data? If True it ignores the cache
        when computing outputs but puts results into the cache (thereby updating the cache)
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

    outputs = tl.List(trait=tl.Unicode(), allow_none=True).tag(attr=True)
    output = tl.Unicode(default_value=None, allow_none=True).tag(attr=True)
    units = tl.Unicode(default_value=None, allow_none=True).tag(attr=True, hidden=True)
    style = tl.Instance(Style)

    dtype = tl.Enum([float], default_value=float)
    cache_output = tl.Bool()
    force_eval = tl.Bool(False)
    cache_ctrl = tl.Instance(CacheCtrl, allow_none=True)
    
    base_ref = tl.Unicode()

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
        return settings["CACHE_NODE_OUTPUT_DEFAULT"]

    @tl.default("cache_ctrl")
    def _cache_ctrl_default(self):
        return get_default_cache_ctrl()

    # debugging
    _requested_coordinates = tl.Instance(Coordinates, allow_none=True)
    _output = tl.Instance(UnitsDataArray, allow_none=True)
    _from_cache = tl.Bool(allow_none=True, default_value=None)
    # Flag that is True if the Node was run multi-threaded, or None if the question doesn't apply
    _multi_threaded = tl.Bool(allow_none=True, default_value=None)

    # util
    _definition_guard = False
    _traits_initialized_guard = False

    def __init__(self, **kwargs):
        """Do not overwrite me"""
        

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

        self._traits_initialized_guard = True

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
        """Overwrite this method if a node needs to do any additional initialization after the standard initialization."""
        pass

    @property
    def attrs(self):
        """List of node attributes"""
        return [name for name in self.traits() if self.trait_metadata(name, "attr")]

    @property
    def _repr_info(self):
        keys = self._repr_keys[:]
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
    def eval(self, coordinates, **kwargs):
        """
        Evaluate the node at the given coordinates.

        Parameters
        ----------
        coordinates : podpac.Coordinates
            {requested_coordinates}
        **kwargs: **dict
            Additional key-word arguments passed down the node pipelines, used internally

        Returns
        -------
        output : {eval_return}
        """

        output = kwargs.get("output", None)
        # check crs compatibility
        if output is not None and "crs" in output.attrs and output.attrs["crs"] != coordinates.crs:
            raise ValueError(
                "Output coordinate reference system ({}) does not match".format(output.crs)
                + "request Coordinates coordinate reference system ({})".format(coordinates.crs)
            )

        if settings["DEBUG"]:
            self._requested_coordinates = coordinates
        item = "output"

        # get standardized coordinates for caching
        cache_coordinates = coordinates.transpose(*sorted(coordinates.dims)).simplify()

        if not self.force_eval and self.cache_output and self.has_cache(item, cache_coordinates):
            data = self.get_cache(item, cache_coordinates)
            if output is not None:
                order = [dim for dim in output.dims if dim not in data.dims] + list(data.dims)
                output.transpose(*order)[:] = data
            self._from_cache = True
        else:
            data = self._eval(coordinates, **kwargs)
            if self.cache_output:
                self.put_cache(data, item, cache_coordinates)
            self._from_cache = False

        # extract single output, if necessary
        # subclasses should extract single outputs themselves if possible, but this provides a backup
        if "output" in data.dims and self.output is not None:
            data = data.sel(output=self.output)

        # transpose data to match the dims order of the requested coordinates
        order = [dim for dim in coordinates.xdims if dim in data.dims]
        if "output" in data.dims:
            order.append("output")
        data = data.part_transpose(order)

        if settings["DEBUG"]:
            self._output = data

        # Add style information
        data.attrs["layer_style"] = self.style

        if self.units is not None:
            data.attrs["units"] = self.units

        # Add crs if it is missing
        if "crs" not in data.attrs:
            data.attrs["crs"] = coordinates.crs

        return data

    def _eval(self, coordinates, output=None, _selector=None):
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
        Get all available coordinates for the Node. Implemented in child classes.

        Returns
        -------
        coord_list : list
            list of available coordinates (Coordinates objects)
        """

        raise NotImplementedError

    def get_bounds(self, crs="default"):
        """Get the full available coordinate bounds for the Node.

        Arguments
        ---------
        crs : str
            Desired CRS for the bounds.
            If not specified, the default CRS in the podpac settings is used. Optional.

        Returns
        -------
        bounds : dict
            Bounds for each dimension. Keys are dimension names and values are tuples (min, max).
        crs : str
            The CRS for the bounds.
        """

        if crs == "default":
            crs = podpac.settings["DEFAULT_CRS"]

        bounds = {}
        for coords in self.find_coordinates():
            ct = coords.transform(crs)
            for dim, (lo, hi) in ct.bounds.items():
                if dim not in bounds:
                    bounds[dim] = (lo, hi)
                else:
                    bounds[dim] = (min(lo, bounds[dim][0]), max(hi, bounds[dim][1]))

        return bounds, crs

    @common_doc(COMMON_DOC)
    def create_output_array(self, coords, data=np.nan, attrs=None, outputs=None, **kwargs):
        """
        Initialize an output data array

        Parameters
        ----------
        coords : podpac.Coordinates
            {arr_coords}
        data : None, number, or array-like (optional)
            {arr_init_type}
        attrs : dict
            Attributes to add to output -- UnitsDataArray.create uses the 'crs' portion contained in here
        outputs : list[string], optional
            Default is self.outputs. List of strings listing the outputs
        **kwargs
            {arr_kwargs}

        Returns
        -------
        {arr_return}
        """

        if attrs is None:
            attrs = {}

        if "layer_style" not in attrs:
            attrs["layer_style"] = self.style
        if "crs" not in attrs:
            attrs["crs"] = coords.crs
        if "units" not in attrs and self.units is not None:
            attrs["units"] = ureg.Unit(self.units)
        if "geotransform" not in attrs:
            try:
                attrs["geotransform"] = coords.geotransform
            except (TypeError, AttributeError):
                pass
        if outputs is None:
            outputs = self.outputs
        if outputs == []:
            outputs = None

        return UnitsDataArray.create(coords, data=data, outputs=outputs, dtype=self.dtype, attrs=attrs, **kwargs)

    def trait_is_defined(self, name):
        return trait_is_defined(self, name)

    def probe(self, lat=None, lon=None, time=None, alt=None, crs=None):
        """Evaluates every part of a node / pipeline at a point and records
        which nodes are actively being used.

        Parameters
        ------------
        lat : float, optional
            Default is None. The latitude location
        lon : float, optional
            Default is None. The longitude location
        time : float, np.datetime64, optional
            Default is None. The time
        alt : float, optional
            Default is None. The altitude location
        crs : str, optional
            Default is None. The CRS of the request.

        Returns
        ---------
        dict
            A dictionary that contains the following for each node:
                * "active": bool,   # If the node is being used or not
                * "value": float,   # The value of the node evaluated at that point
                * "inputs": list,   # List of names of input nodes (based on definition)
                * "name": str,      # node.style.name or self.base_ref if the style name is empty
                * "node_hash": str, # The node's hash
        """
        return probe_node(self, lat, lon, time, alt, crs)

    # -----------------------------------------------------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------------------------------------------------

    @tl.default('base_ref')
    def _default_base_ref(self):
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
                or (
                    isinstance(value, (list, tuple, np.ndarray))
                    and (len(value) > 0)
                    and all(isinstance(elem, Node) for elem in value)
                )
                or (
                    isinstance(value, dict)
                    and (len(value) > 0)
                    and all(isinstance(elem, Node) for elem in value.values())
                )
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

        if not type(self.style) is Style and isinstance(self.style, Style):
            d["style_class"] = self.style.__class__.__module__ + "." + self.style.__class__.__name__
        # style
        if self.style.definition:
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

        if getattr(self, "_definition_guard", False):
            raise NodeDefinitionError("node definition has a circular dependency")

        if not getattr(self, "_traits_initialized_guard", False):
            raise NodeDefinitionError("node is not yet fully initialized")

        try:
            self._definition_guard = True

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
            definition["podpac_version"] = podpac.__version__
            json.dumps(definition, cls=JSONEncoder)
            return definition

        finally:
            self._definition_guard = False

    @property
    def json(self):
        """Definition for this node in JSON format."""

        return json.dumps(self.definition, separators=(",", ":"), cls=JSONEncoder)

    @property
    def json_pretty(self):
        """Definition for this node in JSON format, with indentation suitable for display."""

        return json.dumps(self.definition, indent=4, cls=JSONEncoder)

    @cached_property
    def hash(self):
        """hash for this node, used in caching and to determine equality."""

        # deepcopy so that the cached definition property is not modified by the deletes below
        d = deepcopy(self.definition)

        # omit version
        if "podpac_version" in d:
            del d["podpac_version"]

        # omit style in every node
        for k in d:
            if "style" in d[k]:
                del d[k]["style"]
            if "style_class" in d[k]:
                del d[k]["style_class"]

        s = json.dumps(d, separators=(",", ":"), cls=JSONEncoder)
        return hash_alg(s.encode("utf-8")).hexdigest()

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

        try:
            self.definition
        except NodeDefinitionError as e:
            raise NodeException("Cache unavailable, %s (key='%s')" % (e.args[0], key))

        if self.cache_ctrl is None or not self.has_cache(key, coordinates=coordinates):
            raise NodeException("cached data not found for key '%s' and coordinates %s" % (key, coordinates))

        return self.cache_ctrl.get(self, key, coordinates=coordinates)

    def put_cache(self, data, key, coordinates=None, expires=None, overwrite=True):
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
        expires : float, datetime, timedelta
            Expiration date. If a timedelta is supplied, the expiration date will be calculated from the current time.
        overwrite : bool, optional
            Overwrite existing data, default True.

        Raises
        ------
        NodeException
            Cached data already exists (and overwrite is False)
        """

        try:
            self.definition
        except NodeDefinitionError as e:
            raise NodeException("Cache unavailable, %s (key='%s')" % (e.args[0], key))

        if self.cache_ctrl is None:
            return

        if not overwrite and self.has_cache(key, coordinates=coordinates):
            raise NodeException("Cached data already exists for key '%s' and coordinates %s" % (key, coordinates))

        with thread_manager.cache_lock:
            self.cache_ctrl.put(self, data, key, coordinates=coordinates, expires=expires, update=overwrite)

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

        try:
            self.definition
        except NodeDefinitionError as e:
            raise NodeException("Cache unavailable, %s (key='%s')" % (e.args[0], key))

        if self.cache_ctrl is None:
            return False

        with thread_manager.cache_lock:
            return self.cache_ctrl.has(self, key, coordinates=coordinates)

    def rem_cache(self, key, coordinates=None, mode="all"):
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
            Specify which cache stores are affected. Default 'all'.


        See Also
        ---------
        `podpac.core.cache.cache.CacheCtrl.clear` to remove ALL cache for ALL nodes.
        """

        try:
            self.definition
        except NodeDefinitionError as e:
            raise NodeException("Cache unavailable, %s (key='%s')" % (e.args[0], key))

        if self.cache_ctrl is None:
            return

        self.cache_ctrl.rem(self, item=key, coordinates=coordinates, mode=mode)

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

        if "podpac_version" in definition and definition["podpac_version"] != podpac.__version__:
            warnings.warn(
                "node definition version mismatch "
                "(this node was created with podpac version '%s', "
                "but your current podpac version is '%s')" % (definition["podpac_version"], podpac.__version__)
            )

        if len(definition) == 0:
            raise ValueError("Invalid definition: definition cannot be empty.")

        # parse node definitions in order
        nodes = OrderedDict()
        output_node = None
        for name, d in definition.items():
            if name == "podpac_output_node":
                output_node = d
                continue
            if name == "podpac_version":
                continue

            if "node" not in d:
                raise ValueError("Invalid definition for node '%s': 'node' property required" % name)

            _process_kwargs(name, d, definition, nodes)

        # look for podpac_output_node attribute
        if output_node is None:
            return list(nodes.values())[-1]

        if output_node not in nodes:
            raise ValueError(
                "Invalid definition for value 'podpac_output_node': reference to nonexistent node '%s' in lookup_attrs"
                % (output_node)
            )
        return nodes[output_node]

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
        query_params = _get_query_params_from_url(url)

        if _get_param(query_params, "SERVICE") == "WMS":
            layer = _get_param(query_params, "LAYERS")
        elif _get_param(query_params, "SERVICE") == "WCS":
            layer = _get_param(query_params, "COVERAGE")

        d = None
        if layer.startswith("https://"):
            d = _get_from_url(layer).json()
        elif layer.startswith("s3://"):
            parts = layer.split("/")
            bucket = parts[2]
            key = "/".join(parts[3:])
            s3 = S3CacheStore(s3_bucket=bucket)
            s = s3._load(key)
        elif layer == "%PARAMS%":
            s = _get_param(query_params, "PARAMS")
        else:
            p = _get_param(query_params, "PARAMS")
            if p is None:
                p = "{}"
            if not isinstance(p, dict):
                p = json.loads(p)
            return cls.from_name_params(layer, p)

        if d is None:
            d = json.loads(s, object_pairs_hook=OrderedDict)

        return cls.from_definition(d)

    @classmethod
    def from_name_params(cls, name, params=None):
        """
        Create podpac Node from a WMS/WCS request.

        Arguments
        ---------
        name : str
            The name of the PODPAC Node / Layer
        params : dict, optional
            Default is None. Dictionary of parameters to modify node attributes, style, or completely/partially define the node.
            This dictionary can either be a `Node.definition` or `Node.definition['attrs']`. Node, the specified `name` always
            take precidence over anything defined in `params` (e.g. params['node'] won't be used).

        Returns
        -------
        :class:`Node`
            A full Node with sub-nodes based on the definition of the node from the node name and parameters

        """
        layer = name
        p = params

        d = None
        if p is None:
            p = {}
        definition = {}
        # If one of the special names are in the params list, then add params to the root layer
        if "node" in p or "plugin" in p or "style" in p or "attrs" in p:
            definition.update(p)
        else:
            definition["attrs"] = p
        definition.update({"node": layer})  # The user-specified node name ALWAYS takes precidence.
        d = OrderedDict({layer.replace(".", "-"): definition})

        return cls.from_definition(d)

    @classmethod
    def get_ui_spec(cls, help_as_html=False):
        """Get spec of node attributes for building a ui

        Parameters
        ----------
        help_as_html : bool, optional
            Default is False. If True, the docstrings will be converted to html before storing in the spec.

        Returns
        -------
        dict
            Spec for this node that is readily json-serializable
        """
        filter = []
        spec = {"help": cls.__doc__, "module": cls.__module__ + "." + cls.__name__, "attrs": {}, "style": {}}
        # Strip out starting spaces in the help text so that markdown parsing works correctly
        if spec["help"] is None:
            spec["help"] = "No help text to display."
        spec["help"] = spec["help"].replace("\n    ", "\n")

        if help_as_html:
            from numpydoc.docscrape_sphinx import SphinxDocString
            from docutils.core import publish_string

            tmp = SphinxDocString(spec["help"])
            tmp2 = publish_string(str(tmp), writer_name="html")
            slc = slice(tmp2.index(b'<div class="document">'), tmp2.index(b"</body>"))
            spec["help"] = tmp2[slc].decode()

        # find any default values that are defined by function with decorators
        # e.g. using @tl.default("trait_name")
        #            def _default_trait_name(self): ...
        function_defaults = {}
        for attr in dir(cls):
            atr = getattr(cls, attr)
            if not isinstance(atr, tl.traitlets.DefaultHandler):
                continue
            try:
                try:
                    def_val = atr(cls())
                except:
                    def_val = atr(cls)
                if isinstance(def_val, NodeTrait):
                    def_val = def_val.name
                    print("Changing Nodetrait to string")
                # if "NodeTrait" not in str(atr(cls)):
                function_defaults[atr.trait_name] = def_val
            except Exception:
                _logger.warning(
                    "For node {}: Failed to generate default from function for trait {}".format(
                        cls.__name__, atr.trait_name
                    )
                )

        for attr in dir(cls):
            if attr in filter:
                continue
            attrt = getattr(cls, attr)
            if not isinstance(attrt, tl.TraitType):
                continue
            if not attrt.metadata.get("attr", False):
                continue
            type_ = attrt.__class__.__name__

            try:
                schema = getattr(attrt, "_schema")
            except:
                schema = None

            type_extra = str(attrt)
            if type_ == "Union":
                type_ = [t.__class__.__name__ for t in attrt.trait_types]
                type_extra = "Union"
            elif type_ == "Instance":
                type_ = attrt.klass.__name__
                if type_ == "Node":
                    type_ = "NodeTrait"
                type_extra = attrt.klass
            elif type_ == "Dict" and schema is None:
                try:
                    schema = {
                        "key": getattr(attrt, "_key_trait").__class__.__name__,
                        "value": getattr(attrt, "_value_trait").__class__.__name__,
                    }
                except Exception as e:
                    print("Could not find schema for", attrt, " of type", type_)
                    schema = None

            required = attrt.metadata.get("required", False)
            hidden = attrt.metadata.get("hidden", False)
            if attr in function_defaults:
                default_val = function_defaults[attr]
            else:
                default_val = attrt.default()
            if not isinstance(type_extra, str):
                type_extra = str(type_extra)
            try:
                if np.isnan(default_val):
                    default_val = "nan"
            except:
                pass

            if default_val == tl.Undefined:
                default_val = None

            spec["attrs"][attr] = {
                "type": type_,
                "type_str": type_extra,  # May remove this if not needed
                "values": getattr(attrt, "values", None),
                "default": default_val,
                "help": attrt.help,
                "required": required,
                "hidden": hidden,
                "schema": schema,
            }

        try:
            # This returns the
            style_json = json.loads(cls().style.json)  # load the style from the cls
        except:
            style_json = {}

        spec["style"] = style_json  # this does not work, because node not created yet?

        """
        I will manually define generic defaults here. Eventually we may want to
        dig into this and create node specific styling. This will have to be done under each
        node. But may be difficult to add style to each node?

        Example: podpac.core.algorithm.utility.SinCoords.Style ----> returns a tl.Instance
        BUT if I do:
        podpac.core.algorithm.utility.SinCoords().style.json ---> outputs style

        ERROR if no parenthesis are given. So how can this be done without instantiating the class?

        Will need to ask @MPU how to define a node specific style.
        """
        # spec["style"] = {
        #     "name": "?",
        #     "units": "m",
        #     "clim": [-1.0, 1.0],
        #     "colormap": "jet",
        #     "enumeration_legend": "?",
        #     "enumeration_colors": "?",
        #     "default_enumeration_legend": "unknown",
        #     "default_enumeration_color": (0.2, 0.2, 0.2),
        # }

        spec.update(getattr(cls, "_ui_spec", {}))
        return spec


def _lookup_input(nodes, name, value, definition):
    """check if inputs of a node are stored in nodes, if not add them

    Parameters
    -----------
    nodes: OrderedDict
        Keys: Node names (strings)
        Values: Node objects

    name: string
        the Node whose inputs are being checked

    value: string, list, dictionary:
        the Node (or collection of Nodes) which is being looked

    definition: pipeline definition

    Returns
    --------
    node: the node searched for

    Note: this function calles _process_kwargs, which alters nodes by loading a Node if it is not yet in nodes.

    """
    # containers
    if isinstance(value, list):
        return [_lookup_input(nodes, name, elem, definition) for elem in value]

    if isinstance(value, dict):
        return {k: _lookup_input(nodes, name, v, definition) for k, v in value.items()}

    # node reference
    if not isinstance(value, six.string_types):
        raise ValueError(
            "Invalid definition for node '%s': invalid reference '%s' of type '%s' in inputs"
            % (name, value, type(value))
        )
    # node not yet discovered yet
    if not value in nodes:
        # Look for it in the definition items:
        for found_name, d in definition.items():
            if value != found_name:
                continue
            # Load the node into nodes
            _process_kwargs(found_name, d, definition, nodes)

            break

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
        return {k: _lookup_attr(nodes, name, v) for k, v in value.items()}

    if not isinstance(value, six.string_types):
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


def _process_kwargs(name, d, definition, nodes):
    """create a node and add it to nodes

    Parameters
    -----------
    nodes: OrderedDict
        Keys: Node names (strings)
        Values: Node objects

    name: string
        the Node which will be created

    d: the definition of the node to be created

    definition: pipeline definition

    Returns
    --------
    Nothing, but loads the node with name "name" and definition "d" into nodes


    """
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
            "Invalid definition for node '%s': class '%s' not found in module '%s'" % (name, node_name, module_name)
        )

    kwargs = {}
    for k, v in d.get("attrs", {}).items():
        if isinstance(getattr(node_class, k), tl.TraitType) and hasattr(getattr(node_class, k), "klass") and isinstance(v, OrderedDict) and  getattr(node_class, k).klass == Coordinates:
            kwargs[k] = Coordinates.from_definition(v)
        else:
            kwargs[k] = v

    for k, v in d.get("inputs", {}).items():
        kwargs[k] = _lookup_input(nodes, name, v, definition)

    for k, v in d.get("lookup_attrs", {}).items():
        kwargs[k] = _lookup_attr(nodes, name, v)

    if "style" in d:
        if "style_class" in d:
            style_root = module_root
            # style_string  = "%s.%s" % (style_root, d["style_class"])
            style_string  = d["style_class"]
            module_style_name, style_name = style_string.rsplit(".", 1)
            
            try:
                style_module = importlib.import_module(module_style_name)
            except ImportError:
                raise ValueError("Invalid definition for style module '%s': no module found '%s'" % (name, module_style_name))
            try:
                style_class = getattr(style_module, style_name)
            except AttributeError:
                raise ValueError(
                    "Invalid definition for style '%s': style class '%s' not found in style module '%s'" % (name, style_name, module_style_name)
                )
        else:  
            style_class = getattr(node_class, "style", Style)
        if isinstance(style_class, tl.TraitType):
            # Now we actually have to look through the class to see
            # if there is a custom initializer for style
            for attr in dir(node_class):
                atr = getattr(node_class, attr)
                if not isinstance(atr, tl.traitlets.DefaultHandler) or atr.trait_name != "style":
                    continue
                try:
                    style_class = atr(node_class)
                except Exception as e:
                    try:
                        style_class = atr(node_class())
                    except:
                        style_class = style_class.klass
        try:
            kwargs["style"] = style_class.from_definition(d["style"])
        except Exception as e:
            kwargs["style"] = Style.from_definition(d["style"])


    for k in d:
        if k not in ["node", "inputs", "attrs", "lookup_attrs", "plugin", "style", "style_class"]:
            raise ValueError("Invalid definition for node '%s': unexpected property '%s'" % (name, k))

    for k in kwargs.keys():
        if not (hasattr(node_class, k) and isinstance(getattr(node_class, k), tl.TraitType)):
            logging.warn("Node definition has key '{}' that will not be set at node creation: attribute is not of type tl.TraitType".format(k))

    nodes[name] = node_class(**kwargs)
    

# --------------------------------------------------------#
#  Mixins
# --------------------------------------------------------#


class NoCacheMixin(tl.HasTraits):
    """Mixin to use no cache by default."""

    cache_ctrl = tl.Instance(CacheCtrl, allow_none=True)

    @tl.default("cache_ctrl")
    def _cache_ctrl_default(self):
        return CacheCtrl([])


class DiskCacheMixin(tl.HasTraits):
    """Mixin to add disk caching to the Node by default."""

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
