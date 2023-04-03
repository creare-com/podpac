from podpac.core.node import Node, NodeDefinitionError, NodeException
from podpac.core.utils import NodeTrait
from podpac.core.utils import trait_is_defined
import traitlets as tl
from podpac.core.coordinates import Coordinates
from podpac.core.cache import CacheCtrl, get_default_cache_ctrl, make_cache_ctrl, S3CacheStore, DiskCacheStore
from podpac.core.managers.multi_threading import thread_manager


class CachingNode(Node):
    """
    A node that caches the output of another node.

    Attributes
    ----------
    source : podpac.Node
        The source node to cache.
    cache_ctrl: :class:`podpac.core.cache.cache.CacheCtrl`
        Class that controls caching. If not provided, uses default based on settings.
    cache_coordinates : podpac.Coordinate, optional
        Coordinates that should be used for caching. If not provided. self.source.coordinates will be used, if it exists. Otherwise, use the request coordinates for the cache.  
    _relevant_dimensions : list, optional
        The relevant dimensions for caching.
    """


    source = NodeTrait(allow_none=True).tag(attr=True, required=True)  # if has a coordinates
    cache_coordinates = tl.Instance(Coordinates, allow_none=True, default_value=None, read_only=True)
    _relevant_dimensions =tl.Instance(list, allow_none=True, default_value=None)
    cache_ctrl = tl.Instance(CacheCtrl, allow_none=True)
    _from_cache = tl.Bool(allow_none=True, default_value=False)
    
    
    @property
    def coordinates(self):
        return self.source.coordinates
    
    @tl.default("cache_ctrl")
    def _cache_ctrl_default(self):
        return get_default_cache_ctrl()

    def eval(self, coordinates, **kwargs):
        """
        Evaluate the node at the given coordinates. Check the cache for the output and use it if it exists.
        Otherwise, evaluate the source node and cache the output.

        Parameters
        ----------
        coordinates : podpac.Coordinates
            The coordinates to evaluate the node at.
        kwargs : dict
            Additional keyword arguments to pass to the source node's _eval method.

        Returns
        -------
        podpac.UnitsDataArray
            The output of evaluating the node at the given coordinates.
        """
        
        # Shortcut for users to make setting the cache_ctrl simpler:
        if "cache_ctrl" in kwargs and isinstance(kwargs["cache_ctrl"], list):
            kwargs["cache_ctrl"] = make_cache_ctrl(kwargs["cache_ctrl"])
        
        # Get the output from kwargs, if available
        output = kwargs.get("output", None)

        # Check CRS compatibility
        if output is not None and "crs" in output.attrs and output.attrs["crs"] != coordinates.crs:
            raise ValueError(
                "Output coordinate reference system ({}) does not match".format(output.crs)
                + " request Coordinates coordinate reference system ({})".format(coordinates.crs)
            )

        # Set the item to output
        item = "output"

        # Use self.source.coordinates if not none:
        if self.source.coordinates is not None:
            coordinates = self.cache_coordinates

        # Check if coordinates were passed in
        if trait_is_defined(self, "cache_coordinates") and self.cache_coordinates is not None:
            coordinates=self.cache_coordinates
            self._relevant_dimensions = list(coordinates.dims)

        # Get standardized coordinates for caching
        to_cache_coords = coordinates.transpose(*sorted(coordinates.dims)).simplify()

        # Cache the relevant dims
        extra = None
        if self._relevant_dimensions is not None:
            extra = list(set(to_cache_coords.dims) - set(self._relevant_dimensions)) # drop extra dims
            to_cache_coords = to_cache_coords.drop(extra)


        # Check the cache
        if not self.source.force_eval and self.source.cache_output and self.has_cache(item, to_cache_coords):
            data = self.get_cache(item, to_cache_coords)
            if output is not None:
                order = [dim for dim in output.dims if dim not in data.dims] + list(data.dims)
                output.transpose(*order)[:] = data
            self._from_cache = True
            if extra is not None:
                return data.transpose(*(coordinates.drop(extra)).dims)
            else:
                return data.transpose(*(coordinates).dims)

        # Evaluate the node
        data = self.source.eval(coordinates, **kwargs)
        self._from_cache = False 

        # Get relevant dimensions to cache
        self._relevant_dimensions = list(data.dims)
        extra = list(set(to_cache_coords.dims) - set(self._relevant_dimensions)) # drop extra dims

        # Cache the output
        if self.source.cache_output:
            self.put_cache(data, item, to_cache_coords.drop(extra))
        

        return data
    
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