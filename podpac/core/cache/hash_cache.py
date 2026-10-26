from podpac.core.node import NodeDefinitionError, NodeException

import traitlets as tl
from podpac.core.cache import CacheCtrl, get_default_cache_ctrl, make_cache_ctrl
from podpac import settings
from podpac.core.cache.cache_ctrl import _CACHE_STORES
from podpac.core.cache.cache_interface import CacheNode
from podpac.core.cache.utils import CacheException

_CACHE_UNAVAIL = "Cache unavailable, %s (key='%s')"


class HashCache(CacheNode):
    """
    A node that caches the output of another node.

    Attributes
    ----------
    source : podpac.Node
        The source node to cache.
    cache_ctrl: :class:`podpac.core.cache.cache.CacheCtrl`
        Class that controls caching. If not provided, uses default based on settings.
    relevant_dimensions : list, optional
        The relevant dimensions for caching. Other requested dimensions are dropped from the cache key.
        Default None (all requested dimensions).
    """

    relevant_dimensions = tl.List(allow_none=True, default_value=None).tag(attr=True)
    cache_ctrl = tl.Instance(CacheCtrl, allow_none=True)
    _from_cache = tl.Bool(allow_none=True, default_value=False)
    cache_type = tl.Union(
        [tl.List(tl.Enum(_CACHE_STORES.keys())), tl.Enum(_CACHE_STORES.keys())], allow_none=True, default_value=None
    )

    @tl.default("cache_ctrl")
    def _cache_ctrl_default(self):
        if self.cache_type is None:
            return get_default_cache_ctrl()
        return make_cache_ctrl(self.cache_type)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def outputs(self):
        # Explicitly pass through outputs
        return self.source.outputs

    @property
    def coordinates(self):
        # Explicitly pass through coordinates
        return self.source.coordinates

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def find_coordinates(self):
        """
        Get the available coordinates for the Node. For a DataSource, this is just the coordinates.

        Returns
        -------
        coords_list : list
            singleton list containing the coordinates (Coordinates object)
        """
        # Pass through find_coordinates from the source
        return self.source.find_coordinates()

    def eval(self, coordinates, **kwargs):  # noqa: A003
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

        # Deprecated by ZarrCache
        # if trait_is_defined(self, "cache_coordinates") and self.cache_coordinates is not None:
        #     coordinates = self.cache_coordinates.intersect(coordinates)
        #     self.relevant_dimensions = list(coordinates.dims)
        # elif trait_is_defined(self.source, "coordinates") and self.source.coordinates is not None:
        #     # Use self.source.coordinates if not none:
        #     coordinates = self.source.coordinates.intersect(coordinates)
        #     self.relevant_dimensions = list(coordinates.dims)

        # Get standardized coordinates for caching
        to_cache_coords = coordinates.transpose(*sorted(coordinates.dims)).simplify()

        if self.relevant_dimensions is not None:
            extra = list(set(to_cache_coords.dims) - set(self.relevant_dimensions))
            to_cache_coords = to_cache_coords.drop(extra)

        # Check the cache
        if (
            settings["ENABLE_CACHE"]
            and not self.source.force_eval
            and self.source.cache_output
            and self.has_cache(item, to_cache_coords)
        ):
            try:
                data = self.get_cache(item, to_cache_coords)
            except (NodeException, CacheException):
                data = None  # expired or removed by another process since has_cache

            if data is not None:
                if output is not None:
                    order = [dim for dim in output.dims if dim not in data.dims] + list(data.dims)
                    output.transpose(*order)[:] = data
                self._from_cache = True
                self.source._from_cache = self._from_cache
                # the source may drop requested dims from its output
                return data.transpose(*[dim for dim in coordinates.dims if dim in data.dims])

        # Evaluate the node
        data = self.source.eval(coordinates, **kwargs)
        self._from_cache = False
        self.source._from_cache = self._from_cache

        # Cache the output
        if self.source.cache_output:
            self.put_cache(data, item, to_cache_coords, expires=self.expires)

        return data

    def _eval(self, coordinates, output=None, _selector=None):
        pass  # Nothing to do here

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
            raise NodeException(_CACHE_UNAVAIL % (e.args[0], key))

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
            raise NodeException(_CACHE_UNAVAIL % (e.args[0], key))

        if self.cache_ctrl is None:
            return

        if not overwrite and self.has_cache(key, coordinates=coordinates):
            raise NodeException("Cached data already exists for key '%s' and coordinates %s" % (key, coordinates))

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
            raise NodeException(_CACHE_UNAVAIL % (e.args[0], key))

        if self.cache_ctrl is None:
            return False

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
            raise NodeException(_CACHE_UNAVAIL % (e.args[0], key))

        if self.cache_ctrl is None:
            return

        self.cache_ctrl.rem(self, item=key, coordinates=coordinates, mode=mode)
