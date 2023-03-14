from podpac.core.node import Node
from podpac.core.utils import NodeTrait
from podpac.core.utils import trait_is_defined
import traitlets as tl
from podpac.core.coordinates import Coordinates


class CachingNode(Node):
    """
    A node that caches the output of another node.

    Attributes
    ----------
    source : podpac.Node
        The source node to cache.
    _coordinates : podpac.Coordinates, optional
        The coordinates to use for evaluating the node.
    _relevant_dimensions : list, optional
        The relevant dimensions for caching.
    """

   

    source = NodeTrait(allow_none=True).tag(attr=True, required=True)  # if has a coordinates
    _coordinates = tl.Instance(Coordinates, allow_none=True, default_value=None, read_only=True)
    _relevant_dimensions =tl.Instance(list, allow_none=True, default_value=None)

    @property
    def _from_cache(self):
        return self.source._from_cache
    
    @property
    def coordinates(self):
        return self.source.coordinates

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
        self.cache_ctrl = self.source.cache_ctrl
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

        # Check if coordinates were passed in
        if trait_is_defined(self, "_coordinates") and self._coordinates is not None:
            coordinates=self._coordinates
            self._relevant_dimensions = list(coordinates.dims)

            

        # Get standardized coordinates for caching
        cache_coordinates = coordinates.transpose(*sorted(coordinates.dims)).simplify()

        # Cache the relevant dims
        extra = None
        if self._relevant_dimensions is not None:
            extra = list(set(cache_coordinates.dims) - set(self._relevant_dimensions)) # drop extra dims
            cache_coordinates = cache_coordinates.drop(extra)


        # Check the cache
        if not self.source.force_eval and self.source.cache_output and self.has_cache(item, cache_coordinates):
            data = self.get_cache(item, cache_coordinates)
            if output is not None:
                order = [dim for dim in output.dims if dim not in data.dims] + list(data.dims)
                output.transpose(*order)[:] = data
            self.source._from_cache = True
            if extra is not None:
                return data.transpose(*(coordinates.drop(extra)).dims)
            else:
                return data.transpose(*(coordinates).dims)

        # Evaluate the node
        data = self.source.eval(coordinates, **kwargs)

        # Get relevant dimensions to cache
        self._relevant_dimensions = list(data.dims)
        extra = list(set(cache_coordinates.dims) - set(self._relevant_dimensions)) # drop extra dims

        # Cache the output
        if self.source.cache_output:
            self.put_cache(data, item, cache_coordinates.drop(extra))
        

        return data