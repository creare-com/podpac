from typing import Any, Dict

import traitlets as tl

from podpac.core.node import Node
from podpac.core.utils import NodeTrait
from podpac.core.utils import cached_property
from podpac.core.cache.utils import expiration_timestamp


class CacheNode(Node):
    """
    This CacheNode class extends the Node class and adds caching capabilities.
    It allows a node to cache its outputs, which can significantly speed up
    subsequent computations if the node's output doesn't change.

    Attributes
    ----------
    source : Node or None
        The source node that this CacheNode wraps. The output of the source node
        will be cached by this CacheNode. This attribute is required.
    cache_uid : str
        A unique identifier for this CacheNode. If no UID is provided during
        initialization, the caching will use the node hash.
    expires : float, datetime, timedelta, str
        Expiration date. If timedelta or timedelta string is supplied,
        the expiration date will be calculated from the current time. Floats are interpreted as timestamps.

    Properties
    -------
    hash()
        The node hash used as a unique idenfier for caching. If "cache_uid" is supplied,
        that will be used. Otherwise the parent class, `Node.hash` property is used.
    """

    source = NodeTrait(allow_none=True).tag(attr=True, required=True)
    cache_uid = tl.Unicode(allow_none=True, default="").tag(attr=True)
    expires = tl.Any(allow_none=True, default_value=None)

    @tl.validate("expires")
    def _validate_expires(self, proposal: Dict[str, Any]) -> Any:
        """Validate that `expires` can be parsed by `expiration_timestamp`.

        Parameters
        ----------
        proposal : Dict[str, Any]
            Traitlets proposal, with the new value in 'value'.

        Returns
        -------
        Any
            The unchanged `expires` value.

        Raises
        ------
        ValueError, TypeError
            If the value is not a valid expiration date or delta.
        """
        _ = expiration_timestamp(proposal["value"])
        return proposal["value"]

    @cached_property
    def hash(self):  # noqa: A003
        """hash for this node, used in caching and to determine equality."""
        if self.cache_uid:
            return self.cache_uid
        return super().hash
