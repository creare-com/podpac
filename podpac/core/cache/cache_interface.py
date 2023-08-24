import traitlets as tl
from copy import deepcopy
import json

from podpac.core.node import Node
from podpac.core.utils import NodeTrait
from podpac.core.utils import cached_property
from podpac.core.utils import hash_alg
from podpac.core.utils import JSONEncoder


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

    Properties
    -------
    hash()
        The node hash used as a unique idenfier for caching. If "cache_uid" is supplied,
        that will be used. Otherwise the parent class, `Node.hash` property is used.
    """

    source = NodeTrait(allow_none=True).tag(attr=True, required=True)
    cache_uid = tl.Unicode(allow_none=True, default="").tag(attr=True)

    @cached_property
    def hash(self):
        """hash for this node, used in caching and to determine equality."""
        if self.cache_uid:
            return self.cache_uid
        return super().hash
