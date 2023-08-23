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
    uid : str
        A unique identifier for this CacheNode. If no UID is provided during
        initialization, it will default to the hash of the CacheNode.

    Methods
    -------
    _default_uid()
        Default method to generate the 'uid' attribute for the CacheNode,
        which will be the hash of the CacheNode if 'uid' is not provided during initialization.
    """

    source = NodeTrait(allow_none=True).tag(attr=True, required=True)
    cache_uid = tl.Unicode(allow_none=True, default="").tag(attr=True)

    @cached_property
    def hash(self):
        """hash for this node, used in caching and to determine equality."""
        if self.cache_uid:
            return self.cache_uid

        # deepcopy so that the cached definition property is not modified by the deletes below
        d = deepcopy(self.definition)

        # omit version
        if "podpac_version" in d:
            del d["podpac_version"]

        # omit style in every node
        for k in d:
            if "style" in d[k]:
                del d[k]["style"]

        s = json.dumps(d, separators=(",", ":"), cls=JSONEncoder)
        return hash_alg(s.encode("utf-8")).hexdigest()
