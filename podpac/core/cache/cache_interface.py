from podpac.core.node import Node
from podpac.core.utils import NodeTrait
import traitlets as tl


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
    uid = tl.Unicode(allow_none=True).tag(attr=True)
    _uid = tl.Unicode()

    @tl.default("_uid")
    def _default_uid(self):
        """
        Returns the default value for the 'uid' attribute, which is the hash of the CacheNode.
        
        Returns
        -------
        str
            The hash of the CacheNode.
        """
        if self.uid:
            return self.uid
        return self.hash

    