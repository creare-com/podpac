from __future__ import division, unicode_literals, print_function, absolute_import

import time

from multiprocessing import Process, Queue
import traitlets as tl

from podpac.core.node import Node
from podpac.core.utils import NodeTrait
from podpac.core.coordinates import Coordinates
from podpac.core.settings import settings

DEFAULT_N_PROC = 16

def _f(definition, coords, q, outputkw):
    try:
        n = Node.from_json(definition)
        c = Coordinates.from_json(coords)
        o = n.eval(c)
        if outputkw:
            o.to_format(outputkw['format'], **outputkw['kwargs'])
        o.serialize()
        q.put(o)
    except:
        pass

class NewProcess(Node):
    """
    Source node will be evaluated in another process, and it is blocking!
    """
    source = NodeTrait().tag(attr=True)
    output_format = tl.Dict(None, allow_none=True).tag(attr=True)
    
    def eval(self, coordinates, output=None):
        definition = self.source.json
        coords = coordinates.json
            
        q = Queue()
        process = Process(target=_f, args=(definition, coords, q, self.output_format))
        process.daemon = True
        process.start()
        o = q.get()  # This is blocking!
        process.join()
        process.close()
        o.deserialize()
        return o
        