from __future__ import division, unicode_literals, print_function, absolute_import

import sys

from multiprocessing import Process as mpProcess
from multiprocessing import Queue
import traitlets as tl
import logging

from podpac.core.node import Node
from podpac.core.utils import NodeTrait
from podpac.core.coordinates import Coordinates
from podpac.core.settings import settings

# Set up logging
_log = logging.getLogger(__name__)


def _f(definition, coords, q, outputkw):
    try:
        n = Node.from_json(definition)
        c = Coordinates.from_json(coords)
        o = n.eval(c)
        o._pp_serialize()
        _log.debug("o.shape: {}, output_format: {}".format(o.shape, outputkw))
        if outputkw:
            _log.debug("Saving output results to output format {}".format(outputkw))
            o = o.to_format(outputkw["format"], **outputkw.get("format_kwargs"))
        q.put(o)
    except Exception as e:
        q.put(str(e))


class Process(Node):
    """
    Source node will be evaluated in another process, and it is blocking!
    """

    source = NodeTrait().tag(attr=True)
    output_format = tl.Dict(None, allow_none=True).tag(attr=True)
    timeout = tl.Int(None, allow_none=True)
    block = tl.Bool(True)

    @property
    def outputs(self):
        return self.source.outputs

    def eval(self, coordinates, **kwargs):
        output = kwargs.get("output")
        definition = self.source.json
        coords = coordinates.json

        q = Queue()
        process = mpProcess(target=_f, args=(definition, coords, q, self.output_format))
        process.daemon = True
        _log.debug("Starting process.")
        process.start()
        _log.debug("Retrieving data from queue.")
        o = q.get(timeout=self.timeout, block=self.block)
        _log.debug("Joining.")
        process.join()  # This is blocking!
        _log.debug("Closing.")
        if (sys.version_info.major + sys.version_info.minor / 10.0) >= 3.7:
            process.close()  # New in version Python 3.7
        if isinstance(o, str):
            raise Exception(o)
        if o is None:
            return
        o._pp_deserialize()
        if output is not None:
            output[:] = o.data[:]
        else:
            output = o

        return output
