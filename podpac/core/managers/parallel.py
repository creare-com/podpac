"""
Module to help farm out computation to multiple workers and save the results in a zarr file.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import logging
import traitlets as tl
import numpy as np

from multiprocessing.pool import ThreadPool

from podpac.core.managers.multi_threading import Lock
from podpac.core.node import Node
from podpac.core.utils import NodeTrait
from podpac.core.data.file import Zarr

# Optional dependencies
from lazy_import import lazy_module, lazy_class

zarr = lazy_module("zarr")
zarrGroup = lazy_class("zarr.Group")

# Set up logging
_log = logging.getLogger(__name__)

class Parallel(Node):
    """
    For this class to work properly, the source node should return immediately after an eval. 
    """
    source = NodeTrait().tag(attr=True)
    chunks = tl.Dict().tag(attr=True)
    fill_output = tl.Bool(True).tag(attr=True)
    number_of_workers = tl.Int(1000).tag(attr=True)
    _lock = Lock()
        
    def eval(self, coordinates, output=None):
        # Make a thread pool to manage queue
        pool = ThreadPool(processes=self.number_of_workers)
        
        if output is None and self.fill_output:
            output = self.create_output_array(coordinates)
        shape = [self.chunks[d] for d in coordinates.dims]
        results = []
#         inputs = []
        for coords, slc in coordinates.iterchunks(shape, True):
#             inputs.append(coords)
            out = None
            if self.fill_output:
                out = output[slc]
            with self._lock:
                _log.debug("Node eval with coords: {}, {}".format(slc, coords))
                results.append(pool.apply_async(self.eval_source, [coords, slc, out]))

        for i, res in enumerate(results):
#             _log.debug('Waiting for results: {} {}'.format(i, inputs[i]))
            _log.debug('Waiting for results: {}'.format(i))
            o = res.get()
            
        _log.debug('Completed parallel execution.')
        pool.close()
        
        return output
    
    def eval_source(self, coordinates, coordinates_index, out):
        return self.source.eval(coordinates, output=out)
        
class ParallelOutputZarr(Parallel):
    """
    This class assumes that the node has a 'output_format' attribute
    (currently the Lambda Node, and the MultiProcess Node)
    
    """
    zarr_file = tl.Unicode().tag(attr=True)
    dataset = tl.Any()
    fill_output = tl.Bool(False)
    init_file_mode = tl.Unicode("w").tag(attr=True)
    chunks = tl.Dict()
    _shape = tl.Tuple()
    
==== BASE ====
    def eval(self, coordinates, output=None):
        self._shape = coordinates.shape
        # initialize zarr file
        _log.debug("Creating Zarr file.")
        zn = Zarr(source=self.zarr_file, file_mode=self.init_file_mode)
        if self.source.output or getattr(self.source, 'data_key', None):
            data_key = self.source.output
            if data_key is None:
                data_key = self.source.data_key
            data_key = [data_key]
        elif self.source.outputs:
            data_key = [self.source.outputs]
        else:
            data_key = ['data']
            
        zf = zarr.open(zn._get_store(zn.source), mode=self.init_file_mode)
        chunks_shape = [self.chunks[d] for d in coordinates.dims]

        for  dk in data_key:
            arr = zf.create_dataset(dk, shape=coordinates.shape, chunks=chunks_shape, fill_value=np.nan, overwrite=True)
            arr.attrs['_ARRAY_DIMENSIONS'] = coordinates.dims
        
        for i, d in coordinates.dims:
            zf.create_dataset(d, shape=coordinates.shape[i], chunks=chunks_shape[i], overwrite=True)
        
        # TODO: Add layer style attrs and so forth... 
        
        self.dataset = zf
        # eval
        _log.debug("Starting parallel eval.")
        super(ParallelOutputZarr, self).eval(coordinates, output)

    def eval_source(self, coordinates, coordinates_index, out):
        source = Node.from_definition(self.source.definition)
        _log.debug("Creating output format.")
        output = dict(format='zarr_part', 
                      format_kwargs=dict(part=[[s.start, min(s.stop, self._shape[i]), s.step] for i, s in enumerate(coordinates_index)],
                                         store=self.zarr_file,
            mode="a",
                     )
        _log.debug("Finished creating output format.")
        source.set_trait('output_format', output)
        _log.debug("Evaluating node.")
        _log.debug("output: {}, coordinates.shape: {}".format(output, coordinates.shape))
        
        return source.eval(coordinates, out)
        
        
