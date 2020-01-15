"""
Module to help farm out computation to multiple workers and save the results in a zarr file.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import logging
import traitlets as tl

from multiprocessing.pool import ThreadPool

from podpac.core.managers.multi_threading import Lock
from podpac.core.node import Node
from podpac.core.utils import NodeTrait

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
        
==== BASE ====
    def eval(self, coordinates, output=None):
        # Make a thread pool to manage queue
        pool = ThreadPool(processes=self.number_of_workers)
        
        if output is None and self.fill_output:
            output = self.create_output_array(coordinates)
        shape = [self.chunks[d] for d in coordinates.dims]
        for coords, slc in coordinates.iterchunks(shape, True):
            out = None
            if self.fill_output:
                out = output[slc]
            with self._lock:
                results = pool.apply_async(self.eval_source, [coords, out])
                o = results.get()
    
        pool.close()
        
        return output
    
    def eval_source(self, coordinates, coordinates_index, out):
        self.source.eval(coordinates, output=out)


class ParallelOutputZarr(Parallel):
    """
    This class assumes that the node has a 'output_format' attribute
    (currently the Lambda Node, and the MultiProcess Node)
    
    """
    zarr_file = tl.Unicode().tag(attr=True)
    fill_output = tl.Bool(False)
    init_file_mode = tl.Unicode("w")
    chunks = tl.Dict()

    def eval(self, coordinates, output=None):
        # initialize zarr file
        zn = Zarr(source=kwargs["source"])
        if self.source.output or self.data_key:
            data_key = self.source.output
            if data_key is None:
                data_key = self.data_key
            data_key = [data_key]
        else:
            data_key = [self.source.outputs]
        zf = zarr.open(zn._get_store(self.source), mode=self.init_file_mode)
        chunks_shape = [self.chunks[d] for d in coordinates.dims]

        for dk in data_key:
            arr = zf.create_dataset(dk, shape=coordinates.shape, chunks=chunks_shape, fill_value=np.nan, overwrite=True)
            arr.attrs["_ARRAY_DIMENSIONS"] = coordinates.dims

        for i, d in coordinates.dims:
            zf.create_dataset(d, shape=coordinates.shape[i], chunks=chunks_shape[i], overwrite=True)

        # eval
        super(ParallelOutputZarr, self).eval(coordinates, output)

    def eval_source(self, coordinates, coordinates_index, out):
        output = dict(
            format="zarr_part",
                      format_kwargs=dict(part=[[s.start, s.stop, s.step] for s in coordinates_index]),
                      store=self.zarr_file,
            mode="a",
                     )
        self.source.output_format = output
