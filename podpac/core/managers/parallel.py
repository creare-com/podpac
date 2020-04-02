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
from podpac.core.data.zarr_source import Zarr
from podpac.core.coordinates import Coordinates, merge_dims

# Optional dependencies
from lazy_import import lazy_module, lazy_class

zarr = lazy_module("zarr")
zarrGroup = lazy_class("zarr.Group")

# Set up logging
_log = logging.getLogger(__name__)


class Parallel(Node):
    """
    For this class to work properly, the source node should return immediately after an eval. 
    
    Attributes
    -----------
    chunks: dict
        Dictionary of dimensions and sizes that will be iterated over. If a dimension is not in this dictionary, the
        size of the eval coordinates will be used for the chunk. In this case, it may not be possible to automatically
        set the coordinates of missing dimensions in the final file.
    fill_output: bool
        Default is True. When True, the final results will be assembled and returned to the user. If False, the final
        results should be written to a file by specifying the output_format in a Process or Lambda node. 
        See note below.
    source: podpac.Node
        The source dataset for the computation
    number_of_workers: int
        Default is 1. Number of parallel process workers at one time.
        
    Notes
    ------
    In some cases where the input and output coordinates of the source node is not the same (such as reduce nodes)
    and fill_output is True, the user may need to specify 'output' as part of the eval call.
    """

    source = NodeTrait().tag(attr=True)
    chunks = tl.Dict().tag(attr=True)
    fill_output = tl.Bool(True).tag(attr=True)
    number_of_workers = tl.Int(1).tag(attr=True)
    _lock = Lock()

    def eval(self, coordinates, output=None):
        # Make a thread pool to manage queue
        pool = ThreadPool(processes=self.number_of_workers)

        if output is None and self.fill_output:
            output = self.create_output_array(coordinates)

        shape = []
        for d in coordinates.dims:
            if d in self.chunks:
                shape.append(self.chunks[d])
            else:
                shape.append(coordinates[d].size)

        results = []
        #         inputs = []
        for coords, slc in coordinates.iterchunks(shape, True):
            #             inputs.append(coords)
            out = None
            if self.fill_output and output is not None:
                out = output[slc]
            with self._lock:
                _log.debug("Node eval with coords: {}, {}".format(slc, coords))
                results.append(pool.apply_async(self.eval_source, [coords, slc, out]))

        for i, res in enumerate(results):
            #             _log.debug('Waiting for results: {} {}'.format(i, inputs[i]))
            _log.debug("Waiting for results: {}".format(i))
            o, slc = res.get()
            if self.fill_output:
                if output is None:
                    missing_dims = [d for d in coordinates.dims if d not in self.chunks.keys()]
                    coords = coordinates.drop(missing_dims)
                    missing_coords = Coordinates.from_xarray(o).drop(list(self.chunks.keys()))
                    coords = merge_dims([coords, missing_coords])
                    coords = coords.transpose(*coordinates.dims)
                    output = self.create_output_array(coords)
                output[slc] = o

        _log.debug("Completed parallel execution.")
        pool.close()

        return output

    def eval_source(self, coordinates, coordinates_index, out):
        return (self.source.eval(coordinates, output=out), coordinates_index)


class ParallelOutputZarr(Parallel):
    """
    This class assumes that the node has a 'output_format' attribute
    (currently the "Lambda" Node, and the "Process" Node)
    
    Attributes
    -----------
    zarr_file: str
        Path to the output zarr file that collects all of the computed results. This can reside on S3. 
    dataset: ZarrGroup
        A handle to the zarr group pointing to the output file
    fill_output: bool, optional
        Default is False (unlike parent class). If True, will collect the output data and return it as an xarray.
    init_file_mode: str, optional
        Default is 'w'. Mode used for initializing the zarr file. 
    zarr_chunks: dict
        Size of the chunks in the zarr file for each dimension
    zarr_shape: dict, optional
        Default is the {coordinated.dims: coordinates.shape}, where coordinates used as part of the eval call. This 
        does not need to be specified unless the Node modifies the input coordinates (as part of a Reduce operation, 
        for example). The result can be incorrect and requires care/checking by the user.
    zarr_coordinates: podpac.Coordinates, optional
        Default is None. If the node modifies the shape of the input coordinates, this allows users to set the
        coordinates in the output zarr file. This can be incorrect and requires care by the user.
    """

    zarr_file = tl.Unicode().tag(attr=True)
    dataset = tl.Any()
    fill_output = tl.Bool(False)
    init_file_mode = tl.Unicode("w").tag(attr=True)
    zarr_chunks = tl.Dict().tag(attr=True)
    zarr_shape = tl.Dict(allow_none=True, default_value=None).tag(attr=True)
    zarr_coordinates = tl.Instance(Coordinates, allow_none=True, default_value=None).tag(attr=True)
    _shape = tl.Tuple()

    def eval(self, coordinates, output=None):
        if self.zarr_shape is None:
            self._shape = {d: v for d, v in coordinates.shape}
        else:
            self._shape = tuple(self.zarr_shape.values())

        # initialize zarr file
        chunks = [self.zarr_chunks[d] for d in coordinates]
        zf, data_key = self.initialize_zarr_array(self._shape, chunks)
        self.dataset = zf

        # eval
        _log.debug("Starting parallel eval.")
        missing_dims = [d for d in coordinates.dims if d not in self.chunks.keys()]
        if self.zarr_coordinates is not None:
            missing_dims = missing_dims + [d for d in self.zarr_coordinates.dims if d not in missing_dims]
            set_coords = merge_dims([coordinates.drop(missing_dims), self.zarr_coordinates])
        else:
            set_coords = coordinates.drop(missing_dims)
        set_coords.transpose(*coordinates.dims)

        self.set_zarr_coordinates(set_coords, data_key)

        output = super(ParallelOutputZarr, self).eval(coordinates, output)

        # fill in the coordinates, this is guaranteed to be correct even if the user messed up.
        if output is not None:
            self.set_zarr_coordinates(Coordinates.from_xarray(output), data_key)

        return output

    def set_zarr_coordinates(self, coordinates, data_key):
        # Fill in metadata
        for dk in data_key:
            self.dataset[dk].attrs["_ARRAY_DIMENSIONS"] = coordinates.dims
        for d in coordinates.dims:
            # TODO ADD UNITS AND TIME DECODING INFORMATION
            self.dataset.create_dataset(d, shape=coordinates[d].size, overwrite=True)
            self.dataset[d][:] = coordinates[d].coordinates

    def initialize_zarr_array(self, shape, chunks):
        _log.debug("Creating Zarr file.")
        zn = Zarr(source=self.zarr_file, file_mode=self.init_file_mode)
        if self.source.output or getattr(self.source, "data_key", None):
            data_key = self.source.output
            if data_key is None:
                data_key = self.source.data_key
            data_key = [data_key]
        elif self.source.outputs:
            data_key = self.source.outputs
        else:
            data_key = ["data"]

        zf = zarr.open(zn._get_store(), mode=self.init_file_mode)

        # Intialize the output zarr arrays
        for dk in data_key:
            arr = zf.create_dataset(dk, shape=shape, chunks=chunks, fill_value=np.nan, overwrite=True)

        return zf, data_key

    def eval_source(self, coordinates, coordinates_index, out):
        source = Node.from_definition(self.source.definition)
        _log.debug("Creating output format.")
        output = dict(
            format="zarr_part",
            format_kwargs=dict(
                part=[[s.start, min(s.stop, self._shape[i]), s.step] for i, s in enumerate(coordinates_index)],
                source=self.zarr_file,
                mode="a",
            ),
        )
        _log.debug("Finished creating output format.")
        source.set_trait("output_format", output)
        _log.debug("Evaluating node.")
        _log.debug("output: {}, coordinates.shape: {}".format(output, coordinates.shape))

        return source.eval(coordinates, out), coordinates_index
