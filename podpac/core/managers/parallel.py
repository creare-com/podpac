"""
Module to help farm out computation to multiple workers and save the results in a zarr file.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import time
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
try:
    import botocore
except:

    class dum:
        pass

    class mod:
        ClientError = dum
        ReadTimeoutError = dum

    class botocore:
        exceptions = mod


# Set up logging
_log = logging.getLogger(__name__)


class Parallel(Node):
    """
    This class launches the parallel node evaluations in separate threads. As such, the node does not need to return
    immediately (i.e. does NOT have to be asynchronous). For asynchronous nodes
    (i.e. aws.Lambda with download_result=False) use ParrallelAsync

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
    start_i: int, optional
        Default is 0. Starting chunk. This allow you to restart a run without having to check/submit 1000's of workers
        before getting back to where you were. Empty chunks make the submission slower.

    Notes
    ------
    In some cases where the input and output coordinates of the source node is not the same (such as reduce nodes)
    and fill_output is True, the user may need to specify 'output' as part of the eval call.
    """

    _repr_keys = ["source", "number_of_workers", "chunks"]
    source = NodeTrait().tag(attr=True)
    chunks = tl.Dict().tag(attr=True)
    fill_output = tl.Bool(True).tag(attr=True)
    number_of_workers = tl.Int(1).tag(attr=True)
    _lock = Lock()
    errors = tl.List()
    start_i = tl.Int(0)

    def eval(self, coordinates, **kwargs):
        output = kwargs.get("output")
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
        i = 0
        for coords, slc in coordinates.iterchunks(shape, True):
            #             inputs.append(coords)
            if i < self.start_i:
                _log.debug("Skipping {} since it is less than self.start_i ({})".format(i, self.start_i))
                i += 1
                continue

            out = None
            if self.fill_output and output is not None:
                out = output[slc]
            with self._lock:
                _log.debug("Added {} to worker pool".format(i))
                _log.debug("Node eval with coords: {}, {}".format(slc, coords))
                results.append(pool.apply_async(self.eval_source, [coords, slc, out, i]))
            i += 1

        _log.info("Added all chunks to worker pool. Now waiting for results.")
        start_time = time.time()
        for i, res in enumerate(results):
            #             _log.debug('Waiting for results: {} {}'.format(i, inputs[i]))
            dt = str(np.timedelta64(int(1000 * (time.time() - start_time)), "ms").astype(object))
            _log.info("({}): Waiting for results: {} / {}".format(dt, i + 1, len(results)))

            # Try to get the results / wait for the results
            try:
                o, slc = res.get()
            except Exception as e:
                o = None
                slc = None
                self.errors.append((i, res, e))
                dt = str(np.timedelta64(int(1000 * (time.time() - start_time)), "ms").astype(object))
                _log.warning("({}) {} failed with exception {}".format(dt, i, e))

            dt = str(np.timedelta64(int(1000 * (time.time() - start_time)), "ms").astype(object))
            _log.info("({}) Finished result: {} / {}".format(time.time() - start_time, i + 1, len(results)))

            # Fill output
            if self.fill_output:
                if output is None:
                    missing_dims = [d for d in coordinates.dims if d not in self.chunks.keys()]
                    coords = coordinates.drop(missing_dims)
                    missing_coords = Coordinates.from_xarray(o).drop(list(self.chunks.keys()))
                    coords = merge_dims([coords, missing_coords])
                    coords = coords.transpose(*coordinates.dims)
                    output = self.create_output_array(coords)
                output[slc] = o

        _log.info("Completed parallel execution.")
        pool.close()

        return output

    def eval_source(self, coordinates, coordinates_index, out, i, source=None):
        if source is None:
            source = self.source
            # Make a copy to prevent any possibility of memory corruption
            source = Node.from_definition(source.definition)

        _log.info("Submitting source {}".format(i))
        return (source.eval(coordinates, output=out), coordinates_index)


class ParallelAsync(Parallel):
    """
    This class launches the parallel node evaluations in threads up to n_workers, and expects the node.eval to return
    quickly for parallel execution. This Node was written with aws.Lambda(eval_timeout=1.25<small>) Nodes in mind.

    Users can implement the `check_worker_available` method or specify the `no_worker_exception` attribute, which is an
    exception thrown if workers are not available.

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
    sleep_time: float
        Default is 1 second. Number of seconds to sleep between trying to submit new workers
    no_worker_exception: Exception, optional
        Default is .Exception class used to identify when a submission failed due to no available workers. The default
        is chosen to work with the podpac.managers.Lambda node.
    async_exception: Exception
        Default is botocore.exceptions.ReadTimeoutException. This is an exception thrown by the async function in case
        it time out waiting for a return. In our case, this is a success. The default is chosen to work with the
        podpac.managers.Lambda node.
    Notes
    ------
    In some cases where the input and output coordinates of the source node is not the same (such as reduce nodes)
    and fill_output is True, the user may need to specify 'output' as part of the eval call.
    """

    source = NodeTrait().tag(attr=True)
    chunks = tl.Dict().tag(attr=True)
    fill_output = tl.Bool(True).tag(attr=True)
    sleep_time = tl.Float(1).tag(attr=True)
    no_worker_exception = tl.Type(botocore.exceptions.ClientError).tag(attr=True)
    async_exception = tl.Type(botocore.exceptions.ReadTimeoutError).tag(attr=True)

    def check_worker_available(self):
        return True

    def eval_source(self, coordinates, coordinates_index, out, i, source=None):
        if source is None:
            source = self.source
            # Make a copy to prevent any possibility of memory corruption
            source = Node.from_definition(source.definition)

        success = False
        o = None
        while not success:
            if self.check_worker_available():
                try:
                    o = source.eval(coordinates, output=out)
                    success = True
                except self.async_exception:
                    # This exception is fine and constitutes a success
                    o = None
                    success = True
                except self.no_worker_exception as e:
                    response = e.response
                    if not (response and response.get("Error", {}).get("Code") == "TooManyRequestsException"):
                        raise e  # Raise error again, not the right error
                    _log.debug("Worker {} exception {}".format(i, e))
                    success = False
                    time.sleep(self.sleep_time)
            else:
                _log.debug("Worker unavailable for {}".format(i, e))
                time.sleep(self.sleep_time)
        _log.info("Submitting source {}".format(i))
        return (o, coordinates_index)


class ZarrOutputMixin(tl.HasTraits):
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
    skip_existing: bool
        Default is False. If true, this will check to see if the results already exist. And if so, it will not
        submit a job for that particular coordinate evaluation. This assumes self.chunks == self.zar_chunks
    list_dir: bool, optional
        Default is False. If skip_existing is True, by default existing files are checked by asking for an 'exists' call.
        If list_dir is True, then at the first opportunity a "list_dir" is performed on the directory and the results
        are cached.
    """

    zarr_file = tl.Unicode().tag(attr=True)
    dataset = tl.Any()
    zarr_node = NodeTrait()
    zarr_data_key = tl.Union([tl.Unicode(), tl.List()])
    fill_output = tl.Bool(False)
    init_file_mode = tl.Unicode("a").tag(attr=True)
    zarr_chunks = tl.Dict(default_value=None, allow_none=True).tag(attr=True)
    zarr_shape = tl.Dict(allow_none=True, default_value=None).tag(attr=True)
    zarr_coordinates = tl.Instance(Coordinates, allow_none=True, default_value=None).tag(attr=True)
    zarr_dtype = tl.Unicode("f4")
    skip_existing = tl.Bool(True).tag(attr=True)
    list_dir = tl.Bool(False)
    _list_dir = tl.List(allow_none=True, default_value=[])
    _shape = tl.Tuple()
    _chunks = tl.List()
    aws_client_kwargs = tl.Dict()
    aws_config_kwargs = tl.Dict()

    def eval(self, coordinates, **kwargs):
        output = kwargs.get("output")
        if self.zarr_shape is None:
            self._shape = coordinates.shape
        else:
            self._shape = tuple(self.zarr_shape.values())

        # initialize zarr file
        if self.zarr_chunks is None:
            chunks = [self.chunks[d] for d in coordinates]
        else:
            chunks = [self.zarr_chunks[d] for d in coordinates]
        self._chunks = chunks
        zf, data_key, zn = self.initialize_zarr_array(self._shape, chunks)
        self.dataset = zf
        self.zarr_data_key = data_key
        self.zarr_node = zn
        zn.keys

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
        if self.list_dir:
            dk = data_key
            if isinstance(dk, list):
                dk = dk[0]
            self._list_dir = self.zarr_node.list_dir(dk)

        output = super(ZarrOutputMixin, self).eval(coordinates, output=output)

        # fill in the coordinates, this is guaranteed to be correct even if the user messed up.
        if output is not None:
            self.set_zarr_coordinates(Coordinates.from_xarray(output), data_key)
        else:
            return zf

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
        zn = Zarr(source=self.zarr_file, file_mode=self.init_file_mode, aws_client_kwargs=self.aws_client_kwargs)
        if self.source.output or getattr(self.source, "data_key", None):
            data_key = self.source.output
            if data_key is None:
                data_key = self.source.data_key
            if not isinstance(data_key, list):
                data_key = [data_key]
            elif self.source.outputs:  # If someone restricted the outputs for this node, we need to know
                data_key = [dk for dk in data_key if dk in self.source.outputs]
        elif self.source.outputs:
            data_key = self.source.outputs
        else:
            data_key = ["data"]

        zf = zarr.open(zn._get_store(), mode=self.init_file_mode)

        # Intialize the output zarr arrays
        for dk in data_key:
            try:
                arr = zf.create_dataset(
                    dk,
                    shape=shape,
                    chunks=chunks,
                    fill_value=np.nan,
                    dtype=self.zarr_dtype,
                    overwrite=not self.skip_existing,
                )
            except ValueError:
                pass  # Dataset already exists

        # Recompute any cached properties
        zn = Zarr(source=self.zarr_file, file_mode=self.init_file_mode, aws_client_kwargs=self.aws_client_kwargs)
        return zf, data_key, zn

    def eval_source(self, coordinates, coordinates_index, out, i, source=None):
        if source is None:
            source = self.source

        if self.skip_existing:  # This section allows previously computed chunks to be skipped
            dk = self.zarr_data_key
            if isinstance(dk, list):
                dk = dk[0]
            try:
                exists = self.zarr_node.chunk_exists(
                    coordinates_index, data_key=dk, list_dir=self._list_dir, chunks=self._chunks
                )
            except ValueError as e:  # This was needed in cases where a poor internet connection caused read errors
                exists = False
            if exists:
                _log.info("Skipping {} (already exists)".format(i))
                return out, coordinates_index

        # Make a copy to prevent any possibility of memory corruption
        source = Node.from_definition(source.definition)
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

        if source.has_trait("output_format"):
            source.set_trait("output_format", output)
        _log.debug("output: {}, coordinates.shape: {}".format(output, coordinates.shape))
        _log.debug("Evaluating node.")

        o, slc = super(ZarrOutputMixin, self).eval_source(coordinates, coordinates_index, out, i, source)

        if not source.has_trait("output_format"):
            o.to_format(output["format"], **output["format_kwargs"])
        return o, slc


class ParallelOutputZarr(ZarrOutputMixin, Parallel):
    pass


class ParallelAsyncOutputZarr(ZarrOutputMixin, ParallelAsync):
    pass
