import shutil

# Create a PODPAC node out of a zarr archive
import podpac
from podpac.data import Zarr, ZarrMemory
from podpac.core.interpolation.selector import Selector
from podpac.core.cache.cache_interface import CacheNode
from podpac import settings


import numpy as np
import traitlets as tl
import zarr


class ZarrCache(CacheNode):
    """
    A PODPAC CachingNode which uses Zarr archives to cache data from a source node.

    Attributes
    ----------

    zarr_path_data : str
        The path to the Zarr archive for storing data. Default is f"{self.base_path}/zarr_cache_{self.hash}"
    base_path : str
        Base path for caching to disk. Default is `podpac.settings.cache_path`
    group_data : zarr.hierarchy.Group
        The Zarr group for storing data.
    group_bool : zarr.hierarchy.Group
        The Zarr group for storing boolean indicators of data availability.
    chunks: list
        Chunk size for the Zarr array. If None, the default chunk size is used.
    selector_method : str
        Method for selecting existing data in cache. Default is "nearest"
    cache_type : tl.Enum
        One of 'disk' (default) or 'ram'

    _z_node : Zarr
        Zarr node for data.
    _z_bool : Zarr
        Zarr node for boolean indicators.
    _from_cache : bool
        Flag indicating whether the data was retrieved from the cache.

    Notes
    ---------
    Currently the output caching is a bit sub-optimal. If source.output contains ANY non-cached
    output, then ALL the outputs in source.output is fetched. Users can deal with this by only
    fetching cached outputs, or only fetching non-cached outputs.
    TODO: update this in the future.
    """

    # Public Traits
    zarr_path = tl.Unicode()
    base_path = tl.Unicode().tag(attr=True, required=True)
    group_data = tl.Instance(zarr.hierarchy.Group)
    group_bool = tl.Instance(zarr.hierarchy.Group)
    chunks = tl.Union([tl.List(), tl.Dict()], allow_none=True, default_value=None).tag(attr=True)
    selector_method = tl.Unicode(allow_none=True).tag(attr=True)
    cache_type = tl.Enum(["disk", "ram"], default_value="disk").tag(attr=True)
    data_dtype = tl.Unicode('float64').tag(attr=True)
    compressor = tl.Any(zarr.Blosc(cname='zstd', clevel=5, shuffle=zarr.blosc.SHUFFLE))

    # Private Traits
    _z_node = tl.Instance(Zarr)
    _z_bool = tl.Instance(Zarr)
    _from_cache = tl.Bool(allow_none=True, default_value=False)
    _zarr_path_data = tl.Unicode()
    _zarr_path_bool = tl.Unicode()
    _selector = tl.Instance(Selector, allow_none=True)
    _global_zarr_ram_cache = {}  # This is a class-level variable and should not be over-written!
    _global_zarr_bool_ram_cache = {}  # This is a class-level variable and should not be over-written!

    @property
    def _coordinates(self):
        # Just a straight pass-through for now, but might allow custom coordinates for cache later
        return self.source.coordinates

    @property
    def shape(self):
        shape = self._coordinates.shape
        if self.outputs is not None:
            shape = shape + (len(self.outputs), )
        return shape

    @property
    def outputs(self):
        # TODO: This implementation of multiple outputs DOES NOT (maybe?) match the implementation in data_keys from the FilekeysMixin... so this could/should be revisited.
        return self.source.outputs

    @property
    def output(self):
        return self.source.output

    @property
    def _chunks(self):
        if self.chunks is None:
            return True
        if isinstance(self.chunks, list):
            if self.outputs is None:
                assert len(self.chunks) == len(self._coordinates.dims), "Need to specify chunk size (%s) for every dimension (%s)" % (str(self.chunks), str(self._coordinates.dims))
            else:
                assert len(self.chunks) == len(self._coordinates.dims), "Need to specify chunk size (%s) for every dimension (%s) including outputs" % (str(self.chunks), str(self._coordinates.dims))
            return self.chunks
        if isinstance(self.chunks, dict):
            if self.outputs is None:
                return [self.chunks[k] for k in self._coordinates.dims]
            else:
                return [self.chunks[k] for k in self._coordinates.dims] + [self.chunks.get("output", 1)]

    @tl.default("_selector")
    def _default_selector(self):
        return Selector(method=self.selector_method)

    @tl.default("selector_method")
    def _default_selector_method(self):
        return "nearest"

    @tl.default("base_path")
    def _default_base_path(self):
        return podpac.settings.cache_path

    # Because of the way Zarr Nodes work, If I have one zarr node for both groups (making them into arrays for the same group), every time I eval the zarr node, it evals both datasets/arrays. I don't necessarily want this, given that first I need to eval the Boolean array without evalling the data array, get the data from the server, then only eval the data array.
    @tl.default("zarr_path")
    def _default_zarr_path(self):
        return f"{self.base_path}/zarr_cache_{self.hash}"

    @tl.default("_zarr_path_data")
    def _default_zarr_path_data(self):
        return f"{self.zarr_path}/data.zarr"

    @tl.default("_zarr_path_bool")
    def _default_zarr_path_bool(self):
        return f"{self.zarr_path}/bool.zarr"

    @tl.default("group_data")
    def _default_group_data(self):
        try:
            if self.cache_type == "disk":
                group = zarr.open(
                    self._zarr_path_data, mode="a"
                )  # no need to close, see https://zarr.readthedocs.io/en/stable/tutorial.html#persistent-arrays
            if self.cache_type == "ram":
                if self.hash not in self._global_zarr_ram_cache:
                    self._global_zarr_ram_cache[self.hash] = zarr.group()
                group = self._global_zarr_ram_cache[self.hash]  # assumes ram not persistent
            if "data" not in group:
                group.create_dataset(
                    "data",
                    shape=self.shape,
                    chunks=self._chunks,
                    dtype=self.data_dtype,
                    fill_value=np.nan,
                    write_empty_chunks=False,
                    compressor=self.compressor
                )  # adjust dtype as necessary
                self._create_coordinate_zarr_dataset(group, ['data'])

            return group
        except Exception as e:
            raise ValueError(f"Failed to open zarr data group. Original error: {e}")

    @tl.default("group_bool")
    def _default_group_bool(self):
        try:
            if self.cache_type == "disk":
                group = zarr.open(
                    self._zarr_path_bool, mode="a"
                )  # no need to close, see https://zarr.readthedocs.io/en/stable/tutorial.html#persistent-arrays
            if self.cache_type == "ram":
                if self.hash not in self._global_zarr_bool_ram_cache:
                    self._global_zarr_bool_ram_cache[self.hash] = zarr.group()
                group = self._global_zarr_bool_ram_cache[self.hash]  # assumes ram not persistent
            if "contains" not in group:
                group.create_dataset(
                    "contains",
                    shape=self.shape,
                    chunks=self._chunks,
                    dtype="bool",
                    fill_value=False,
                    write_empty_chunks=False,
                    compressor=self.compressor
                )
                self._create_coordinate_zarr_dataset(group, ['contains'])
            return group
        except Exception as e:
            raise ValueError(f"Failed to open zarr boolean group. Original error: {e}")

    @tl.default("_z_node")
    def _default_z_node(self):
        if self.cache_type == "disk":
            try:
                self.group_data  # ensure group exists
                return Zarr(source=self._zarr_path_data, coordinates=self._coordinates, file_mode="r+")
            except Exception as e:
                raise ValueError(f"Failed to create Zarr node. Original error: {e}")
        elif self.cache_type == "ram":
            try:
                self.group_data  # ensure group exists
                return ZarrMemory(dataset=self.group_data, coordinates=self._coordinates)
            except Exception as e:
                raise ValueError(f"Failed to create Zarr node. Original error: {e}")

    @tl.default("_z_bool")
    def _default_z_bool(self):
        if self.cache_type == "disk":
            try:
                self.group_bool  # ensure group exists
                return Zarr(source=self._zarr_path_bool, coordinates=self._coordinates, file_mode="r+")
            except Exception as e:
                raise ValueError(f"Failed to create Zarr node. Original error: {e}")
        elif self.cache_type == "ram":
            try:
                self.group_bool  # ensure group exists
                return ZarrMemory(dataset=self.group_bool, coordinates=self._coordinates)
            except Exception as e:
                raise ValueError(f"Failed to create Zarr node. Original error: {e}")

    def _create_coordinate_zarr_dataset(self, group, datasets=[]):
        """
        Create a Zarr dataset for storing coordinates.

        Returns
        -------
        zarr.Dataset
            The Zarr dataset for storing coordinates.
        """
        for dim in self._coordinates.dims:
            if dim not in group:
                if dim == "time":
                    group.create_dataset(
                        dim,
                        shape=self._coordinates[dim].shape,
                        dtype=str(self._coordinates["time"].bounds[0].dtype),
                    )
                else:
                    group.create_dataset(dim, shape=self._coordinates[dim].shape, dtype="float64")
                group[dim][:] = self._coordinates.xcoords[dim][1]
        for dataset in datasets:
            group[dataset].attrs["_ARRAY_DIMENSIONS"] = self._coordinates.dims
            group[dataset].attrs["_OUTPUT_LABELS"] = self.outputs


    def _create_slices(self, c3, index_arrays):
        """
        Create slices for the given coordinates and index arrays.

        Parameters
        ----------
        c3 : podpac.Coordinates
            The coordinates.
        index_arrays : list of np.ndarray
            The index arrays.

        Returns
        -------
        slices : dict
            The slices for the given coordinates and index arrays.
        """
        slices = {}
        for dim, indices in zip(c3.keys(), index_arrays):
            indices = indices.flatten()  # convert to 1D array
            slices[dim] = slice(indices[0], indices[-1] + 1)
        return slices

    def rem_cache(self):
        if self.cache_type == "disk":
            shutil.rmtree(self.zarr_path)
        elif self.cache_type == "ram":
            self.group_data["data"][:] = np.nan
            self.group_bool["contains"][:] = False

            if self.hash in self._global_zarr_bool_ram_cache:
                del self._global_zarr_bool_ram_cache[self.hash]
            if self.hash in self._global_zarr_ram_cache:
                del self._global_zarr_ram_cache[self.hash]

    def get_source_data(self, request_coords):
        """
        Retrieve data from the source at the specified coordinates.

        Parameters
        ----------
        request_coords : podpac.Coordinates
            The coordinates at which data is requested from the source.

        Returns
        -------
        data : np.ndarray
            The data retrieved from the source at the specified coordinates.
        """
        data = self.source.eval(request_coords)
        return data

    def fill_zarr(self, data, request_coords):
        """
        Fill the Zarr cache with data at the specified coordinates.

        Parameters
        ----------
        data : np.ndarray
            The data to be stored in the Zarr cache.
        request_coords : podpac.Coordinates
            The coordinates at which the data should be stored in the Zarr cache.
        """
        c3, index_arrays = self._selector.select(self._coordinates, request_coords)
        slices = self._create_slices(c3, index_arrays)
        slices = tuple(slices.get(dim) for dim in self._coordinates.dims)

        if self.outputs is not None and data.output.size < len(self.outputs):
            for output in np.atleast_1d(data.output):
                output_index = self.outputs.index(output.item())
                if len(data.shape) == len(slices):
                    # When there is only a single output, the array is already "squeezed"
                    self._z_node.dataset["data"][slices + (output_index,)] = data
                else:
                    self._z_node.dataset["data"][slices + (output_index,)] = data.sel(output=output)
                self._z_bool.dataset["contains"][slices + (output_index,)] = True
        else:
            self._z_node.dataset["data"][slices] = data.data
            self._z_bool.dataset["contains"][slices] = True

    def subselect_has(self, request_coords, source_output):
        """
        Fetch the coordinates for which the Zarr cache does not have data yet.

        Parameters
        ----------
        request_coords : podpac.Coordinates
            The coordinates at which data is requested.

        Returns
        -------
        false_coords : podpac.Coordinates or None
            The subset of the requested coordinates for which the Zarr cache does not have data yet.
            If the Zarr cache has data for all requested coordinates, returns None.
        """

        c3, index_arrays = self._selector.select(self._coordinates, request_coords)
        slices = self._create_slices(c3, index_arrays)
        slice_inds = tuple(slices.get(dim) for dim in self._coordinates.dims)

        # Check if all values are True for the outputs being evaluated
        if source_output is not None and self.outputs is not None:
            output_inds = [self.outputs.index(so) for so in source_output]
            bool_data = np.stack([self._z_bool.dataset["contains"][slice_inds + (oi, )] for oi in output_inds], axis=-1)
        else:
            bool_data = self._z_bool.dataset["contains"][slice_inds]

        # check if all values are True
        if np.all(bool_data):
            return None  # or any other indicator that all data is present


        false_indices = np.where(bool_data == False)

        false_indices_unique = tuple(np.unique(indices) for indices in false_indices)

        # This works out because outputs is always axis=-1
        false_coords = {}
        for dim, indices in zip(c3.dims, false_indices_unique):
            false_coords[dim] = c3[dim][indices]

        return podpac.Coordinates(
            [false_coords.get(dim) for dim in self._coordinates.dims],
            dims=self._coordinates.dims, crs=self._coordinates.crs
        )

    def _eval(self, coordinates, output=None, _selector=None):
        """
        Evaluate the data at the requested coordinates, fetching missing data from the source node and filling the Zarr cache as necessary.
        If requested coordinates are out of the source node's bounds, return an array filled with NaNs.

        Parameters
        ----------
        coordinates : podpac.Coordinates
            The coordinates at which data is requested.

        Returns
        -------
        data : np.ndarray
            The data at the requested coordinates. If coordinates were outside of the source's bounds, those positions will be filled with np.nan.
        """
        self._from_cache = False

        # Initialize output
        dim_order = coordinates.dims
        coordinates = coordinates.transpose(*self._coordinates.dims)
        if self.source.output is not None:
            if isinstance(self.source.output, list):
                data = self.create_output_array(coordinates, outputs=self.source.output)
                output_inds = [self.outputs.index(so) for so in self.source.output]
            else:
                data = self.create_output_array(coordinates, outputs=[self.source.output])
                output_inds = [self.outputs.index(self.source.output)]
        else:
            output_inds = slice(None)
            data = self.create_output_array(coordinates)

        # Find valid request coordinates that are within the source's bounds
        valid_coordinates, valid_request_indices = coordinates.intersect(self._coordinates, return_index=True)

        if valid_coordinates.size > 0:
            subselect_coords = self.subselect_has(valid_coordinates, self.source.output)

            if subselect_coords is None and settings["ENABLE_CACHE"] and self.source.cache_output:
                self._from_cache = True
            else:
                missing_data = self.get_source_data(subselect_coords)
                if settings["ENABLE_CACHE"] and self.source.cache_output:
                    self.fill_zarr(missing_data, subselect_coords)

            c3, index_arrays = self._selector.select(self._coordinates, valid_coordinates)
            slices = self._create_slices(c3, index_arrays)
            slices_inds = tuple(slices.get(dim) for dim in self._coordinates.dims)

            # Use the slices to place data from Zarr cache into the correct location in the result array
            data[valid_request_indices] = self._z_node.dataset["data"][slices_inds][..., output_inds]

        return data.transpose(*dim_order, ...)
