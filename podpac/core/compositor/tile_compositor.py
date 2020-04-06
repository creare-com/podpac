from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import traitlets as tl

import podpac
from podpac.core.coordinates import Coordinates
from podpac.core.units import UnitsDataArray
from podpac.core.utils import common_doc, cached_property, ind2slice
from podpac.core.data.datasource import DataSource, COMMON_DATA_DOC


@common_doc(COMMON_DATA_DOC)
class TileMixin(tl.HasTraits):
    """DataSource mixin for tiles. Defines the tile native_coordinates from global coordinates.

    Attributes
    ----------
    tile : tuple
        indices for this tile in the grid
    ntiles : tuple
        shape of the grid
    global_coordinates : Coordinates
        coordinates for the entire grid
    """

    tile = tl.Tuple().tag(readonly=True)
    ntiles = tl.Tuple().tag(readonly=True)
    global_coordinates = tl.Instance(Coordinates).tag(readonly=True)

    @tl.validate("tile")
    def _validate_tile(self, d):
        # TODO check that len(tile) == self.global_coordinates.ndim
        return d["value"]

    @tl.validate("ntiles")
    def _validate_shape(self, d):
        # TODO check that it divides the global_coordinates evenly
        return d["value"]

    @property
    def _repr_keys(self):
        return super(self, TileMixin)._repr_keys + ["tile"]

    @property
    def width(self):
        """Tuple of the number of coordinates that the tile covers in each dimension."""
        return tuple(int(n / m) for n, m in zip(self.global_coordinates.shape, self.ntiles))

    @property
    def tile_coordinates_index(self):
        """Tuple with indices for the coordinates of this tile"""
        return tuple(slice(w * i, w * (i + 1)) for i, w in zip(self.tile, self.width))

    def get_native_coordinates(self):
        """{get_native_coordinates}
        """
        return self.global_coordinates[self.tile_coordinates_index]


@common_doc(COMMON_DATA_DOC)
class TileCompositor(DataSource):
    """Composite tiled datasources.

    Attributes
    ----------
    sources : list
        The tiled data sources.
    native_coordinates : Coordinates
        Coordinates encompassing all of the tiled sources.

    Notes
    -----
    This compositor aggregates source data first and then interpolates the requested coordinates.
    """

    @property
    def sources(self):
        """ Tiled data sources.

        Child classes should define these sources, including
         * global_coordinates (the compositor native_coordinates)
         * tile_coordinates_index
        """

        raise NotImplementedError()

    def get_native_coordinates(self):
        """{get_native_coordinates}
        """
        raise NotImplementedError()

    def get_data(self, coordinates, coordinates_index):
        """ """
        # TODO probably needs to be UnitsDataArray for multiple outputs handling
        b = np.zeros(self.native_coordinates.shape, dtype=bool)
        b[coordinates_index] = True

        output = self.create_output_array(coordinates)
        for source in self.sources:
            c, I = coordinates.intersect(source.native_coordinates, return_indices=True)
            if c.size == 0:
                continue

            bb = b[source.tile_coordinates_index]
            Js = ind2slice(np.where(bb))
            source_data = source.get_data(source.native_coordinates[Js], Js)
            output.data[I] = source_data.data
        return output
