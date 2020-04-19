from __future__ import division, unicode_literals, print_function, absolute_import

import numpy as np
import traitlets as tl

import podpac
from podpac.core.coordinates import Coordinates
from podpac.core.units import UnitsDataArray
from podpac.core.utils import common_doc, cached_property, ind2slice
from podpac.core.data.datasource import DataSource, COMMON_DATA_DOC


@common_doc(COMMON_DATA_DOC)
class TileCompositor(DataSource):
    """Composite tiled datasources.

    Attributes
    ----------
    sources : list
        The tiled data sources.
    coordinates : Coordinates
        Coordinates encompassing all of the tiled sources.

    Notes
    -----
    This compositor aggregates source data first and then interpolates the requested coordinates.
    """

    @property
    def sources(self):
        """ Tiled data sources (using the TileMixin).

        Child classes should define these sources including a reference to itself and the tile_coordinates_index.
        """

        raise NotImplementedError()

    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """

        output = self.create_output_array(coordinates)
        for source in self.sources:
            c, I = source.coordinates.intersect(coordinates, return_indices=True)
            if c.size == 0:
                continue
            source_data = source.get_data(c, I)
            output.loc[source_data.coords] = source_data

        return output


@common_doc(COMMON_DATA_DOC)
class UniformTileCompositor(TileCompositor):
    """Composite a grid of uniformly tiled datasources.

    Attributes
    ----------
    sources : list
        The tiled data sources.
    coordinates : Coordinates
        Coordinates encompassing all of the tiled sources.
    shape : tuple
        shape of the tile grid
    tile_width : tuple
        shape of the coordinates for each tile

    Notes
    -----
    This compositor aggregates source data first and then interpolates the requested coordinates.
    """

    shape = tl.Tuple()
    _repr_keys = ["shape"]

    @property
    def sources(self):
        """ Tiled data sources (using the UniformTileMixin).

        Child classes should define these sources including a reference to itself and the tile index in the grid.
        """

        raise NotImplementedError()

    @cached_property
    def tile_width(self):
        """Tuple of the number of coordinates that the tile covers in each dimension."""
        return tuple(int(n / m) for n, m in zip(self.coordinates.shape, self.shape))


@common_doc(COMMON_DATA_DOC)
class UniformTileMixin(tl.HasTraits):
    """DataSource mixin for uniform tiles in a grid.

    Defines the tile coordinates from the grid coordinates using the tile position in the grid.

    Attributes
    ----------
    grid : TileCompositor
        tiling compositor containing the grid coordinates, grid shape, and tile sources
    tile : tuple
        index for this tile in the grid
    width : tuple
        width
    """

    grid = tl.Instance(TileCompositor)
    tile = tl.Tuple()

    @tl.validate("tile")
    def _validate_tile(self, d):
        tile = d["value"]
        if len(tile) != len(self.grid.shape):
            raise ValueError("tile index does not match grid shape (%d != %d)" % (len(tile), len(self.grid.shape)))
        if not all(0 <= i < n for (i, n) in zip(tile, self.grid.shape)):
            raise ValueError("tile index %s out of range for grid shape %s)" % (len(tile), len(self.grid.shape)))
        return tile

    @property
    def width(self):
        return self.grid.tile_width

    def get_coordinates(self):
        """{get_coordinates}
        """
        Is = tuple(slice(w * i, w * (i + 1)) for i, w in zip(self.tile, self.width))
        return self.grid.coordinates[Is]

    @property
    def _repr_keys(self):
        return super(UniformTileMixin, self)._repr_keys + ["tile"]
