from __future__ import division, unicode_literals, print_function, absolute_import

import sys
import traitlets as tl


class BaseCoordinates(tl.HasTraits):
    """Base class for single or stacked one-dimensional coordinates."""

    def _set_name(self, value):
        raise NotImplementedError

    @property
    def name(self):
        """:str: Dimension name."""
        raise NotImplementedError

    @property
    def dims(self):
        """:tuple: Dimensions."""
        raise NotImplementedError

    @property
    def udims(self):
        """:tuple: Tuple of unstacked dimension names, for compatibility. This is the same as the dims."""
        return self.dims

    @property
    def xdims(self):
        """:tuple: Tuple of indexing dimensions used to create xarray DataArray."""

        if self.ndim == 1:
            return (self.name,)
        else:
            return tuple("%s-%d" % (self.name, i) for i in range(1, self.ndim + 1))

    @property
    def ndim(self):
        """coordinates array ndim."""
        raise NotImplementedError

    @property
    def size(self):
        """coordinates array size."""
        raise NotImplementedError

    @property
    def shape(self):
        """coordinates array shape."""
        raise NotImplementedError

    @property
    def coordinates(self):
        """Coordinate values."""
        raise NotImplementedError

    @property
    def xcoords(self):
        """xarray coords"""
        raise NotImplementedError

    @property
    def definition(self):
        """Coordinates definition."""
        raise NotImplementedError

    @property
    def full_definition(self):
        """Coordinates definition, containing all properties. For internal use."""
        raise NotImplementedError

    @property
    def is_stacked(self):
        """stacked or unstacked property"""
        raise NotImplementedError

    @classmethod
    def from_definition(cls, d):
        """Get Coordinates from a coordinates definition."""
        raise NotImplementedError

    def copy(self):
        """Deep copy of the coordinates and their properties."""
        raise NotImplementedError

    def unique(self, return_index=False):
        """Remove duplicate coordinate values."""
        raise NotImplementedError

    def get_area_bounds(self, boundary):
        """Get coordinate area bounds, including boundary information, for each unstacked dimension."""
        raise NotImplementedError

    def select(self, bounds, outer=False, return_index=False):
        """Get coordinate values that are with the given bounds."""
        raise NotImplementedError

    def simplify(self):
        """Get the simplified/optimized representation of these coordinates."""
        raise NotImplementedError

    def flatten(self):
        """Get a copy of the coordinates with a flattened array."""
        raise NotImplementedError

    def reshape(self, newshape):
        """Get a copy of the coordinates with a reshaped array (wraps numpy.reshape)."""
        raise NotImplementedError

    def issubset(self, other):
        """Report if these coordinates are a subset of other coordinates."""
        raise NotImplementedError

    def horizontal_resolution(self, latitude, ellipsoid_tuple, coordinate_name, restype="nominal", units="meter"):
        """Get horizontal resolution of coordiantes."""
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    # python 2 compatibility
    if sys.version < "3":

        def __ne__(self, other):
            return not self.__eq__(other)
