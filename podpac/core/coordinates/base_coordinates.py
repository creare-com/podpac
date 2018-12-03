from __future__ import division, unicode_literals, print_function, absolute_import

import traitlets as tl

class BaseCoordinates(tl.HasTraits):
    """Base class for single or stacked one-dimensional coordinates."""

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
        """:tuple: Dimensions."""
        raise NotImplementedError

    @property
    def coordinates(self):
        """Coordinate values."""
        raise NotImplementedError

    @property
    def size(self):
        """Number of coordinates."""
        raise NotImplementedError

    @property
    def definition(self):
        """Coordinates definition."""
        raise NotImplementedError

    def from_definition(self, d):
        """Get Coordinates from a coordinates definition."""
        raise NotImplementedError

    def intersect(self, other, outer=False, return_indices=False):
        """Get coordinate values that are with the bounds of other Coordinates."""
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.size

    def __repr__(self):
        raise NotImplementedError