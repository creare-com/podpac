from __future__ import division, unicode_literals, print_function, absolute_import

import traitlets as tl

class BaseCoordinates1d(tl.HasTraits):
    """
    Base class for single or stacked one-dimensional coordinates.

    Attributes
    ----------
    name : str
        Dimension name.
    dims : tuple
        Tuple of individual dimensions.
    coordinates : array
        Coordinate values.
    size : int
        Number of coordinates.
    is_monotonic : bool
        If the coordinates are monotonically increasing or decreasing.
    is_uniform : bool
        If the coordinates are uniformly spaced.
    """

    @property
    def name(self):
        raise NotImplementedError

    @property
    def dims(self):
        raise NotImplementedError

    @property
    def coordinates(self):
        raise NotImplementedError

    @property
    def size(self):
        raise NotImplementedError

    @property
    def is_monotonic(self):
        raise NotImplementedError

    @property
    def is_uniform(self):
        raise NotImplementedError

    def select(self, bounds, outer=False, return_indices=False):
        raise NotImplementedError

    def intersect(self, other, outer=False, return_indices=False):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.size