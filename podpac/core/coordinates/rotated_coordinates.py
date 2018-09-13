from __future__ import division, unicode_literals, print_function, absolute_import

import traitlets as tl
from podpac.core.coordinates.base_coordinates import BaseCoordinates

class RotatedCoordinates(BaseCoordinates):
    """
    Base class for single or stacked one-dimensional coordinates.

    Attributes
    ----------
    TODO
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

    def __repr__(self):
        raise NotImplementedError