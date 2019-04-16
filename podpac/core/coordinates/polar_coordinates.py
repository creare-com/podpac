from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict

import numpy as np
import traitlets as tl
import lazy_import
rasterio = lazy_import.lazy_module('rasterio')

from podpac.core.utils import ArrayTrait
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d
from podpac.core.coordinates.dependent_coordinates import DependentCoordinates

class PolarCoordinates(DependentCoordinates):
    center = ArrayTrait(shape=(2,), dtype=float, read_only=True)
    radius = tl.Instance(Coordinates1d, read_only=True)
    theta = tl.Instance(Coordinates1d, read_only=True)
    ndims = 2

    def __init__(self, center, radius, theta=None, theta_size=None,
                 dims=None, ctypes=None, units=None, segment_lengths=None, crs=None):
        
        # radius
        if not isinstance(radius, Coordinates1d):
            radius = ArrayCoordinates1d(radius)
        
        # theta
        if theta is not None and theta_size is not None:
            raise TypeError("PolarCoordinates expected theta or theta_size, not both.")
        if theta is None and theta_size is None:
            raise TypeError("PolarCoordinates requires theta or theta_size.")

        if theta_size is not None:
            theta = UniformCoordinates1d(start=0, stop=2*np.pi, size=theta_size+1)[:-1]
        elif not isinstance(theta, Coordinates1d):
            theta = ArrayCoordinates1d(theta)

        self.set_trait('center', center)
        self.set_trait('radius', radius)
        self.set_trait('theta', theta)

        # properties
        self._set_properties(dims, crs, units, ctypes, segment_lengths)

    @tl.validate('dims')
    def _validate_dims(self, d):
        val = super(PolarCoordinates, self)._validate_dims(d)
        if val != ('lat', 'lon'):
            raise ValueError("PolarCoordinates dims must be ('lat', 'lon'), not '%s'" % (val,))
        return val

    @tl.validate('radius')
    def _validate_radius(self, d):
        val = d['value']
        if np.any(val.coordinates <= 0):
            raise ValueError("PolarCoordinates radius must all be positive")
        return val

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate Constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_definition(cls, d):
        if 'center' not in d:
            raise ValueError('PolarCoordinates definition requires "center" property')
        if 'radius' not in d:
            raise ValueError('PolarCoordinates definition requires "radius" property')
        if 'theta' not in d and 'theta_size' not in d:
            raise ValueError('PolarCoordinates definition requires "theta" or "theta_size" property')
        if 'dims' not in d:
            raise ValueError('PolarCoordinates definition requires "dims" property')

        # center
        center = d['center']
        
        # radius
        if isinstance(d['radius'], list):
            radius = ArrayCoordinates1d(d['radius'])
        elif 'values' in d['radius']:
            radius = ArrayCoordinates1d.from_definition(d['radius'])
        elif 'start' in d['radius'] and 'stop' in d['radius'] and ('step' in d['radius'] or 'size' in d['radius']):
            radius = UniformCoordinates1d.from_definition(d['radius'])
        else:
            raise ValueError("Could not parse radius coordinates definition with keys %s" % d.keys())
        
        # theta
        if 'theta' not in d:
            theta = None
        elif isinstance(d['theta'], list):
            theta = ArrayCoordinates1d(d['theta'])
        elif 'values' in d['theta']:
            theta = ArrayCoordinates1d.from_definition(d['theta'])
        elif 'start' in d['theta'] and 'stop' in d['theta'] and ('step' in d['theta'] or 'size' in d['theta']):
            theta = UniformCoordinates1d.from_definition(d['theta'])
        else:
            raise ValueError("Could not parse theta coordinates definition with keys %s" % d.keys())

        kwargs = {k:v for k,v in d.items() if k not in ['center', 'radius', 'theta']}
        return PolarCoordinates(center, radius, theta, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    # standard methods
    # ------------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        if self.ctypes[0] == self.ctypes[1]:
            ctypes = "ctype['%s']" % self.ctypes[0]
        else:
            ctypes = "ctypes[%s]" % ', '.join(self.ctypes)

        return "%s(%s): center%s, shape%s, %s" % (
            self.__class__.__name__, self.dims, self.center, self.shape, ctypes)

    def __eq__(self, other):
        if not isinstance(other, PolarCoordinates):
            return False

        if not np.allclose(self.center, other.center):
            return False
        
        if self.radius != other.radius:
            return False

        if self.theta != other.theta:
            return False

        # defined coordinate properties should match
        for name in self._properties.union(other._properties):
            if getattr(self, name) != getattr(other, name):
                return False

        return True

    def __getitem__(self, index):
        if isinstance(index, slice):
            index = index, slice(None)

        if isinstance(index, tuple) and isinstance(index[0], slice) and isinstance(index[1], slice):
            return PolarCoordinates(self.center, self.radius[index[0]], self.theta[index[1]], **self.properties)
        else:
            return super(PolarCoordinates, self).__getitem__(index)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def shape(self):
        return self.radius.size, self.theta.size

    @property
    def idims(self):
        return ('r', 't')

    @property
    def coordinates(self):
        r, theta = np.meshgrid(self.radius.coordinates, self.theta.coordinates)
        lat = r * np.sin(theta) + self.center[0]
        lon = r * np.cos(theta) + self.center[1]
        return lat.T, lon.T

    @property
    def properties(self):
        """:dict: Dictionary of the coordinate properties. """
        return {key:getattr(self, key) for key in self._properties}

    def _get_definition(self, full=True):
        d = OrderedDict()
        d['dims'] = self.dims
        d['center'] = self.center
        d['radius'] = self.radius.definition
        d['theta'] = self.theta.definition
        d.update(self._full_properties if full else self.properties)
        return d

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def copy(self):
        return PolarCoordinates(self.center, self.radius, self.theta, **self.properties)

    # TODO return PolarCoordinates when possible
    # def intersect(self, other, outer=False):
    #     raise NotImplementedError("TODO")

    # TODO return PolarCoordinates when possible
    # def select(self, other, outer=False):
    #     raise NotImplementedError("TODO")

    # ------------------------------------------------------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------------------------------------------------------
    
    # def plot(self, marker='b.', center_marker='bx'):
    #     from matplotlib import pyplot
    #     super(PolarCoordinates, self).plot(marker=marker)
    #     cx, cy = self.center
    #     pyplot.plot(cx, cy, center_marker)