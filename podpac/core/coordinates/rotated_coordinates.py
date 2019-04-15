from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict

import numpy as np
import traitlets as tl
import lazy_import
rasterio = lazy_import.lazy_module('rasterio')

from podpac.core.utils import ArrayTrait
from podpac.core.coordinates.dependent_coordinates import DependentCoordinates

class RotatedCoordinates(DependentCoordinates):
    theta = tl.Float(read_only=True)
    ulc = ArrayTrait(shape=(2,), dtype=float, read_only=True)
    step = ArrayTrait(shape=(2,), dtype=float, read_only=True)
    shape = tl.Tuple(tl.Integer(), tl.Integer(), read_only=True)
    ndims = 2

    def __init__(self, shape=None, theta=None, ulc=None, step=None, lrc=None,
                 dims=None, ctypes=None, units=None, segment_lengths=None, coord_ref_sys=None):
        self.set_trait('shape', shape)
        self.set_trait('theta', theta)
        self.set_trait('ulc', ulc)
        if step is None:
            deg = np.rad2deg(theta)
            a = ~rasterio.Affine.rotation(deg) * ~rasterio.Affine.translation(*ulc)
            d = np.array(a * lrc) - np.array(a * ulc)
            step = d / np.array([shape[0]-1, -(shape[1]-1)])
        self.set_trait('step', step)

        self._set_properties(dims, coord_ref_sys, units, ctypes, segment_lengths)

    @tl.validate('dims')
    def _validate_dims(self, d):
        val = super(RotatedCoordinates, self)._validate_dims(d)
        for dim in val:
            if dim not in ['lat', 'lon']:
                raise ValueError("RotatedCoordinates dims must be 'lat' or 'lon', not '%s'" % dim)
        return val

    @tl.validate('shape')
    def _validate_shape(self, d):
        val = d['value']
        if val[0] <= 0 or val[1] <= 0:
            raise ValueError("Invalid shape %s, shape must be positive" % (val,))
        return val

    @tl.validate('step')
    def _validate_step(self, d):
        val = d['value']
        if val[0] == 0 or val[1] == 0:
            raise ValueError("Invalid step %s, step cannot be 0" % val)
        return val

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate Constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_geotransform(cls, geotransform, shape, dims=None,
                               ctypes=None, units=None, segment_lengths=None, coord_ref_sys=None):
        affine = rasterio.Affine.from_gdal(*geotransform)
        ulc = affine.c, affine.f
        deg = affine.rotation_angle
        scale = ~affine.rotation(deg) * ~affine.translation(*ulc) * affine
        step = np.array([scale.a, scale.e])
        return cls(shape, np.deg2rad(deg), ulc, step, dims=dims,
                   ctypes=ctypes, units=units, segment_lengths=segment_lengths, coord_ref_sys=coord_ref_sys)

    # ------------------------------------------------------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def definition(self):
        d = OrderedDict()
        d['shape'] = self.shape
        d['theta'] = self.theta
        d['ulc'] = self.ulc
        d['step'] = self.step
        d['dims'] = self.dims
        d.update(self.properties)
        return d

    @classmethod
    def from_definition(cls, d):
        if 'shape' not in d:
            raise ValueError('RotatedCoordinates definition requires "shape" property')
        if 'theta' not in d:
            raise ValueError('RotatedCoordinates definition requires "theta" property')
        if 'ulc' not in d:
            raise ValueError('RotatedCoordinates definition requires "ulc" property')
        if 'step' not in d and 'lrc' not in d:
            raise ValueError('RotatedCoordinates definition requires "step" or "lrc" property')
        if 'dims' not in d:
            raise ValueError('RotatedCoordinates definition requires "dims" property')

        shape = d['shape']
        theta = d['theta']
        ulc = d['ulc']
        kwargs = {k:v for k,v in d.items() if k not in ['shape', 'theta', 'ulc']}
        return RotatedCoordinates(shape, theta, ulc, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    # standard methods
    # ------------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        if self.ctypes[0] == self.ctypes[1]:
            ctypes = "ctype['%s']" % self.ctypes[0]
        else:
            ctypes = "ctypes[%s]" % ', '.join(self.ctypes)

        return "%s(%s): ULC%s, LRC%s, rad[%.4f], shape%s, %s" % (
            self.__class__.__name__, self.dims, self.ulc, self.lrc, self.theta, self.shape, ctypes)

    def __eq__(self, other):
        if not isinstance(other, RotatedCoordinates):
            return False

        if self.shape != other.shape:
            return False

        if self.affine != other.affine:
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
            I = np.arange(self.shape[0])[index[0]]
            J = -np.arange(self.shape[1])[index[1]]
            ulc = self.affine * [I[0], J[0]]
            step = self.step * [index[0].step or 1, index[1].step or 1]
            shape = I.size, J.size
            return RotatedCoordinates(shape, self.theta, ulc, step, **self.properties)

        else:
            return super(RotatedCoordinates, self).__getitem__(index)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def deg(self):
        return np.rad2deg(self.theta)

    @property
    def affine(self):
        t = rasterio.Affine.translation(*self.ulc)
        r = rasterio.Affine.rotation(self.deg)
        s = rasterio.Affine.scale(*self.step)
        return t * r * s

    @property
    def lrc(self):
        return np.array(self.affine * np.array([self.shape[0]-1, -(self.shape[1]-1)]))

    @property
    def geotransform(self):
        return self.affine.to_gdal()

    @property
    def coordinates(self):
        I = np.arange(self.shape[0])
        J = -np.arange(self.shape[1])
        c1, c2 = self.affine * np.meshgrid(I, J)
        return c1.T, c2.T

    @property
    def properties(self):
        """:dict: Dictionary of the coordinate properties. """
        return {key:getattr(self, key) for key in self._properties}

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def copy(self):
        return RotatedCoordinates(self.shape, self.theta, self.ulc, self.step, **self.properties)

    def intersect(self, other, outer=False, return_indices=False):
        # TODO return RotatedCoordinates when possible
        return super(RotatedCoordinates, self).intersect(other, outer=outer, return_indices=return_indices)

    def select(self, bounds, outer=False, return_indices=False):
        # TODO return RotatedCoordinates when possible
        return super(RotatedCoordinates, self).select(bounds, outer=outer, return_indices=return_indices)

    # ------------------------------------------------------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------------------------------------------------------
    
    # def plot(self, marker='b.', ulc_marker='bo', lrc_marker='bx'):
    #     from matplotlib import pyplot
    #     super(RotatedCoordinates, self).plot(marker=marker)
    #     ulcx, ulcy = self.ulc
    #     lrcx, lrcy = self.lrc
    #     pyplot.plot(ulcx, ulcy, ulc_marker)
    #     pyplot.plot(lrcx, lrcy, 'bx')