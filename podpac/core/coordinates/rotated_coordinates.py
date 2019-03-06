from __future__ import division, unicode_literals, print_function, absolute_import

from operator import mul
from collections import OrderedDict

import numpy as np
import traitlets as tl
import lazy_import
rasterio = lazy_import.lazy_module('rasterio')

from podpac.core.settings import settings
from podpac.core.units import Units
from podpac.core.coordinates.base_coordinates import BaseCoordinates
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates

IDIMS = 'ijkl'

class RotatedCoordinates(BaseCoordinates):
    """
    """

    shape = tl.Tuple(traits=tl.Integer(), read_only=True)
    affine = tl.Instance(rasterio.Affine, read_only=True)

    dims = tl.Tuple(traits=tl.Enum(['lat', 'lon', 'alt', 'time']), read_only=True)
    ctypes = tl.Tuple(traits=tl.Enum(['point', 'left', 'right', 'midpoint'], read_only=True))
    segment_lengths = tl.Tuple(traits=tl.Float(), read_only=True) # currently only uniform segments are supported
    units = tl.Instance(Units, allow_none=True, read_only=True)
    coord_ref_sys = tl.Enum(['WGS84', 'SPHER_MERC'], allow_none=True, read_only=True)

    _properties = tl.Set()
    _segment_lengths = tl.Bool()
    
    def __init__(self, affine, shape, dims=None, ctypes=None, units=None, segment_lengths=None, coord_ref_sys=None):
        self.set_trait('affine', affine)
        self.set_trait('shape', shape)
        self.set_trait('dims', dims)
        
        if units is not None:
            self.set_trait('units', units)

        if coord_ref_sys is not None:
            self.set_trait('coord_ref_sys', coord_ref_sys)

        if ctypes is not None:
            if isinstance(ctypes, str):
                ctyes = [ctypes for dim in self.dims]
            self.set_trait('ctypes', ctypes)

        if segment_lengths is not None:
            if isinstance(segment_lengths, numbers.Number):
                segment_lengths = [segment_length for dim in self.dims]
            self.set_trait('segment_lengths', segment_lengths)

    @tl.default('ctypes')
    def _default_ctype(self):
        return ['midpoint' for dim in self.dims]

    @tl.default('segment_lengths')
    def _default_segment_lengths(self):
        return [step if ctype != 'point' else None for step, ctype in zip(self.step, self.ctypes)]

    @tl.default('coord_ref_sys')
    def _default_coord_ref_sys(self):
        return settings['DEFAULT_COORD_REF_SYS']

    @tl.validate('shape')
    def _validate_shape(self, d):
        val = d['value']
        if len(val) != 2:
            raise ValueError("Invalid shape %s, must have ndim 2, not %d" % val, len(val))
        return val

    @tl.validate('dims', 'ctype', 'segment_lengths')
    def _validate_dims(self, d):
        val = d['value']
        if len(val) != len(self.shape):
            raise ValueError("%s value %s does not match shape %s (%d != %d)" % (
                d['trait'].name, val, shape, len(val), len(shape)))
        return val

    @tl.observe('ctypes', 'units', 'coord_ref_sys')
    def _set_property(self, d):
        self._properties.add(d['name'])

    @tl.observe('segment_lengths')
    def _set_segment_lengths(self, d):
        self._segment_lengths = True

    def __repr__(self):
        if len(set(self.ctypes)) == 1:
            ctypes = "ctype['%s']" % self.ctypes[0]
        else:
            ctypes = "ctypes[%s]" % ', '.join(self.ctypes)

        return "%s(%s): ULC%s, LRC%s, rad[%.4f], shape%s, %s" % (
            self.__class__.__name__, self.dims, self.ulc, self.lrc, self.theta, self.shape, ctypes)

    def __eq__(self):
        if not isinstance(other, RotatedCoordinates1d):
            return False

        if self.affine != other.affine:
            return False
        
        if self.shape != other.shape:
            return False
        
        # defined coordinate properties should also match
        for name in self._properties.union(other._properties):
            if getattr(self, name) != getattr(other, name):
                return False

        # only check segment_lengths if one of the coordinates has custom segment lengths
        if self._segment_lengths or other._segment_lengths:
            if not np.all(self.segment_lengths == other.segment_lengths):
                return False

        return True

    # ------------------------------------------------------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def definition(self):
        raise NotImplementedError("TODO")

    @classmethod
    def from_definition(cls, d):
        raise NotImplementedError("TODO")

    # ------------------------------------------------------------------------------------------------------------------
    # Alternate Constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_geotransform(cls, geotransform, shape,
                          dims=None, ctypes=None, units=None, segment_lengths=None, coord_ref_sys=None):
        affine = rasterio.Affine.from_gdal(geotransform)
        return cls(affine, shape,
                   dims=dims, ctypes=ctypes, units=units, segment_lengths=segment_lengths, coord_ref_sys=coord_ref_sys)

    @classmethod
    def from_corners(cls, ulc, lrc, theta, shape,
                     dims=None, ctypes=None, units=None, segment_lengths=None, coord_ref_sys=None):
        deg = np.rad2deg(theta)
        r = ~rasterio.Affine.rotation(deg)
        d = r * ulc - r * lrc
        step = d / shape
        return cls.from_ulc_step(ulc, step, theta, shape)

    @classmethod
    def from_ulc_and_step(cls, ulc, step, theta, shape,
                          dims=None, ctypes=None, units=None, segment_lengths=None, coord_ref_sys=None):
        deg = np.rad2deg(theta)
        affine = rasterio.Affine.translation(*ulc) * rasterio.Affine.rotation(deg) * rasterio.Affine.scale(*step)
        return RotatedCoordinates(
            affine, shape,
            dims=dims, ctypes=ctypes, units=units, segment_lengths=segment_lengths, coord_ref_sys=coord_ref_sys)

    # ------------------------------------------------------------------------------------------------------------------
    # Standard methods, array-like
    # ------------------------------------------------------------------------------------------------------------------

    def __getitem__(self, index):
        if isinstance(index, slice) or (isinstance(index, tuple) and all(isinstance(I, slice) for I in index)):
            # TODO calculate new ulc and shape without calculating coordinates
            values = np.array(self.coordinates).T[index]
            ulc = [a.flatten()[0] for a in values.T]
            shape = values.shape[1:]
            return RotatedCoordinates.from_ulc_and_step(ulc, self.step, self.theta, shape)
        else:
            values = np.array(self.coordinates).T[index].T
            cs = [ArrayCoordinates1d(a.flatten(), name=dim) for a, dim in zip(values, self.dims)]
            return StackedCoordinates(cs)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def idims(self):
        return list(IDIMS)[:self.ndim]

    @property
    def size(self):
        return mul(self.shape)

    @property
    def theta(self):
        return np.deg2rad(self.affine.rotation_angle)

    @property
    def ulc(self):
        return np.array([self.affine.c, self.affine.f])

    @property
    def lrc(self):
        # TODO calculate without calculating coordinates
        return np.array([a.flatten()[-1] for a in self.coordinates])

    @property
    def step(self):
        scale = self.scale
        return (scale.a, scale.e)

    @property
    def rotation(self):
        return rasterio.Affine.rotation(np.rad2deg(self.theta))

    @property
    def translation(self):
        return rasterio.Affine.translation(*self.ulc)

    @property
    def scale(self):
        return ~self.rotation * ~self.translation * self.affine

    @property
    def geotransform(self):
        return self.affine.to_gdal()

    @property
    def coordinates(self):
        i = np.arange(self.shape[0])
        j = np.arange(self.shape[1])
        return self.affine * np.meshgrid(i, j)

    @property
    def coords(self):
        idims = self.idims
        coords = OrderedDict()
        for dim, c in zip(self.dims, self.coordinates):
            coords[dim] = (idims, c.T)
        return coords

    @property
    def properties(self):
        """:dict: Dictionary of the coordinate properties. """

        return {key:getattr(self, key) for key in self._properties}

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------
    
    def intersect(self, d):
        raise NotImplementedError("TODO")

    def copy(self):
        kwargs = self.properties
        if self._segment_lengths:
            kwargs['segment_lengths'] = self.segment_lengths
        return RotatedCoordinates(self.affine, self.shape, **kwargs)

    def plot(self, marker='b.', ulc_marker='bo'):
        from matplotlib import pyplot
        pyplot.plot(*self.coordinates, marker)
        pyplot.plot(*self.ulc, ulc_marker)
        # pyplot.xlabel(self.dims[0])
        # pyplot.ylabel(self.dims[1])
        pyplot.axis('equal')