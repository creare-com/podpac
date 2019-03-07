from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict

import numpy as np
import traitlets as tl
import lazy_import
rasterio = lazy_import.lazy_module('rasterio')

from podpac.core.settings import settings
from podpac.core.units import Units
from podpac.core.coordinates.utils import Dimension, CoordinateType, CoordinateReferenceSystem
from podpac.core.coordinates.base_coordinates import BaseCoordinates
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates

# TODO integration with Coordinates
# TODO serialization
# TODO intersect
# TODO maybe the dims are always lat, lon (or lon, lat) but the idims may need to be reversed sometimes?
# TODO: require theta, ulc/translation, and step/scale in init (instead of affine)
#       - some affine transformations are not supported here (e.g. skew)
#       - this would also fix a bug with negative step
#       - we could generalize and make an AffineCoordinates base class

class RotatedCoordinates(BaseCoordinates):
    """
    Notes
    -----
     * Only uniform segment lengths are currently supported.
    """

    affine = tl.Instance(rasterio.Affine, read_only=True)
    shape = tl.Tuple(tl.Integer(), tl.Integer(), read_only=True)
    dims = tl.Tuple(Dimension(), Dimension(), read_only=True)
    idims = tl.Tuple(tl.Unicode(), tl.Unicode(), default_value=('i', 'j'), read_only=True)
    ctypes = tl.Tuple(CoordinateType(), CoordinateType(), read_only=True)
    segment_lengths = tl.Tuple(tl.Float(allow_none=True), tl.Float(allow_none=True), read_only=True)
    units = tl.Instance(Units, allow_none=True, read_only=True)
    coord_ref_sys = CoordinateReferenceSystem(allow_none=True, read_only=True)

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
                ctypes = (ctypes, ctypes)
            self.set_trait('ctypes', ctypes)

        if segment_lengths is not None:
            if isinstance(segment_lengths, numbers.Number):
                segment_lengths = (segment_lengths, segment_lengths)
            self.set_trait('segment_lengths', segment_lengths)

    @tl.default('ctypes')
    def _default_ctype(self):
        return ('midpoint', 'midpoint')

    @tl.default('segment_lengths')
    def _default_segment_lengths(self):
        sli = np.abs(self.step[0]) if self.ctypes[0] != 'point' else None
        slj = np.abs(self.step[1]) if self.ctypes[1] != 'point' else None
        return (sli, slj)

    @tl.default('coord_ref_sys')
    def _default_coord_ref_sys(self):
        return settings['DEFAULT_COORD_REF_SYS']

    @tl.validate('segment_lengths')
    def _validate_segment_lengths_tuple(self, d):
        val = d['value']
        self._validate_segment_lengths(0, self.dims[0], self.ctypes[0], val[0])
        self._validate_segment_lengths(1, self.dims[1], self.ctypes[1], val[1])
        return val

    def _validate_segment_lengths(self, i, dim, ctype, segment_lengths):
        if segment_lengths is None:
            if ctype != 'point':
                raise TypeError("segment_lengths cannot be None for '%s' coordinates in dim %d '%s'" % (ctype, i, dim))
        else:
            if ctype == 'point':
                raise TypeError("segment_lengths must be None for '%s' coordinates in dim %d '%s'" % (i, dim))
            if segment_lengths <= 0.0:
                raise ValueError("segment_lengths must be positive in dim %d '%s'" % (i, dim))

    @tl.observe('dims', 'ctypes', 'units', 'coord_ref_sys')
    def _set_property(self, d):
        self._properties.add(d['name'])

    @tl.observe('segment_lengths')
    def _set_segment_lengths(self, d):
        self._segment_lengths = True

    def __repr__(self):
        if self.ctypes[0] == self.ctypes[1]:
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

        if self.dims != other.dims:
            return False
        
        # defined coordinate properties should also match
        for name in self._properties.union(other._properties):
            if getattr(self, name) != getattr(other, name):
                return False

        # only check segment_lengths if one of the coordinates has custom segment lengths
        if self._segment_lengths or other._segment_lengths:
            if self.segment_lengths != other.segment_lengths:
                return False

        return True

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
    # Serialization
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def definition(self):
        raise NotImplementedError("TODO")

    @classmethod
    def from_definition(cls, d):
        raise NotImplementedError("TODO")

    # ------------------------------------------------------------------------------------------------------------------
    # Standard methods, array-like
    # ------------------------------------------------------------------------------------------------------------------

    def __getitem__(self, index):
        if isinstance(index, slice):
            index = index, slice(None)

        if isinstance(index, tuple) and isinstance(index[0], slice) and isinstance(index[1], slice):
            I = np.arange(self.shape[0])[index[0]]
            J = -np.arange(self.shape[1])[index[1]]
            import ipdb; ipdb.set_trace() # BREAKPOINT
            ulc = self.affine * (I[0], J[0])
            shape = I.size, J.size
            step = self.step[0] * (index[0].step or 1), self.step[1] * (index[1].step or 1)
            return RotatedCoordinates.from_ulc_and_step(ulc, step, self.theta, shape, **self.properties)
        else:
            values = np.array(self.coordinates).T[index].T
            cs = [
                ArrayCoordinates1d(
                    a.flatten(),
                    name=self.dims[i],
                    ctype=self.ctypes[i] if 'ctypes' in self._properties else None,
                    segment_lengths=self.segment_lengths[i] if self._segment_lengths else None,
                    units=self.units if 'units' in self._properties else None,
                    coord_ref_sys=self.coord_ref_sys if 'coord_ref_sys' in self._properties else None)
                for i, a in enumerate(values)
            ]
            return StackedCoordinates(cs)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def size(self):
        return self.shape[0] * self.shape[1]

    @property
    def theta(self):
        return np.deg2rad(self.affine.rotation_angle)

    @property
    def ulc(self):
        return np.array([self.affine.c, self.affine.f])

    @property
    def lrc(self):
        return self.affine * np.array([self.shape[0]-1, -(self.shape[1]-1)])

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
        I = np.arange(self.shape[0])
        J = -np.arange(self.shape[1])
        return self.affine * np.meshgrid(I, J)

    @property
    def coords(self):
        x, y = self.coordinates
        coords = {
            self.dims[0]: (self.idims, x.T),
            self.dims[1]: (self.idims, y.T)
        }
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
        x, y = self.coordinates
        ulcx, ulcy = self.ulc
        lrcx, lrcy = self.lrc
        pyplot.plot(x, y, marker)
        pyplot.plot(ulcx, ulcy, ulc_marker)
        pyplot.plot(lrcx, lrcy, 'bx')
        pyplot.xlabel(self.dims[0])
        pyplot.ylabel(self.dims[1])
        pyplot.axis('equal')