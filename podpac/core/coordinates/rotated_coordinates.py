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
    
    def __init__(self, affine, shape, dims=None, ctypes=None, units=None, segment_lengths=None, coord_ref_sys=None):
        self.set_trait('affine', affine)
        self.set_trait('shape', shape)

        # TODO dims
        # TODO ctypes
        # TODO units
        # TODO segment_lengths
        # TODO coord_ref_sys

    def __repr__(self):
        raise NotImplementedError("TODO")

    def __eq__(self):
        raise NotImplementedError("TODO")

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
        raise NotImplementedError("TODO")

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

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------
    
    def intersect(self, d):
        raise NotImplementedError("TODO")

    def copy(self):
        raise NotImplementedError("TODO")

    def plot(self, marker='b.', ulc_marker='bo'):
        from matplotlib import pyplot
        pyplot.plot(*self.coordinates, marker)
        pyplot.plot(*self.ulc, ulc_marker)
        # pyplot.xlabel(self.dims[0])
        # pyplot.ylabel(self.dims[1])
        pyplot.axis('equal')