from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict

import numpy as np
import traitlets as tl
import lazy_import

from podpac.core.settings import settings
from podpac.core.units import Units
from podpac.core.utils import ArrayTrait, TupleTrait
from podpac.core.coordinates.utils import Dimension, CoordinateType, CoordinateReferenceSystem
from podpac.core.coordinates.base_coordinates import BaseCoordinates
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinatesShaped

# TODO: integration with Coordinates
# TODO: serialization
# TODO: intersect



class DependentCoordinates(BaseCoordinates):

    coordinates = TupleTrait(trait=ArrayTrait(), read_only=True)
    dims = TupleTrait(trait=Dimension(), read_only=True)
    idims = TupleTrait(trait=tl.Unicode(), read_only=True)
    units = TupleTrait(trait=tl.Instance(Units, allow_none=True), allow_none=True, read_only=True)
    ctypes = TupleTrait(trait=CoordinateType(), read_only=True)
    segment_lengths = TupleTrait(trait=tl.Float(allow_none=True), read_only=True)
    coord_ref_sys = CoordinateReferenceSystem(allow_none=True, read_only=True)

    _properties = tl.Set()
    
    def __init__(self, coordinates, dims=None, coord_ref_sys=None, units=None, ctypes=None, segment_lengths=None):
        self.set_trait('coordinates', coordinates)
        self._set_properties(dims, coord_ref_sys, units, ctypes, segment_lengths)

    def _set_properties(self, dims, coord_ref_sys, units, ctypes, segment_lengths):
        self.set_trait('dims', dims)
        if coord_ref_sys is not None:
            self.set_trait('coord_ref_sys', coord_ref_sys)
        if units is not None:
            if isinstance(units, Units):
                units = tuple(units for dim in self.dims)
            self.set_trait('units', units)
        if ctypes is not None:
            if isinstance(ctypes, str):
                ctypes = tuple(ctypes for dim in self.dims)
            self.set_trait('ctypes', ctypes)
        if segment_lengths is not None:
            if isinstance(segment_lengths, numbers.Number):
                segment_lengths = tuple(segment_lengths for dim in self.dims)
            self.set_trait('segment_lengths', segment_lengths)

    @tl.default('idims')
    def _default_idims(self):
        return tuple('ijkl')[:self.ndims]

    @tl.default('ctypes')
    def _default_ctype(self):
        return tuple('midpoint' for dim in self.dims)

    @tl.default('coord_ref_sys')
    def _default_coord_ref_sys(self):
        return settings['DEFAULT_COORD_REF_SYS']

    @tl.validate('coordinates')
    def _validate_coordinates(self, d):
        val = d['value']
        if len(val) == 0:
            raise ValueError("Dependent coordinates cannot be empty")

        for i, a in enumerate(val):
            if a.shape != val[0].shape:
                raise ValueError("coordinates shape mismatch at position %d, %s != %s" % (
                    i, a.shape, val[0].shape))
        return val

    @tl.validate('dims')
    def _validate_dims(self, d):
        print("validae dims")
        val = self._validate_sizes(d)
        if len(set(val)) != len(val):
            raise ValueError("Duplicate dimension in dims list %s" % val)
        return val

    @tl.validate('segment_lengths')
    def _validate_segment_lengths(self, d):
        val = self._validate_sizes(d)
        for i, ctype in enumerate(self.ctypes):
            if segment_lengths is None:
                if ctype != 'point':
                    raise TypeError("segment_lengths cannot be None for '%s' coordinates at position %d" % (ctype, i))
            else:
                if ctype == 'point':
                    raise TypeError("segment_lengths must be None for '%s' coordinates at position %d" % (ctype, i))
                if segment_lengths <= 0.0:
                    raise ValueError("segment_lengths must be positive at pos %d" % i)
        return val

    @tl.validate('idims', 'ctypes', 'units')
    def _validate_sizes(self, d):
        print("validate sizes")
        if len(d['value']) != self.ndims:
            raise ValueError("coordinates and %s size mismatch, %d != %d" % (
                d['trait'].name, len(d['value']), self.ndims))
        return d['value']

    @tl.observe('dims', 'idims', 'ctypes', 'units', 'coord_ref_sys', 'segment_lengths')
    def _set_property(self, d):
        self._properties.add(d['name'])

    # ------------------------------------------------------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def definition(self):
        d = OrderedDict()
        d['dims'] = self.dims
        d['values'] = self.coordinates
        d.update(self.properties)
        return d

    @classmethod
    def from_definition(cls, d):
        if 'dims' not in d:
            raise ValueError('DependentCoordinates definition requires "dims" property')
        if 'values' not in d:
            raise ValueError('DependentCoordinates definition requires "values" property')
        
        coordinates = d['values']
        kwargs = {k:v for k,v in d.items() if k not in ['values']}
        return DependentCoordinates(coordinates, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    # standard methods
    # -----------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        if self.ctypes[0] == self.ctypes[1]:
            ctypes = "ctype['%s']" % self.ctypes[0]
        else:
            ctypes = "ctypes[%s]" % ', '.join(self.ctypes)

        return "%s(%s): shape%s, %s" % (
            self.__class__.__name__, self.dims, self.shape, ctypes)

    def __eq__(self):
        if not isinstance(other, DependentCoordinates):
            return False

        # shortcut
        if self.shape != other.shape:
            return False

        # defined coordinate properties should match
        for name in self._properties.union(other._properties):
            if getattr(self, name) != getattr(other, name):
                return False

        # full coordinates check
        if self.coordinates != other.coordinates:
            return False

        return True

    def __iter__(self):
        return tuple(getattr(self, dim) for dim in self.dims)

    def _get(self, dim):
        if dim not in self.dims:
            raise KeyError("Cannot get dimension '%s' in RotatedCoordinates %s" % (dim, self.dims))
        
        idx = self.dims.index(idx)
        coordinates = self.coordinates[idx]
        properties = {}
        if 'units' in self.properties:
            properties['units'] = self.units[idx]
        if 'ctypes' in self.properties:
            properties['ctype'] = self.ctypes[idx]
        if self.ctypes[idx] != 'point':
            properties['segment_lengths'] = self.segment_lengths[idx]
        if 'coord_ref_sys' in self.properties:
            properties['coord_ref_sys'] = self.coord_ref_sys

        return ArrayCoordinatesShaped(coordinates, name=dim, **properties)

    def __getitem__(self, index):
        if isinstance(index, str):
            return self._get(index)

        else:
            coordinates = tuple(a for a in np.array(self.coordinates).T[index].T)
            return DependentCoordinates(coordinates, **self.properties)

    # -----------------------------------------------------------------------------------------------------------------
    # properties
    # -----------------------------------------------------------------------------------------------------------------

    @property
    def name(self):
        return ','.join(self.dims)

    @property
    def udims(self):
        return self.dims

    @property
    def shape(self):
        return self.coordinates[0].shape

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def ndims(self):
        return len(self.coordinates)

    @property
    def coords(self):
        return dict(zip(self.dims, self.coordinates))

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def copy(self):
        return DependentCoordinates(self.coordinates, **self.properties)

    def intersect(self, other, outer=False):
        raise NotImplementedError("TODO")

    # ------------------------------------------------------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------------------------------------------------------
    
    def plot(self, marker='b.'):
        from matplotlib import pyplot
        if self.ndims != 2:
            raise NotImplementedError("Only 2d DependentCoordinates plots are supported")
        x, y = self.coordinates
        pyplot.plot(x, y, marker)
        pyplot.xlabel(self.dims[0])
        pyplot.ylabel(self.dims[1])
        pyplot.axis('equal')