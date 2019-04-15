from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict

import numpy as np
import traitlets as tl
import lazy_import
from six import string_types
import numbers

from podpac.core.settings import settings
from podpac.core.units import Units
from podpac.core.utils import ArrayTrait, TupleTrait
from podpac.core.coordinates.utils import Dimension, CoordinateType, CoordinateReferenceSystem
from podpac.core.coordinates.utils import make_coord_array, make_coord_value, make_coord_delta
from podpac.core.coordinates.base_coordinates import BaseCoordinates
from podpac.core.coordinates.coordinates1d import Coordinates1d
from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.stacked_coordinates import StackedCoordinates

class DependentCoordinates(BaseCoordinates):

    coordinates = TupleTrait(trait=ArrayTrait(), read_only=True)
    dims = TupleTrait(trait=Dimension(allow_none=True), read_only=True)
    idims = TupleTrait(trait=tl.Unicode(), read_only=True)
    units = TupleTrait(trait=tl.Instance(Units, allow_none=True), allow_none=True, read_only=True)
    ctypes = TupleTrait(trait=CoordinateType(), read_only=True)
    segment_lengths = TupleTrait(trait=tl.Any(allow_none=True), read_only=True)
    coord_ref_sys = CoordinateReferenceSystem(allow_none=True, read_only=True)

    _properties = tl.Set()
    
    def __init__(self, coordinates, dims=None, coord_ref_sys=None, units=None, ctypes=None, segment_lengths=None):
        coordinates = [np.array(a) for a in coordinates]
        coordinates = [make_coord_array(a.flatten()).reshape(a.shape) for a in coordinates]
        self.set_trait('coordinates', coordinates)
        self._set_properties(dims, coord_ref_sys, units, ctypes, segment_lengths)
        
    def _set_properties(self, dims, coord_ref_sys, units, ctypes, segment_lengths):
        if dims is not None:
            self.set_trait('dims', dims)
        if coord_ref_sys is not None:
            self._set_coord_ref_sys(coord_ref_sys)
        if units is not None:
            self._set_units(units)
        if ctypes is not None:
            self._set_ctype(ctypes)
        if segment_lengths is not None:
            self._set_segment_lengths(segment_lengths)
        else:
            self.segment_lengths # force validation

    @tl.default('dims')
    def _default_dims(self):
        return tuple(None for c in self.coordinates)

    @tl.default('idims')
    def _default_idims(self):
        return tuple('ijkl')[:self.ndims]

    @tl.default('ctypes')
    def _default_ctype(self):
        return tuple('point' for dim in self.dims)

    @tl.default('units')
    def _default_units(self):
        return tuple(None for dim in self.dims)

    @tl.default('segment_lengths')
    def _default_segment_lengths(self):
        return tuple(None for dim in self.dims)

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
        val = self._validate_sizes(d)
        for i, dim in enumerate(val):
            if dim is not None and dim in val[:i]:
                raise ValueError("Duplicate dimension '%s' in stacked coords" % dim)
        return val

    @tl.validate('segment_lengths')
    def _validate_segment_lengths(self, d):
        val = self._validate_sizes(d)
        for i, (segment_lengths, ctype) in enumerate(zip(val, self.ctypes)):
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
        if len(d['value']) != self.ndims:
            raise ValueError("%s and coordinates size mismatch, %d != %d" % (
                d['trait'].name, len(d['value']), self.ndims))
        return d['value']

    @tl.observe('dims', 'idims', 'ctypes', 'units', 'coord_ref_sys', 'segment_lengths')
    def _set_property(self, d):
        self._properties.add(d['name'])

    def _set_name(self, value):
        # only set if the dims have not been set already
        if 'dims' not in self._properties:
            dims = [dim.strip() for dim in value.split(',')]
            self.set_trait('dims', dims)
        elif self.name != value:
            raise ValueError("Dimension mismatch, %s != %s" % (value, self.name))

    def _set_coord_ref_sys(self, value):
        # set name if it is not set already, otherwise check that it matches
        if 'coord_ref_sys' not in self._properties:
            self.set_trait('coord_ref_sys', value)

        elif self.coord_ref_sys != value:
            raise ValueError("%s coord_ref_sys mismatch, %s != %s" % (
                self.__class__.__name__, value, self.coord_ref_sys))
    
    def _set_ctype(self, value):
        # only set ctypes if they are not set already
        if 'ctypes' not in self._properties:
            if isinstance(value, string_types):
                self.set_trait('ctypes', tuple(value for dim in self.dims))
            else:
                self.set_trait('ctypes', value)

    def _set_units(self, value):
        # only set units if is not not set already
        if 'units' not in self._properties:
            if isinstance(value, Units):
                self.set_trait('units', tuple(value for dim in self.dims))
            else:
                self.set_trait('units', value)

    def _set_distance_units(self, value):
        if 'units' not in self._properties:
            self.set_trait('units', [value if dim in ['lat', 'lon', 'alt'] else None for dim in self.dims])

    def _set_segment_lengths(self, value):
        if isinstance(value, numbers.Number):
            value = tuple(value for dim in self.dims)
        value = tuple(make_coord_delta(sl) if sl is not None else None for sl in value)
        self.set_trait('segment_lengths', value)

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
        if 'values' not in d:
            raise ValueError('DependentCoordinates definition requires "values" property')
        
        coordinates = d['values']
        kwargs = {k:v for k,v in d.items() if k not in ['values']}
        return DependentCoordinates(coordinates, **kwargs)

    # -----------------------------------------------------------------------------------------------------------------
    # standard methods
    # -----------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        rep = str(self.__class__.__name__)
        for i, dim in enumerate(self.dims):
            rep += '\n\t%s' % self.rep(dim, index=i)
        return rep

    def rep(self, dim, index=None):
        if dim is not None:
            index = self.dims.index(dim)
        else:
            dim = '?'# unnamed dimensions

        c = self.coordinates[index]
        ctype = self.ctypes[index]
        bounds = np.min(c), np.max(c)
        return "%s(%s->%s): Bounds[%s, %s], shape%s, ctype[%s]" % (
            self.__class__.__name__, ','.join(self.idims), dim, bounds[0], bounds[1], self.shape, ctype)

    def __eq__(self, other):
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
        if not np.array_equal(self.coordinates, other.coordinates):
            return False

        return True

    def __iter__(self):
        return iter(self[dim] for dim in self.dims)

    def __getitem__(self, index):
        if isinstance(index, string_types):
            dim = index
            if dim not in self.dims:
                raise KeyError("Cannot get dimension '%s' in RotatedCoordinates %s" % (dim, self.dims))
        
            i = self.dims.index(dim)
            return ArrayCoordinatesNd(self.coordinates[i], **self._properties_at(i))

        else:
            coordinates = tuple(a[index] for a in self.coordinates)
            # return DependentCoordinates(coordinates, **self.properties)

            # NOTE: this is optional, but we can convert to StackedCoordinates if ndim is 1
            if coordinates[0].ndim == 1:
                cs = [ArrayCoordinates1d(a, **self._properties_at(i)) for i, a in enumerate(coordinates)]
                return StackedCoordinates(cs)
            else:
                return DependentCoordinates(coordinates, **self.properties)

    def _properties_at(self, index=None, dim=None):
        if index is None:
            index = self.dims.index(dim)
        properties = {}
        properties['name'] = self.dims[index]
        if 'units' in self._properties:
            properties['units'] = self.units[index]
        if 'ctypes' in self._properties:
            properties['ctype'] = self.ctypes[index]
        if self.ctypes[index] != 'point':
            properties['segment_lengths'] = self.segment_lengths[index]
        if 'coord_ref_sys' in self._properties:
            properties['coord_ref_sys'] = self.coord_ref_sys
        return properties

    # -----------------------------------------------------------------------------------------------------------------
    # properties
    # -----------------------------------------------------------------------------------------------------------------

    @property
    def name(self):
        return '%s' % ','.join([dim or '?' for dim in self.dims])

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
    def dtypes(self):
        return tuple(c.dtype for c in self.coordinates)

    @property
    def bounds(self):
        """:dict: Dictionary of (low, high) coordinates bounds in each unstacked dimension"""
        if None in self.dims:
            raise ValueError("Cannot get bounds for DependentCoordinates with un-named dimensions")
        return {dim: self[dim].bounds for dim in self.dims}

    @property
    def area_bounds(self):
        """:dict: Dictionary of (low, high) coordinates area_bounds in each unstacked dimension"""
        if None in self.dims:
            raise ValueError("Cannot get area_bounds for DependentCoordinates with un-named dimensions")
        return {dim: self[dim].area_bounds for dim in self.dims}

    @property
    def coords(self):
        if None in self.dims:
            raise ValueError("Cannot get coords for DependentCoordinates with un-named dimensions")
        return {dim: (self.idims, c) for dim, c in (zip(self.dims, self.coordinates))}

    @property
    def properties(self):
        """:dict: Dictionary of the coordinate properties. """

        return {key:getattr(self, key) for key in self._properties}

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def copy(self):
        return DependentCoordinates(self.coordinates, **self.properties)

    def intersect(self, other, outer=False, return_indices=False):
        from podpac.core.coordinates.coordinates import Coordinates
        if not isinstance(other, (BaseCoordinates, Coordinates)):
            raise TypeError("Cannot intersect with type '%s'" % type(other))

        # bundle Coordinates1d object, if necessary
        if isinstance(other, Coordinates1d):
            other = Coordinates([other])

        # check for compatibility
        for dim, dtype, units in zip(self.dims, self.dtypes, self.units):
            if dim not in other:
                continue
            o = other.get(dim) # get other once
            if dtype is not None and o.dtype is not None and dtype != o.dtype:
                raise ValueError("Cannot intersect mismatched dtypes ('%s' != '%s')" % (dtype, o.dtype))
            if units != o.units:
                raise NotImplementedError("Still need to implement handling different units")
            if self.coord_ref_sys != o.coord_ref_sys:
                raise NotImplementedError("Still need to implement handling different CRS")

        return self.select(other.bounds, outer=outer, return_indices=return_indices)

    def select(self, bounds, outer=False, return_indices=False):
        # logical AND of selection in each dimension
        Is = [self._within(a, bounds.get(dim), outer) for dim, a in zip(self.dims, self.coordinates)]
        I = np.logical_and.reduce(Is)

        if np.all(I):
            return self._select_all(return_indices)
        
        if return_indices:
            return self[I], np.where(I)
        else:
            return self[I]

    def _within(self, coordinates, bounds, outer):
        if bounds is None:
            return np.ones(self.shape, dtype=bool)

        lo, hi = bounds
        lo = make_coord_value(lo)
        hi = make_coord_value(hi)

        if outer:
            below = coordinates[coordinates <= lo]
            above = coordinates[coordinates >= hi]
            lo = max(below) if below.size else -np.inf
            hi = min(above) if above.size else  np.inf
        
        gt = coordinates >= lo
        lt = coordinates <= hi
        return gt & lt

    def _select_all(self, return_indices):
        if return_indices:
            return self, slice(None)
        else:
            return self

    # ------------------------------------------------------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------------------------------------------------------
    
    # def plot(self, marker='b.'):
    #     from matplotlib import pyplot
    #     if self.ndims != 2:
    #         raise NotImplementedError("Only 2d DependentCoordinates plots are supported")
    #     x, y = self.coordinates
    #     pyplot.plot(x, y, marker)
    #     pyplot.xlabel(self.dims[0])
    #     pyplot.ylabel(self.dims[1])
    #     pyplot.axis('equal')

class ArrayCoordinatesNd(ArrayCoordinates1d):
    """
    Partial implementation for internal use.
    
    Provides name, dtype, units, size, bounds, area_bounds (and others).
    Prohibits coords, intersect, select (and others).

    Used primarily for intersection with DependentCoordinates.
    """

    coordinates = ArrayTrait(read_only=True)

    def __init__(self, coordinates,
                       name=None, ctype=None, units=None, segment_lengths=None, coord_ref_sys=None):

        self.set_trait('coordinates', coordinates)
        self._is_monotonic = None
        self._is_descending = None
        self._is_uniform = None

        Coordinates1d.__init__(self,
            name=name, ctype=ctype, units=units, segment_lengths=segment_lengths, coord_ref_sys=coord_ref_sys)

    def __repr__(self):
        return "%s(%s): Bounds[%s, %s], shape%s, ctype['%s']" % (
            self.__class__.__name__, self.name or '?', self.bounds[0], self.bounds[1], self.shape, self.ctype)

    @property
    def shape(self):
        return self.coordinates.shape

    # Restricted methods and properties

    @classmethod
    def from_xarray(cls, x):
        raise RuntimeError("ArrayCoordinatesNd from_xarray is unavailable.")

    @classmethod
    def from_definition(cls, d):
        raise RuntimeError("ArrayCoordinatesNd from_definition is unavailable.")

    @property
    def definition(self):
        raise RuntimeError("ArrayCoordinatesNd definition is unavailable.")

    @property
    def coords(self):
        raise RuntimeError("ArrayCoordinatesNd coords is unavailable.")

    def intersect(self, other, outer=False, return_indices=False):
        raise RuntimeError("ArrayCoordinatesNd intersect is unavailable.")

    def select(self, bounds, outer=False, return_indices=False):
        raise RuntimeError("ArrayCoordinatesNd select is unavailable.")