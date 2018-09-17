# TODO spec here is uncertain, not in use yet
# class CoordinateGroup(BaseCoordinates):
#     """CoordinateGroup Summary
#     """
    
#     # TODO list or array?
#     _items = tl.List(trait=tl.Instance(Coordinates))

#     @tl.validate('_items')
#     def _validate_items(self, d):
#         items = d['value']
#         if not items:
#             return items

#         # unstacked dims must match, but not necessarily in order
#         dims = set(items[0].dims_map)
#         for g in items:
#             if set(g.dims_map) != dims:
#                 raise ValueError(
#                     "Mismatching dims: '%s != %s" % (dims, set(g.dims)))

#         return items

#     def __init__(self, items=[], **kwargs):
#         return super(CoordinateGroup, self).__init__(_items=items, **kwargs)

#     def __repr__(self):
#         rep = self.__class__.__name__
#         rep += '\n' + '\n'.join([repr(g) for g in self._items])
#         return rep
    
#     def __getitem__(self, key):
#         if isinstance(key, (int, slice)):
#             return self._items[key]
        
#         elif isinstance(key, tuple):
#             if len(key) != 2:
#                 raise IndexError("Too many indices for CoordinateGroup")
            
#             k, dim = key
#             # TODO list or array?
#             return [item[dim] for item in self._items[k]]
        
#         else:
#             raise IndexError(
#                 "invalid CoordinateGroup index type '%s'" % type(key))

#     def __len__(self):
#         return len(self._items)

#     def __iter__(self):
#         return self._items.__iter__()

#     def append(self, c):
#         if not isinstance(c, Coordinates):
#             raise TypeError(
#                 "Can only append Coordinates objects, not '%s'" % type(c))
        
#         self._items.append(c)
   
#     def stack(self, stack_dims, copy=True):
#         """ stack all """

#         if copy:
#             return CoordinateGroup(
#                 [c.stack(stack_dims, copy=True) for c in self._items])
#         else:
#             for c in self._items:
#                 c.stack(stack_dims)
#             return self

#     def unstack(self, copy=True):
#         """ unstack all"""
#         if copy:
#             return CoordinateGroup(
#                 [c.unstack(stack_dims, copy=True) for c in self._items])
#         else:
#             for c in self._items:
#                 c.unstack(stack_dims)
#             return self            

#     def intersect(self, other, coord_ref_sys=None, pad=1, ind=False):
#         return CoordinateGroup([c.intersect(other) for c in self._items])
    
#     @property
#     def dims(self):
#         """ unordered (set) and unstacked """
#         if len(self._items) == 0:
#             return {}
#         return set(self._items[0].dims_map)

#     def add_unique(self, other):
#         return self._add(other, unique=True)
    
#     def __add__(self, other):
#         return self._add(other)
    
#     def _add(self, other, unique=False):
#         if unique:
#             raise NotImplementedError("TODO")

#         if isinstance(other, Coordinates):
#             # TODO should this concat, fail, or do something else?
#             # items = self._items + [other]
#             raise NotImplementedError("TODO")
#         elif isinstance(other, CoordinateGroup):
#             items = self._items + g._items
#         else:
#             raise TypeError("Cannot add '%s', only BaseCoordinates" % type(c))
        
#         return CoordinateGroup(self._items + [other])

#     def __iadd__(self, other):
#         if isinstance(other, Coordinates):
#             # TODO should this append, fail, or do something else?
#             # TypeError("Cannot add individual Coordinates, use 'append'")
#             # self._items.append(other)
#             raise NotImplementedError("TODO")
#         elif isinstance(other, CoordinateGroup):
#             self._items += g._items
#         else:
#             raise TypeError("Cannot add '%s' to CoordinateGroup" % type(c))

#         return self

#     def iterchunks(self, shape, return_slice=False):
#         raise NotImplementedError("TODO")

#     @property
#     def latlon_bounds_str(self):
#         # TODO should this be a single latlon bounds or a list of bounds?
#         raise NotImplementedError("TODO")