
from __future__ import division, unicode_literals, print_function, absolute_import

import json
import traitlets as tl
from podpac.core.coordinates.coordinates import Coordinates

class GroupCoordinates(tl.HasTraits):
    """
    Group of Coordinates
    """
    
    _items = tl.List(trait=tl.Instance(Coordinates))

    @tl.validate('_items')
    def _validate_items(self, d):
        items = d['value']
        if not items:
            return items

        # unstacked dims must match, but not necessarily in order
        udims = items[0].udims
        for c in items:
            if set(c.udims) != set(udims):
                raise ValueError("Mismatching dims: %s !~ %s" % (udims, c.udims))

        return items

    def __init__(self, items=[], **kwargs):
        return super(GroupCoordinates, self).__init__(_items=items, **kwargs)

    def __repr__(self):
        rep = self.__class__.__name__
        rep += '\n' + '\n'.join([repr(c) for c in self._items])
        return rep
    
    # ------------------------------------------------------------------------------------------------------------------
    # standard list-like methods
    # ------------------------------------------------------------------------------------------------------------------

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._items[key]

        elif isinstance(key, slice):
            return GroupCoordinates(self._items[key])

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return self._items.__iter__()

    def append(self, c):
        if not isinstance(c, Coordinates):
            raise TypeError("Can only append Coordinates objects, not '%s'" % type(c))

        self._items = self._items + [c]

    def __add__(self, other):
        if not isinstance(other, GroupCoordinates):
            raise TypeError("Can only add GroupCoordinates objects, not '%s'" % type(other))

        return GroupCoordinates(self._items + other._items)

    def __iadd__(self, other):
        if not isinstance(other, GroupCoordinates):
            raise TypeError("Can only add GroupCoordinates objects, not '%s'" % type(other))

        self._items = self._items + other._items
        return self
    
    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def udims(self):
        if len(self._items) == 0:
            return set()
        
        return set(self._items[0].udims)

    @property
    def definition(self):
        return [c.definition for c in self._items]

    @property
    def json(self):
        return json.dumps(self.definition)

    @property
    def hash(self):
        return hash(self.json)

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    # Currently nothing here, but we could add methods that map to _items as necessary