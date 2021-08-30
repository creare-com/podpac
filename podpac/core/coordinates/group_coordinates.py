from __future__ import division, unicode_literals, print_function, absolute_import

import json
import traitlets as tl
from podpac.core.utils import hash_alg
from podpac.core.coordinates.coordinates import Coordinates
from podpac.core.utils import JSONEncoder


class GroupCoordinates(tl.HasTraits):
    """
    List of multi-dimensional Coordinates.

    GroupCoordinates contains a list of :class:`Coordinates` containing the same set of unstacked dimensions.

    The GroupCoordinates object is list-like and can be indexed, appended, looped, etc like a standard ``list``. The
    following ``Coordinates`` methods are wrapped for convenience:

     * :meth:`intersect`

    Parameters
    ----------
    udims : tuple
        Tuple of shared dimensions.
    """

    _items = tl.List(trait=tl.Instance(Coordinates))

    @tl.validate("_items")
    def _validate_items(self, d):
        items = d["value"]
        if not items:
            return items

        # unstacked dims must match, but not necessarily in order
        udims = items[0].udims
        for c in items:
            if set(c.udims) != set(udims):
                raise ValueError("Mismatching dims: %s !~ %s" % (udims, c.udims))

        return items

    def __init__(self, coords_list):
        """
        Create a Coordinates group.

        Arguments
        ---------
        coords_list : list
            list of :class:`Coordinates`
        """

        return super(GroupCoordinates, self).__init__(_items=coords_list)

    def __repr__(self):
        rep = self.__class__.__name__
        rep += "\n" + "\n".join([repr(c) for c in self._items])
        return rep

    # ------------------------------------------------------------------------------------------------------------------
    # alternative constructors
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def from_definition(cls, d):
        """
        Create a Coordinates group from a group definition.

        Arguments
        ---------
        d : list
            group definition

        Returns
        -------
        :class:`CoordinatesGroup`
            Coordinates group

        See Also
        --------
        definition, from_json
        """

        return cls([Coordinates.from_definition(elem) for elem in d])

    @classmethod
    def from_json(cls, s):
        """
        Create a Coordinates group from a group JSON definition.

        Arguments
        ---------
        s : str
            group JSON definition

        Returns
        -------
        :class:`CoordinatesGroup`
            Coordinates group

        See Also
        --------
        json
        """

        d = json.loads(s)
        return cls.from_definition(d)

    # ------------------------------------------------------------------------------------------------------------------
    # standard list-like methods
    # ------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return self._items.__iter__()

    def append(self, c):
        """Append :class:`Coordinates` to the group.

        Arguments
        ---------
        c : :class:`Coordinates`
            Coordinates to append.
        """

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
        """:tuple: Tuple of shared dimensions."""

        if len(self._items) == 0:
            return set()

        return set(self._items[0].udims)

    @property
    def definition(self):
        """
        Serializable coordinates group definition.

        The ``definition`` can be used to create new GroupCoordinates::

            g = podpac.GroupCoordinates([...])
            g2 = podpac.GroupCoordinates.from_definition(g.definition)

        See Also
        --------
        from_definition, json
        """

        return [c.definition for c in self._items]

    @property
    def json(self):
        """
        Serialized coordinates group definition.

        The ``definition`` can be used to create new GroupCoordinates::

            g = podpac.GroupCoordinates(...)
            g2 = podpac.GroupCoordinates.from_json(g.json)

        See Also
        --------
        json
        """

        return json.dumps(self.definition, separators=(",", ":"), cls=JSONEncoder)

    @property
    def hash(self):
        """
        GroupCoordinates hash.

        *Note: To be replaced with the __hash__ method.*
        """

        return hash_alg(self.json.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def intersect(self, other, outer=False, return_index=False):
        """
        Intersect each Coordinates in the group with the given coordinates.

        Parameters
        ----------
        other : :class:`Coordinates1d`, :class:`StackedCoordinates`, :class:`Coordinates`
            Coordinates to intersect with.
        outer : bool, optional
            If True, do an *outer* intersection. Default False.
        return_index : bool, optional
            If True, return slice or indices for the selection in addition to coordinates. Default False.

        Returns
        -------
        intersections : :class:`GroupCoordinates`
            Coordinates group consisting of the intersection of each :class:`Coordinates`.
        idx : list
            List of lists of indices for each :class:`Coordinates` item, only if ``return_index`` is True.
        """

        intersections = [c.intersect(other, outer=outer, return_index=True) for c in self._items]
        g = [c for c, I in intersections]

        if return_index:
            return g, [I for c, I in intersections]
        else:
            return g
