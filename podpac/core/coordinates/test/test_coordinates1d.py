import pytest

from podpac.core.coordinates.coordinates1d import Coordinates1d


class TestCoordinates1d(object):
    """
    See test_array_coordinates1d.py for additional Coordinates1d coverage
    """

    def test_common_api(self):
        c = Coordinates1d(name="lat")

        attrs = [
            "name",
            "is_monotonic",
            "is_descending",
            "is_uniform",
            "start",
            "stop",
            "step",
            "dims",
            "idims",
            "udims",
            "shape",
            "size",
            "dtype",
            "deltatype",
            "bounds",
            "xcoords",
            "definition",
            "full_definition",
        ]

        for attr in attrs:
            try:
                getattr(c, attr)
            except NotImplementedError:
                pass

        try:
            c.from_definition({})
        except NotImplementedError:
            pass

        try:
            c.copy()
        except NotImplementedError:
            pass

        try:
            c.select([0, 1])
        except NotImplementedError:
            pass

        try:
            c.select([0, 1], outer=True, return_index=True)
        except NotImplementedError:
            pass

        try:
            c._select([0, 1], False, False)
        except NotImplementedError:
            pass

        try:
            c.simplify()
        except NotImplementedError:
            pass

        try:
            c.flatten()
        except NotImplementedError:
            pass

        try:
            c.reshape((10, 10))
        except NotImplementedError:
            pass

        try:
            c.issubset(c)
        except NotImplementedError:
            pass
