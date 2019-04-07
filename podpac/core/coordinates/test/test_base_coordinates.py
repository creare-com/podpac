
from podpac.core.coordinates.base_coordinates import BaseCoordinates

class TestBaseCoordinates(object):
    def test_common_api(self):
        c = BaseCoordinates()

        attrs = ['name', 'dims', 'idims', 'udims', 'coordinates', 'coords', 'size', 'shape', 'definition']
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
            c.intersect(None)
        except NotImplementedError:
            pass

        try:
            c.intersect(None, outer=True, return_indices=True)
        except NotImplementedError:
            pass

        try:
            c[0]
        except NotImplementedError:
            pass

        try:
            repr(c)
        except NotImplementedError:
            pass

        try:
            c == c
        except NotImplementedError:
            pass