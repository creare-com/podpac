
from podpac.core.coordinates.base_coordinates import BaseCoordinates

class TestBaseCoordinates(object):
    def test_common_api(self):
        c = BaseCoordinates()

        attrs = ['name', 'dims', 'idims', 'udims', 'coordinates', 'coords', 'size', 'shape', 'definition', 'full_definition']
        for attr in attrs:
            try:
                getattr(c, attr)
            except NotImplementedError:
                pass

        for method_name in ['_set_name', '_set_ctype']:
            try:
                method = getattr(c, method_name)
                method(None)
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
            c.select([0, 1], outer=True, return_indices=True)
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