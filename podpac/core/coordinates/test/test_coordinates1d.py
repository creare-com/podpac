
# see test_array_coordinates1d.py
from podpac.core.coordinates.coordinates1d import Coordinates1d

class TestCoordinates1d(object):
    def test_common_api(self):
        c = Coordinates1d(name='lat')

        attrs = ['name', 'units', 'crs', 'ctype', 'segment_lengths',
                 'is_monotonic', 'is_descending', 'is_uniform',
                 'dims', 'idims', 'udims', 'shape', 'size', 'dtype', 'deltatype',
                 'bounds', 'area_bounds', 'coords',
                 'properties', 'definition', 'full_definition']

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
            c.copy(name='lon', ctype='point')
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
            c._select([0, 1], False, False)
        except NotImplementedError:
            pass

        try:
            c.intersect(c)
        except NotImplementedError:
            pass

        try:
            c.intersect(c, outer=True, return_indices=True)
        except NotImplementedError:
            pass

        assert c != None