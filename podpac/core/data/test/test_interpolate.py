"""
Test interpolation methods
"""


from podpac.core.data import interpolate

class TestInterpolate(object):

    """ Test interpolation methods
    """

    def test_interpolate_module(self):
        """smoke test interpolate module"""

        assert interpolate is not None
