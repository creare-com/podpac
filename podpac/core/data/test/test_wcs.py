import pytest
import traitlets as tl

import podpac
from podpac.core.data.ogc import WCS


class TestWCS(object):
    """test WCS data source
    TODO: this needs to be reworked with real examples
    """

    def test_init(self):
        """test basic init of class"""

        node = WCS(source="source", layer="layer")
        assert isinstance(node, WCS)
