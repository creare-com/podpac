from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
from matplotlib import pyplot
import pytest

from podpac.core.style import Style


class TestStyle(object):
    def test_init(self):
        s = Style()

    def test_init_from_node(self):
        from podpac.core.node import Node

        node = Node()
        s = Style(node)

    def test_cmap(self):
        style = Style()
        assert style.cmap.name == "viridis"

        style = Style(colormap="cividis")
        assert style.cmap.name == "cividis"

        style = Style(enumeration_colors=("c", "k"))
        assert style.cmap.name == "from_list"
        assert style.cmap.colors == ("c", "k")

        with pytest.raises(TypeError, match="Style can have a colormap or enumeration_colors"):
            style = Style(colormap="cividis", enumeration_colors=("c", "k"))

    def test_serialization(self):
        # default
        style = Style()
        d = style.definition
        assert isinstance(d, OrderedDict)
        assert len(d.keys()) == 0

        s = Style.from_json(style.json)
        assert isinstance(s, Style)

        # with traits
        style = Style(name="test", units="meters", colormap="cividis", clim=(-1, 1))
        d = style.definition
        assert isinstance(d, OrderedDict)
        assert set(d.keys()) == {"name", "units", "colormap", "clim"}
        assert d["name"] == "test"
        assert d["units"] == "meters"
        assert d["colormap"] == "cividis"
        assert d["clim"] == [-1, 1]

        s = Style.from_json(style.json)
        assert s.name == style.name
        assert s.units == style.units
        assert s.colormap == style.colormap
        assert s.clim == style.clim

        # enumeration traits
        style = Style(enumeration_legend=("apples", "oranges"), enumeration_colors=["r", "o"])
        d = style.definition
        assert isinstance(d, OrderedDict)
        assert set(d.keys()) == {"enumeration_legend", "enumeration_colors"}
        assert d["enumeration_legend"] == ("apples", "oranges")
        assert d["enumeration_colors"] == ("r", "o")

        s = Style.from_json(style.json)
        assert s.enumeration_legend == style.enumeration_legend
        assert s.enumeration_colors == style.enumeration_colors
        assert s.cmap.colors == style.cmap.colors

    def test_eq(self):
        style1 = Style(name="test")
        style2 = Style(name="test")
        style3 = Style(name="other")

        assert style1 is not style2
        assert style1 is not style3
        assert style2 is not style3

        assert style1 == style1
        assert style2 == style2
        assert style3 == style3

        assert style1 == style2
        assert style1 != style3
