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

        style = Style(enumeration_colors=({0: "c", 1: "k"}))
        assert style.cmap.name == "from_list"
        assert style.cmap.colors == ("c", "k")

        with pytest.raises(TypeError, match="Style can have a colormap or enumeration_colors"):
            style = Style(colormap="cividis", enumeration_colors=({0: "c", 1: "k"}))

    def test_enumeration(self):
        # matplotlib enumeration tuples
        style = Style(
            enumeration_colors={1: "r", 3: "o"},
            enumeration_legend={1: "apples", 3: "oranges"},
            default_enumeration_color="k",
        )
        assert style.full_enumeration_colors == ("k", "r", "k", "o")
        assert style.full_enumeration_legend == ("unknown", "apples", "unknown", "oranges")

        # invalid
        with pytest.raises(ValueError, match="Style enumeration_legend keys must match enumeration_colors keys"):
            style = Style(enumeration_colors={1: "r", 3: "o"}, enumeration_legend={1: "apples"})

        with pytest.raises(ValueError, match="Style enumeration_colors keys cannot be negative"):
            style = Style(enumeration_colors={-1: "r", 3: "o"}, enumeration_legend={-1: "apples", 3: "oranges"})

        with pytest.raises(TypeError, match="Style enumeration_legend requires enumeration_colors"):
            style = Style(enumeration_legend={-1: "apples", 3: "oranges"})

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
        style = Style(enumeration_legend=({0: "apples", 1: "oranges"}), enumeration_colors=({0: "r", 1: "o"}))
        d = style.definition
        assert isinstance(d, OrderedDict)
        assert set(d.keys()) == {"enumeration_legend", "enumeration_colors"}
        assert d["enumeration_legend"] == {0: "apples", 1: "oranges"}
        assert d["enumeration_colors"] == {0: "r", 1: "o"}

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
