from __future__ import division, unicode_literals, print_function, absolute_import

from collections import OrderedDict
from matplotlib import pyplot

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

        style = Style(cmap=pyplot.get_cmap("cividis"))
        assert style.cmap.name == "cividis"

        style = Style(is_enumerated=True, enumeration_colors=("c", "k"))
        assert style.cmap.name == "from_list"
        assert style.cmap.colors == ("c", "k")

    def test_serialization(self):
        # default
        style = Style()
        d = style.definition
        assert isinstance(d, OrderedDict)
        assert set(d.keys()) == {"cmap"}
        assert d["cmap"] == "viridis"

        s = Style.from_json(style.json)
        assert isinstance(s, Style)

        # with traits
        style = Style(name="test", units="meters", cmap=pyplot.get_cmap("cividis"), clim=(-1, 1))
        d = style.definition
        assert isinstance(d, OrderedDict)
        assert set(d.keys()) == {"name", "units", "cmap", "clim"}
        assert d["name"] == "test"
        assert d["units"] == "meters"
        assert d["cmap"] == "cividis"
        assert d["clim"] == [-1, 1]

        s = Style.from_json(style.json)
        assert s.name == style.name
        assert s.units == style.units
        assert s.cmap == style.cmap
        assert s.clim == style.clim

        # enumeration traits
        style = Style(is_enumerated=True, enumeration_legend=("apples", "oranges"), enumeration_colors=["r", "o"])
        d = style.definition
        assert isinstance(d, OrderedDict)
        assert set(d.keys()) == {"is_enumerated", "enumeration_legend", "enumeration_colors"}
        assert d["is_enumerated"] == True
        assert d["enumeration_legend"] == ("apples", "oranges")
        assert d["enumeration_colors"] == ("r", "o")

        s = Style.from_json(style.json)
        assert s.is_enumerated == style.is_enumerated
        assert s.enumeration_legend == style.enumeration_legend
        assert s.enumeration_colors == style.enumeration_colors
        assert s.cmap.colors == style.cmap.colors

    def test_eq(self):
        style1 = Style(units="meters")
        style2 = Style(units="meters")
        style3 = Style(units="feet")

        assert style1 is not style2
        assert style1 is not style3
        assert style2 is not style3

        assert style1 == style1
        assert style2 == style2
        assert style3 == style3

        assert style1 == style2
        assert style1 != style3
