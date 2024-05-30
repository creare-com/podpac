from __future__ import division, print_function, absolute_import

import json
import six
from collections import OrderedDict

import traitlets as tl
import matplotlib
import matplotlib.cm
from matplotlib.colors import ListedColormap

from podpac.core.units import ureg
from podpac.core.utils import trait_is_defined, JSONEncoder, TupleTrait

DEFAULT_ENUMERATION_LEGEND = "unknown"
DEFAULT_ENUMERATION_COLOR = (0.2, 0.2, 0.2)


class Style(tl.HasTraits):
    """Summary

    Attributes
    ----------
    name : str
        data name
    units : TYPE
        data units
    clim : list
        [low, high], color map limits
    colormap : str
        matplotlib colormap name
    cmap : matplotlib.cm.ColorMap
        matplotlib colormap property
    enumeration_colors : dict
        data colors (replaces colormap/cmap)
    enumeration_legend : dict
        data legend, should correspond with enumeration_colors
    """

    def __init__(self, node=None, *args, **kwargs):
        if node:
            self.name = node.__class__.__name__
            self.units = node.units
        super(Style, self).__init__(*args, **kwargs)

    name = tl.Unicode()
    units = tl.Unicode(allow_none=True, default_value="")
    clim = tl.List(default_value=[None, None])
    colormap = tl.Unicode(allow_none=True, default_value=None)
    enumeration_legend = tl.Dict(key_trait=tl.Int(), value_trait=tl.Unicode(), default_value=None, allow_none=True)
    enumeration_colors = tl.Dict(key_trait=tl.Int(), default_value=None, allow_none=True)
    default_enumeration_legend = tl.Unicode(default_value=DEFAULT_ENUMERATION_LEGEND)
    default_enumeration_color = tl.Any(default_value=DEFAULT_ENUMERATION_COLOR)

    @tl.validate("colormap")
    def _validate_colormap(self, d):
        if isinstance(d["value"], six.string_types):
            try: 
                matplotlib.colormaps[d["value"]]
            except AttributeError:
                # Need for matplotlib prior to 3.5
                matplotlib.cm.get_cmap(d["value"])
        if d["value"] and self.enumeration_colors:
            raise TypeError("Style can have a colormap or enumeration_colors, but not both")
        return d["value"]

    @tl.validate("enumeration_colors")
    def _validate_enumeration_colors(self, d):
        enum_colors = d["value"]
        if enum_colors and self.colormap:
            raise TypeError("Style can have a colormap or enumeration_colors, but not both")
        return enum_colors

    @tl.validate("enumeration_legend")
    def _validate_enumeration_legend(self, d):
        # validate against enumeration_colors
        enum_legend = d["value"]
        if not self.enumeration_colors:
            raise TypeError("Style enumeration_legend requires enumeration_colors")
        if set(enum_legend) != set(self.enumeration_colors):
            raise ValueError("Style enumeration_legend keys must match enumeration_colors keys")
        return enum_legend

    @property
    def full_enumeration_colors(self):
        """Convert enumeration_colors into a tuple suitable for matplotlib ListedColormap."""
        return tuple(
            [
                self.enumeration_colors.get(value, self.default_enumeration_color)
                for value in range(min(self.enumeration_colors), max(self.enumeration_colors) + 1)
            ]
        )

    @property
    def full_enumeration_legend(self):
        """Convert enumeration_legend into a tuple suitable for matplotlib."""
        return tuple(
            [
                self.enumeration_legend.get(value, self.default_enumeration_legend)
                for value in range(min(self.enumeration_legend), max(self.enumeration_legend) + 1)
            ]
        )

    @property
    def cmap(self):
        if self.colormap:
            try: 
                return matplotlib.colormaps[self.colormap]
            except AttributeError: 
                # Need for matplotlib prior to 3.5
                return matplotlib.cm.get_cmap(self.colormap)
        elif self.enumeration_colors:
            return ListedColormap(self.full_enumeration_colors)
        else:
            try: 
                return matplotlib.colormaps["viridis"]
            except AttributeError:
                # Need for matplotlib prior to 3.5
                return matplotlib.cm.get_cmap("viridis")

    @property
    def json(self):
        """JSON-serialized style definition

        The `json` can be used to create new styles.

        See Also
        ----------
        from_json
        """

        return json.dumps(self.definition, separators=(",", ":"), cls=JSONEncoder)

    @classmethod
    def get_style_ui(self):
        """
        Attempting to expose style units to get_ui_spec(). This will grab defaults in general.
        BUT this will not set defaults for each particular node.
        """
        d = OrderedDict()
        if self.name:
            d["name"] = self.name
        if self.units:
            d["units"] = self.units
        if self.colormap:
            d["colormap"] = self.colormap
        if self.enumeration_legend:
            d["enumeration_legend"] = self.enumeration_legend
        if self.enumeration_colors:
            d["enumeration_colors"] = self.enumeration_colors
        if self.default_enumeration_legend != DEFAULT_ENUMERATION_LEGEND:
            d["default_enumeration_legend"] = self.default_enumeration_legend
        if self.default_enumeration_color != DEFAULT_ENUMERATION_COLOR:
            d["default_enumeration_color"] = self.default_enumeration_color
        if self.clim != [None, None]:
            d["clim"] = self.clim
        return d

    @property
    def definition(self):
        d = OrderedDict()
        if self.name:
            d["name"] = self.name
        if self.units:
            d["units"] = self.units
        if self.colormap:
            d["colormap"] = self.colormap
        if self.enumeration_legend:
            d["enumeration_legend"] = self.enumeration_legend
        if self.enumeration_colors:
            d["enumeration_colors"] = self.enumeration_colors
        if self.default_enumeration_legend != DEFAULT_ENUMERATION_LEGEND:
            d["default_enumeration_legend"] = self.default_enumeration_legend
        if self.default_enumeration_color != DEFAULT_ENUMERATION_COLOR:
            d["default_enumeration_color"] = self.default_enumeration_color
        if self.clim != [None, None]:
            d["clim"] = self.clim
        return d

    @classmethod
    def from_definition(cls, d):
        # parse enumeration keys to int
        if "enumeration_colors" in d:
            d["enumeration_colors"] = {int(key): value for key, value in d["enumeration_colors"].items()}
        if "enumeration_legend" in d:
            d["enumeration_legend"] = {int(key): value for key, value in d["enumeration_legend"].items()}
        return cls(**d)

    @classmethod
    def from_json(cls, s):
        """Create podpac Style from a style JSON definition.

        Parameters
        -----------
        s : str
            JSON definition

        Returns
        --------
        Style
            podpac Style object
        """

        d = json.loads(s)
        return cls.from_definition(d)

    def __eq__(self, other):
        if not isinstance(other, Style):
            return False

        return self.json == other.json
