from __future__ import division, print_function, absolute_import

import json
import six
from collections import OrderedDict

import traitlets as tl
import matplotlib
import matplotlib.cm
from matplotlib.colors import ListedColormap

from podpac.core.units import ureg
from podpac.core.utils import trait_is_defined, JSONEncoder


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
    enumeration_colors : tuple
        data colors (replaces colormap/cmap)
    enumeration_legend : tuple
        data legend, should correspond with enumeration_colors
    """

    def __init__(self, node=None, *args, **kwargs):
        if node:
            self.name = node.__class__.__name__
            self.units = node.units
        super(Style, self).__init__(*args, **kwargs)

    name = tl.Unicode()
    units = tl.Unicode(allow_none=True)
    clim = tl.List(default_value=[None, None])
    colormap = tl.Unicode(allow_none=True, default_value=None)
    enumeration_legend = tl.Tuple(trait=tl.Unicode)
    enumeration_colors = tl.Tuple(trait=tl.Tuple)

    @tl.validate("colormap")
    def _validate_colormap(self, d):
        if isinstance(d["value"], six.string_types):
            matplotlib.cm.get_cmap(d["value"])
        if d["value"] and self.enumeration_colors:
            raise TypeError("Style can have a colormap or enumeration_colors, but not both")
        return d["value"]

    @tl.validate("enumeration_colors")
    def _validate_enumeration_colors(self, d):
        if d["value"] and self.colormap:
            raise TypeError("Style can have a colormap or enumeration_colors, but not both")
        return d["value"]

    @property
    def cmap(self):
        if self.colormap:
            return matplotlib.cm.get_cmap(self.colormap)
        elif self.enumeration_colors:
            return ListedColormap(self.enumeration_colors)
        else:
            return matplotlib.cm.get_cmap("viridis")

    @property
    def json(self):
        """ JSON-serialized style definition
        
        The `json` can be used to create new styles. 
        
        See Also
        ----------
        from_json
        """

        return json.dumps(self.definition, separators=(",", ":"), cls=JSONEncoder)

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
        if self.clim != [None, None]:
            d["clim"] = self.clim
        return d

    @classmethod
    def from_definition(cls, d):
        return cls(**d)

    @classmethod
    def from_json(cls, s):
        """ Create podpac Style from a style JSON definition.
        
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
