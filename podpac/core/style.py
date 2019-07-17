from __future__ import division, print_function, absolute_import

import traitlets as tl
import matplotlib
import matplotlib.cm
import json
from collections import OrderedDict

from podpac.core.units import ureg
from podpac.core.utils import trait_is_defined, JSONEncoder


class Style(tl.HasTraits):
    """Summary

    Attributes
    ----------
    clim : TYPE
        Description
    cmap : TYPE
        Description
    enumeration_colors : TYPE
        Description
    enumeration_legend : TYPE
        Description
    is_enumerated : TYPE
        Description
    name : TYPE
        Description
    units : TYPE
        Description
    """

    def __init__(self, node=None, *args, **kwargs):
        if node:
            self.name = node.__class__.__name__
            self.units = node.units
        super(Style, self).__init__(*args, **kwargs)

    name = tl.Unicode()
    units = tl.Unicode(allow_none=True)

    is_enumerated = tl.Bool(default_value=False)
    enumeration_legend = tl.Tuple(trait=tl.Unicode)
    enumeration_colors = tl.Tuple(trait=tl.Tuple)

    clim = tl.List(default_value=[None, None])
    cmap = tl.Instance("matplotlib.colors.Colormap")

    @tl.default("cmap")
    def _cmap_default(self):
        return matplotlib.cm.get_cmap("viridis")

    @property
    def json(self):
        """ JSON-serialized style definition
        
        The `json` can be used to create new styles. 
        
        See Also
        ----------
        from_json
        """

        return json.dumps(self.definition, cls=JSONEncoder)

    @property
    def definition(self):
        d = OrderedDict()
        for t in self.trait_names():
            if not trait_is_defined(self, t):
                continue
            d[t] = getattr(self, t)
        d["cmap"] = self.cmap.name
        return d

    @classmethod
    def from_definition(cls, d):
        if "cmap" in d:
            d["cmap"] = matplotlib.cm.get_cmap(d["cmap"])
        return cls(**d)

    @classmethod
    def from_json(cls, s):
        """ Create podpac Style from a style JSON definition.
        
        Parameters
        -----------
        s : str
            JSON definition
            kkkk
        Returns
        --------
        Style
            podpac Style object
        """

        d = json.loads(s)
        return cls.from_definition(d)
