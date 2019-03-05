
from __future__ import division, print_function, absolute_import

import traitlets as tl
import matplotlib
import matplotlib.cm

from podpac.core.units import Units

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
    units = Units(allow_none=True)

    is_enumerated = tl.Bool(default_value=False)
    enumeration_legend = tl.Tuple(trait=tl.Unicode)
    enumeration_colors = tl.Tuple(trait=tl.Tuple)

    clim = tl.List(default_value=[None, None])
    cmap = tl.Instance('matplotlib.colors.Colormap')
    
    @tl.default('cmap') 
    def _cmap_default(self):
        return matplotlib.cm.get_cmap('viridis')