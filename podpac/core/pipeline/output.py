"""
Pipeline output Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import warnings

import traitlets as tl
import numpy as np

from collections import OrderedDict
from podpac.core.node import Node

class Output(tl.HasTraits):
    """
    Base Pipeline Output class.

    Attributes
    ----------
    node : Node
        output node
    name : string
        output name
    """

    node = tl.Instance(Node)
    name = tl.Unicode()

    def write(self):
        """Summary
        
        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError

    @property
    def pipeline_definition(self):
        raise NotImplementedError

class NoOutput(Output):
    def write(self):
        pass

class FileOutput(Output):
    """Summary

    Attributes
    ----------
    format : TYPE
        Description
    outdir : TYPE
        Description
    """
    
    outdir = tl.Unicode()
    format = tl.CaselessStrEnum(values=['pickle', 'geotif', 'png'], default='pickle')

    _path = tl.Unicode(allow_none=True, default_value=None)
    @property
    def path(self):
        return self._path

    # TODO: docstring?
    def write(self):
        self._path = self.node.write(self.name, outdir=self.outdir, format=self.format)


class FTPOutput(Output):
    """Summary

    Attributes
    ----------
    url : TYPE
        Description
    user : TYPE
        Description
    """

    url = tl.Unicode()
    user = tl.Unicode()
    pw = tl.Unicode()

class S3Output(Output):
    """Summary

    Attributes
    ----------
    bucket : TYPE
        Description
    user : TYPE
        Description
    """

    bucket = tl.Unicode()
    user = tl.Unicode()

class ImageOutput(Output):
    """Summary
    Attributes
    ----------
    format : TYPE
        Description
    image : TYPE
        Description
    vmax : TYPE
        Description
    vmin : TYPE
        Description
    """

    format = tl.CaselessStrEnum(values=['png'], default_value='png')
    vmax = tl.CFloat(allow_none=True, default_value=np.nan)
    vmin = tl.CFloat(allow_none=True, default_value=np.nan)
    image = tl.Bytes(allow_none=True, default_value=None)

    # TODO: docstring?
    def write(self):
        try:
            self.image = self.node.get_image(format=self.format, vmin=self.vmin, vmax=self.vmax)
        except:
            pass

    @property
    def pipeline_definition(self):
        d = OrderedDict()
        d['mode'] = "image"
        d['format'] = self.format
        d['vmin'] = self.vmin
        d['vmax'] = self.vmax
        d['nodes'] = [self.name]
        return d
