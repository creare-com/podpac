"""
Pipeline output Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import warnings

import traitlets as tl
import numpy as np

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

class NoOutput(Output):
    """Summary
    """
    
    # TODO: docstring?
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
    format = tl.CaselessStrEnum(values=['pickle', 'geotif', 'png'], default_value='pickle')

    # TODO: docstring?
    def write(self):
        self.node.write(self.name, outdir=self.outdir, format=self.format)


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

    format = tl.CaselessStrEnum(values=['png'], default='png')
    vmax = tl.CFloat(allow_none=True, default_value=np.nan)
    vmin = tl.CFloat(allow_none=True, default_value=np.nan)
    image = tl.Bytes()

    def __init__(self, **kwargs):
        warnings.warn("image output deprecated, use Node.get_image instead", DeprecationWarning)
        super(ImageOutput, self).__init__(**kwargs)

    # TODO: docstring?
    def write(self):
        try:
            self.image = self.node.get_image(format=self.format,
                                             vmin=self.vmin,
                                             vmax=self.vmax)
        except:
            pass