"""
Pipeline output Summary
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings
from collections import OrderedDict
from io import BytesIO

import numpy as np
import traitlets as tl

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
        d = OrderedDict()
        for key, value in self.traits().items():
            if value.metadata.get('attr', False):
                d[key] = getattr(self, key)
        d['nodes'] = [self.name]
        return d


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
    format = tl.CaselessStrEnum(
        values=['pickle', 'geotif', 'png'], default_value='pickle')

    _path = tl.Unicode(allow_none=True, default_value=None)

    @property
    def path(self):
        return self._path

    # TODO: docstring?
    def write(self):
        self._path = self.node.write(
            self.name, outdir=self.outdir, format=self.format)


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

    format = tl.CaselessStrEnum(
        values=['png'], default_value='png').tag(attr=True)
    mode = tl.Unicode(default_value="image").tag(attr=True)
    vmax = tl.CFloat(allow_none=True, default_value=np.nan).tag(attr=True)
    vmin = tl.CFloat(allow_none=True, default_value=np.nan).tag(attr=True)
    image = tl.Instance(BytesIO, allow_none=True, default_value=None)

    # TODO: docstring?
    def write(self):
        try:
            self.image = self.node.get_image(
                format=self.format, vmin=self.vmin, vmax=self.vmax)
        except:
            pass
