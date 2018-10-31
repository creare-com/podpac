"""
Pipeline output Summary
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings
from collections import OrderedDict
from io import BytesIO


import os
import traitlets as tl
import numpy as np
import traitlets as tl

try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

from podpac.core.node import Node
from podpac.core.units import get_image


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

    def write(self, output, coordinates):
        """Write the node output

        Arguments
        ---------
        output : UnitsDataArray
            Node evaluation output to write
        coordinates : Coordinates
            Evaluated coordinates.

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
    def write(self, output, coordinates):
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
        values=['pickle', 'geotif', 'png'], default_value='pickle').tag(attr=True)
    mode = tl.Unicode(default_value="file").tag(attr=True)

    _path = tl.Unicode(allow_none=True, default_value=None)

    @property
    def path(self):
        return self._path

    # TODO: docstring?
    def write(self, output, coordinates):
        filename = '%s_%s_%s' % (self.name, self.node.hash, coordinates.hash)
        path = os.path.join(self.outdir, filename)

        if self.format == 'pickle':
            path = '%s.pkl' % path
            with open(path, 'wb') as f:
                cPickle.dump(output, f)
        elif self.format == 'png':
            raise NotImplementedError("format '%s' not yet implemented" % self.format)
        elif self.format == 'geotif':
            raise NotImplementedError("format '%s' not yet implemented" % self.format)

        self._path = path

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
    def write(self, output, coordinates):
        try:
            self.image = get_image(output, format=self.format, vmin=self.vmin, vmax=self.vmax)
        except Exception as e:
            warnings.warn("Error getting image from output: %s" % e)
