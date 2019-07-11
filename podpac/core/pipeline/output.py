"""
Pipeline output Summary
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import warnings
from collections import OrderedDict
from io import BytesIO

try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

import traitlets as tl
import numpy as np
import traitlets as tl

from podpac.core.node import Node
from podpac.core.units import get_image

class Output(tl.HasTraits):
    """
    Base class for Pipeline Outputs.

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
    def definition(self):
        d = OrderedDict()
        for key, value in self.traits().items():
            if value.metadata.get('attr', False):
                d[key] = getattr(self, key)
        return d


class NoOutput(Output):
    """ No Output """

    def __init__(self, node, name):
        super(NoOutput, self).__init__(node=node, name=name)

    def write(self, output, coordinates):
        pass


class FileOutput(Output):
    """ Output a file to the local filesystem.

    Attributes
    ----------
    format : TYPE
        Description
    outdir : TYPE
        Description
    """

    outdir = tl.Unicode()
    format = tl.CaselessStrEnum(
        values=['pickle', 'geotif', 'png', 'nc', 'json'], default_value='pickle').tag(attr=True)
    mode = tl.Unicode(default_value="file").tag(attr=True)

    _path = tl.Unicode(allow_none=True, default_value=None)

    def __init__(self, node, name, format=None, outdir=None, mode=None):
        kwargs = {}
        if format is not None:
            kwargs['format'] = format
        if outdir is not None:
            kwargs['outdir'] = outdir
        if mode is not None:
            kwargs['mode'] = mode
        super(FileOutput, self).__init__(node=node, name=name, **kwargs)

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
        elif self.format == 'nc':
            raise NotImplementedError("format '%s' not yet implemented" % self.format)
        elif self.format == 'json':
            raise NotImplementedError("format '$s' not yet implemented" % self.format)

        self._path = path

class FTPOutput(Output):
    """Output a file and send over FTP.

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

    def __init__(self, node, name, url=None, user=None, pw=None):
        kwargs = {}
        if url is not None:
            kwargs['url'] = url
        if user is not None:
            kwargs['user'] = user
        if pw is not None:
            kwargs['pw'] = pw
        super(FTPOutput, self).__init__(node=node, name=name, **kwargs)

class S3Output(Output):
    """Output a file and send to S3

    Attributes
    ----------
    bucket : TYPE
        Description
    user : TYPE
        Description
    """

    bucket = tl.Unicode()
    user = tl.Unicode()

    def __init__(self, node, name, bucket=None, user=None):
        kwargs = {}
        if bucket is not None:
            kwargs['bucket'] = bucket
        if user is not None:
            kwargs['user'] = user
        super(S3Output, self).__init__(node=node, name=name, **kwargs)

class ImageOutput(Output):
    """Output an image in RAM

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

    format = tl.CaselessStrEnum(values=['png'], default_value='png').tag(attr=True)
    mode = tl.Unicode(default_value="image").tag(attr=True)
    vmin = tl.CFloat(allow_none=True, default_value=np.nan).tag(attr=True)
    vmax = tl.CFloat(allow_none=True, default_value=np.nan).tag(attr=True)
    image = tl.Bytes(allow_none=True, default_value=None)

    def __init__(self, node, name, format=None, mode=None, vmin=None, vmax=None):
        kwargs = {}
        if format is not None:
            kwargs['format'] = format
        if mode is not None:
            kwargs['mode'] = mode
        if vmin is not None:
            kwargs['vmin'] = vmin
        if vmax is not None:
            kwargs['vmax'] = vmax

        super(ImageOutput, self).__init__(node=node, name=name, **kwargs)

    # TODO: docstring?
    def write(self, output, coordinates):
        self.image = get_image(output, format=self.format, vmin=self.vmin, vmax=self.vmax)
