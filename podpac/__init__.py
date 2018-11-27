"""
Podpac Module

Public API
See https://creare-com.github.io/podpac-docs/developer/contributing.html#public-api
for more information about import conventions

Attributes
----------
version_info : TYPE
    Description
"""

# Public API
from podpac.core.coordinates import Coordinates, crange, clinspace
from podpac.core.node import Node, NodeException
import podpac.core.authentication as authentication
import podpac.core.utils as utils
from podpac import settings

# Organized submodules
# These files are simply wrappers to create a curated namespace of podpac modules
from podpac import algorithm
from podpac import data
from podpac import interpolators
from podpac import coordinates
from podpac import compositor
from podpac import pipeline
from podpac import datalib   # handles imports in datalib/__init__.py

## Developer API
from podpac import core

# version handling
from podpac import version
__version__ = version.version()
version_info = version.VERSION_INFO
