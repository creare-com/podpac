"""
Datalib Public API

This module import the podpacdatalib package
and exposed its contents to podpac.datalib
"""

try:
    from podpacdatalib import *
except ModuleNotFoundError:
    import logging

    _logger = logging.getLogger(__name__)
    _logger.warning(
        "The podpacdatalib module is not installed but user tried to import podpac.datalib which depends on it."
    )
