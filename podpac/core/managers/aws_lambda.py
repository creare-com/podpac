"""
:deprecated: See `aws` module
"""

import warnings

warnings.warn(
    "The `aws_lambda` module is deprecated and will be removed in podpac 2.0. See the `aws` module "
    "for AWS management utilites",
    DeprecationWarning,
)
from podpac.core.managers.aws import *
