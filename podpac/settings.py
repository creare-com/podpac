"""Podpac Settings

Attributes
----------
AWS_ACCESS_KEY_ID : str
    Description
AWS_REGION_NAME : str
    Description
AWS_SECRET_ACCESS_KEY : str
    Description
CACHE_TO_S3 : bool
    Description
LOG : dict
    Dict config for PODPAC logging
    See https://docs.python.org/3/library/logging.config.html#logging.config.dictConfig
    and https://docs.python.org/3/howto/logging-cookbook.html#an-example-dictionary-based-configuration
ROOT_PATH : str
    Path to podpac working directory
S3_BUCKET_NAME : str
    Description
S3_JSON_FOLDER : str
    Description
S3_OUTPUT_FOLDER : str
    Description
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os

CACHE_TO_S3 = False
ROOT_PATH = None
# Some settings for testing AWS Lambda function handlers locally
AWS_ACCESS_KEY_ID = None
AWS_SECRET_ACCESS_KEY = None
AWS_REGION_NAME = None
S3_BUCKET_NAME = None
S3_JSON_FOLDER = None
S3_OUTPUT_FOLDER = None
LOG = None


if S3_BUCKET_NAME and CACHE_TO_S3:
    CACHE_DIR = 'cache'
else:
    if ROOT_PATH:
        CACHE_DIR = os.path.abspath(os.path.join(ROOT_PATH, 'cache'))
    else:
        CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cache'))

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

