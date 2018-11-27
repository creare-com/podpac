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

DEBUG = True

CACHE_TO_S3 = False
ROOT_PATH = os.path.expanduser('~') 
# Some settings for testing AWS Lambda function handlers locally
AWS_ACCESS_KEY_ID = None
AWS_SECRET_ACCESS_KEY = None
AWS_REGION_NAME = None
S3_BUCKET_NAME = None
S3_JSON_FOLDER = None
S3_OUTPUT_FOLDER = None


if S3_BUCKET_NAME and CACHE_TO_S3:
    CACHE_DIR = 'cache'
else:
    CACHE_DIR = os.path.abspath(os.path.join(ROOT_PATH, 'cache'))
    os.makedirs(CACHE_DIR, exist_ok=True)
