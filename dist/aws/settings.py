"""Summary

Attributes
----------
CACHE_TO_S3 : bool
    Description
ROOT_PATH : TYPE
    Description
S3_BUCKET_NAME : TYPE
    Description
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os

CACHE_TO_S3 = False
#TODO for now we'll just cache in /tmp/, but this will change with the new caching spec.
ROOT_PATH = "/tmp/"
# Some settings for testing AWS Lambda function handlers locally
AWS_ACCESS_KEY_ID = None
AWS_SECRET_ACCESS_KEY = None
AWS_REGION_NAME = None
S3_BUCKET_NAME = None
S3_JSON_FOLDER = 'json/'
S3_OUTPUT_FOLDER = 'output/'



if S3_BUCKET_NAME and CACHE_TO_S3:
    CACHE_DIR = 'cache'
else:
    if ROOT_PATH:
        CACHE_DIR = os.path.abspath(os.path.join(ROOT_PATH, 'cache'))
    else:
        CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cache'))

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
