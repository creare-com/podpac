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

S3_BUCKET_NAME = None
CACHE_TO_S3 = False
ROOT_PATH = None


if S3_BUCKET_NAME and CACHE_TO_S3:
    CACHE_DIR = 'cache'
else:
    if ROOT_PATH:
        CACHE_DIR = os.path.abspath(ROOT_PATH, 'cache')
    else:
        CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cache'))

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
