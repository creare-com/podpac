from __future__ import division, unicode_literals, print_function, absolute_import

import os

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cache'))
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

