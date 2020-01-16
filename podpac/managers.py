"""
Managers Public Module
"""

# REMINDER: update api docs (doc/source/user/api.rst) to reflect changes to this file

from podpac.core.managers import aws
from podpac.core.managers.aws import Lambda
from podpac.core.managers.parallel import Parallel, ParallelOutputZarr
from podpac.core.managers.multi_process import Process
