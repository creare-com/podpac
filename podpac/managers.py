"""
Managers Public Module
"""

# REMINDER: update api docs (doc/source/user/api.rst) to reflect changes to this file

from podpac.core.manager import aws
from podpac.core.manager.aws import Lambda
from podpac.core.manager.parallel import Parallel, ParallelOutputZarr
from podpac.core.manager.multi_process import Process
