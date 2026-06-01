"""PODPAC Version

Attributes
----------
VERSION : OrderedDict
    dict of podpac version {('MAJOR': int), ('MINOR': int), ('HOTFIX': int)}
VERSION_INFO : tuple of int
    (MAJOR, MINOR, HOTFIX)
"""

import subprocess
import os
import logging
from collections import OrderedDict

##############
## UPDATE VERSION HERE
##############
MAJOR = 4
MINOR = 0
HOTFIX = 2
##############


VERSION_INFO = OrderedDict([("MAJOR", MAJOR), ("MINOR", MINOR), ("HOTFIX", HOTFIX)])

VERSION = (VERSION_INFO["MAJOR"], VERSION_INFO["MINOR"], VERSION_INFO["HOTFIX"])

_log = logging.getLogger(__name__)


def semver():
    """Return semantic version of current PODPAC installation

    Returns
    -------
    str
        Semantic version of the current PODPAC installation
    """
    return ".".join([str(v) for v in VERSION])


def version():
    """Retrieve PODPAC semantic version as string

    Returns
    -------
    str
        Semantic version if outside git repository
        Returns `git describe --always` if inside the git repository
    """

    version_full = semver()
    CWD = os.path.dirname(__file__)
    got_git = os.path.exists(os.path.join(os.path.dirname(__file__), "..", ".git"))
    if not got_git:
        return version_full
    try:
        # determine git binary
        git = "git"
        try:
            subprocess.check_output([git, "--version"])
        except (OSError, subprocess.CalledProcessError):
            git = "/usr/bin/git"
            try:
                subprocess.check_output([git, "--version"])
            except (OSError, subprocess.CalledProcessError):
                return version_full

        version_full = subprocess.check_output([git, "describe", "--always", "--tags"], cwd=CWD).strip().decode("ascii")
        version_full = version_full.replace("-", "+", 1).replace("-", ".")  # Make this consistent with PEP440

    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as e:
        _log.warning("Could not determine PODPAC version from git repo.\n" + str(e))

    return version_full
