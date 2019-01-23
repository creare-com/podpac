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
from collections import OrderedDict

##############
## UPDATE VERSION HERE
##############
MAJOR = 0
MINOR = 2
HOTFIX = 3
##############


VERSION_INFO = OrderedDict([
    ('MAJOR', MAJOR,),
    ('MINOR', MINOR,),
    ('HOTFIX', HOTFIX,),
])

VERSION = (VERSION_INFO['MAJOR'],
           VERSION_INFO['MINOR'],
           VERSION_INFO['HOTFIX'])

def semver():
    """Return semantic version of current PODPAC installation
    
    Returns
    -------
    str
        Semantic version of the current PODPAC installation
    """
    return '.'.join([str(v) for v in VERSION])


def version():
    """Retrieve PODPAC semantic version as string
    
    Returns
    -------
    str
        Semantic version if on the master branch (or outside git repository).
        Includes +git hash if in a git repository off the master branch (i.e. 0.0.0+hash)
    """

    version_full = semver()
    CWD = os.path.dirname(__file__)
    got_git = os.path.exists(os.path.join(os.path.dirname(__file__),
                                          '..',
                                          '.git'))
    if not got_git:
        return version_full
    try:
        current_branch = ''
        git = "git"
        try:
            subprocess.check_output([git, "--version"])
        except Exception:
            git = '/usr/bin/git'
            try:
                subprocess.check_output([git, "--version"])
            except Exception as e:
                return version_full

        branches = subprocess.check_output([git, "branch"], cwd=CWD).decode('ascii')
        for branch in branches.split('\n'):
            if branch.startswith('*'):
                current_branch = branch.split(' ')[-1]
    
        git_hash = subprocess.check_output([git, "describe", "--always", "--abbrev=0",
                                            "--match", '"NOT A TAG"', '--dirty=*'], cwd=CWD) \
                                            .strip().decode('ascii')[:-1]
        git_hash_short = git_hash[0:7]
        
        if current_branch != 'master':
            version_full += '+' + git_hash_short
               
    except Exception as e:
        print("Could not determine PODPAC version from git repo.\n" + str(e))
    
    return version_full
