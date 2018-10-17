import subprocess
import os
from collections import OrderedDict

VERSION_INFO = OrderedDict([
    ('MAJOR', 0,),
    ('MINOR', 2,),
    ('HOTFIX', 0,),
    ])

VERSION = (VERSION_INFO['MAJOR'], 
           VERSION_INFO['MINOR'],
           VERSION_INFO['HOTFIX'])

def version():
    VERSION_FULL  = '.'.join([str(v) for v in VERSION])
    CWD = os.path.dirname(__file__)
    got_git = os.path.exists(os.path.join(
                                          os.path.dirname(__file__),
                                          '..',
                                          '.git'))
    if not got_git:
        return VERSION_FULL
    try:
        current_branch = ''
        git = "git"
        try:
            subprocess.check_output([git,"--version"])
        except Exception:
            git = '/usr/bin/git'
            pass
        branches = subprocess.check_output([git, "branch"], cwd=CWD).decode('ascii')
        for branch in branches.split('\n'):
            if branch.startswith('*'):
                current_branch = branch.split(' ')[-1]
    
        GIT_HASH = subprocess.check_output([git, "describe", "--always", "--abbrev=0",
                                            "--match", '"NOT A TAG"', '--dirty=*'], cwd=CWD).strip().decode('ascii')[:-1]
        
        if(current_branch is not 'master'):
            VERSION_FULL += '+' + GIT_HASH
               
    except Exception as e:
        print("Could not determine PODPAC version from git repo.\n"+str(e))
    
    return VERSION_FULL
