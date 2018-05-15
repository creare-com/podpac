#!/bin/sh
#
# serve docs 

# clean docs if asked
if [ "$1" == "clean" ]; then
    ./clean-docs.sh
fi

# build sphinx-docs
sphinx-autobuild source build 
