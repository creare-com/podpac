#!/bin/sh
#
# serve docs 

# clean docs if asked
if [ "$1" == "clean" ]; then
    ./clean-docs.sh
fi

# build sphinx-docs
sphinx-autobuild -b html -i source/example-links.inc -i api/* -i source/changelog.md source build 
