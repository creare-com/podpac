#!/bin/sh
#
# serve docs 

DOCS_PATH=../doc

# go to docs
cd $DOCS_PATH

# build sphinx-docs
sphinx-autobuild -aE $DOCS_PATH/source $DOCS_PATH/build
