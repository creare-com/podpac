#!/bin/sh
#
# Copy docs to podpac-docs repository
# requires `podpac-docs` repository to be in the same parent directory as `podpac`

PODPAC_DOCS_PATH=../../podpac-docs
DOCS_PATH=../doc

# build sphinx-docs
sphinx-build -aE $DOCS_PATH/source $DOCS_PATH/build

# copy recursively to new repository
cp -rf $DOCS_PATH/build/* $PODPAC_DOCS_PATH

# add all new files to git
cd $PODPAC_DOCS_PATH && git add *
