#!/bin/sh
#
# Build and copy docs to podpac-docs repository
# requires `podpac-docs` repository to be in the same parent directory as `podpac`

PODPAC_DOCS_PATH=../../podpac-docs

# clean docs
./clean-docs.sh

# build sphinx-docs
sphinx-build -aE source build

# copy recursively to new repository
cp -rf build/* $PODPAC_DOCS_PATH

# add all new files to git
cd $PODPAC_DOCS_PATH && git add *
