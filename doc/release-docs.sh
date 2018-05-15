#!/bin/sh
#
# Build and copy docs to podpac-docs repository
# requires `podpac-docs` repository to be in the same parent directory as `podpac`

PODPAC_DOCS_PATH="../../podpac-docs"

# clean docs
./clean-docs.sh

# build sphinx-docs
sphinx-build source build

if [ -d "$PODPAC_DOCS_PATH" ]; then
    # copy recursively to new repository
    cp -rf build/* $PODPAC_DOCS_PATH
else
    echo "podpac-docs directory does not exist in the same parent directory as podpac"
fi
