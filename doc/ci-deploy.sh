#!/bin/bash
# 
# this script runs from the `doc` directory in the Github Document Deployment Workflow

set -e # Exit with nonzero exit code if anything fails

# Cloning repositories is handled by workflow actions in previous step
PODPAC_DOCS="https://github.com/creare-com/podpac-docs"
PODPAC_DOCS_PATH="../../podpac-docs"
COMMIT_AUTHOR=`git log --format="%cn" -n 1`
COMMIT_AUTHOR_EMAIL=`git log --format="%ce" -n 1`
PODPAC_EXAMPLES_PATH="../../podpac-examples"

# Run our compile script
./release-docs.sh

# Now let's go have some fun with the cloned repo
cd $PODPAC_DOCS_PATH
git config user.name "$COMMIT_AUTHOR"
git config user.email "$COMMIT_AUTHOR_EMAIL"

# If there are no changes to the compiled out (e.g. this is a README update) then just bail.
if git diff --quiet; then
    echo "No changes to the output on this push; exiting."
    exit 0
fi

# Commit the "changes", i.e. the new version.
# The delta will show diffs between new and old versions.
git add -A .
git commit -m "Deploy podpac docs to GitHub Pages: $COMMIT_HASH"

# Now that we're all set up, we can push.
git push $PODPAC_DOCS master
