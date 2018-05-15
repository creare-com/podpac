#!/bin/bash
# Adapted from https://gist.github.com/domenic/ec8b0fc8ab45f39403dd
# 
# this script runs from the `doc` directory in the travis deploy job
# Note this decrypts deploy key `ci-doc-deploy` for creare-com/podpac-docs repository 
# generated using the travis ruby client (`gem install travis`)
# 
# ```
# $ travis encrypt-file ci-doc-deploy
# ```

set -e # Exit with nonzero exit code if anything fails

PODPAC_DOCS="https://github.com/creare-com/podpac-docs"
SSH_PODPAC_DOCS="git@github.com:creare-com/podpac-docs.git"
PODPAC_DOCS_PATH="../../podpac-docs"
COMMIT_AUTHOR=`git log --format="%cn" -n 1`
COMMIT_AUTHOR_EMAIL=`git log --format="%ce" -n 1`

# Only deploy to site on develop or master. Ignore pull requests
if [ "$TRAVIS_BRANCH" == "master" ] || [ "$TRAVIS_BRANCH" == "develop" ]; then
    echo "Deploying docs to podpac-docs"
else
    echo "Skipping docs build on branch $TRAVIS_BRANCH"
    exit 0
fi

# Get the deploy key by using Travis's stored variables to decrypt ci-doc-deploy
openssl aes-256-cbc -K $encrypted_f01ce353ad15_key -iv $encrypted_f01ce353ad15_iv -in ci-doc-deploy.enc -out ci-doc-deploy -d
chmod 0600 ci-doc-deploy
eval `ssh-agent -s`
ssh-add ci-doc-deploy

# clone podpac-docs repo into directory next to podpac
git clone $PODPAC_DOCS $PODPAC_DOCS_PATH

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
git commit -m "Deploy podpac docs to GitHub Pages: $TRAVIS_COMMIT"



# Now that we're all set up, we can push.
git push $SSH_PODPAC_DOCS master
