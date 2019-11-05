#!/bin/sh
#
# Build podpac lambda distribution and dependencies
# 
# Currently, this builds the function using the local 
# podpac repository, including any outstanding changes.
# 
# Usage:
# 
# $ bash build_lambda.sh [s3-bucket] [function-name] 
# 
# Requires:
# - Docker
# - `settings.json` to be copied to the root directory of the podpac repository
#   This will not be required in the future
#   
# Example usage:
# 
# $ bash build_lambda.sh


# variables
COMMIT_SHA="$(git rev-parse HEAD)"
TAG="$(git describe --always)"
DOCKER_NAME="podpac"
DOCKER_TAG=$TAG

echo "Creating docker image from podpac version ${TAG}"
echo "${DOCKER_NAME}:${DOCKER_TAG}"

# Navigate to root, build docker, and extract zips
pushd ../../
docker build -f dist/aws/Dockerfile --no-cache --tag $DOCKER_NAME:$DOCKER_TAG --build-arg COMMIT_SHA="${COMMIT_SHA}" --build-arg TAG="${TAG}" .
docker run --name "${DOCKER_NAME}" -itd $DOCKER_NAME:$DOCKER_TAG
docker cp "${DOCKER_NAME}":/tmp/vendored/podpac_dist.zip ./dist/aws
docker cp "${DOCKER_NAME}":/tmp/vendored/podpac_deps.zip ./dist/aws
docker stop "${DOCKER_NAME}"
docker rm "${DOCKER_NAME}"
popd
