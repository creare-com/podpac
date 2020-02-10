#!/bin/sh
#
# Build podpac lambda distribution and dependencies.
# Change $REF to specify a specific branch, tag, or commit in podpac to build from.
# 
# Usage:
# 
# $ bash build_lambda.sh
# 
# Requires:
# - Docker

# variables
REF="master"
# REF="tags/1.1.0"  # Change $REF to the branch, tag, or commit in podpac you want to use
# REF="develop"

DOCKER_NAME="podpac"
DOCKER_TAG=$REF

echo "Creating docker image from podpac version ${REF}"
echo "${DOCKER_NAME}:${DOCKER_TAG}"

# Navigate to root, build docker, and extract zips
pushd ../../
docker build -f dist/aws/Dockerfile --no-cache --tag $DOCKER_NAME:$DOCKER_TAG --build-arg REF="${REF}" .
docker run --name "${DOCKER_NAME}" -itd $DOCKER_NAME:$DOCKER_TAG
docker cp "${DOCKER_NAME}":/tmp/vendored/podpac_dist.zip ./dist/aws
docker cp "${DOCKER_NAME}":/tmp/vendored/podpac_deps.zip ./dist/aws
docker stop "${DOCKER_NAME}"
docker rm "${DOCKER_NAME}"
popd

echo "Built podpac deployment package: podpac_dist.zip"
echo "Built podpac dependencies: podpac_deps.zip"
