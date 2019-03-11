#!/bin/bash

# Set up some variables
if [ -z "$1" ]
  then
    TAG=""
    COMMIT_SHA="$(git rev-parse HEAD)"
    VERSION="develop"
  else
    TAG=$1
    VERSION=$TAG
    COMMIT_SHA=""
fi

set -e
aws s3 ls s3://podpac-s3/podpac

# TODO change this to expect tag from args, exit if there is no tag given.
DOCKER_NAME="podpac"
DOCKER_TAG="latest"
echo "${COMMIT_SHA}"
echo $DOCKER_NAME:$DOCKER_TAG

# Build docker, and extract zips
docker build -f dist/aws/Dockerfile --no-cache --tag $DOCKER_NAME:$DOCKER_TAG --build-arg COMMIT_SHA="${COMMIT_SHA}" --build-arg TAG="${TAG}" .
docker run --name "${DOCKER_NAME}" -itd $DOCKER_NAME:$DOCKER_TAG
docker cp "${DOCKER_NAME}":/tmp/vendored/podpac_dist_latest.zip .
docker cp "${DOCKER_NAME}":/tmp/vendored/podpac_deps_latest.zip .
docker stop "${DOCKER_NAME}"
docker rm "${DOCKER_NAME}"

# Upload zips to S3, according to naming convention
if [ -z $TAG ]
  then
    echo "tag is empty"
  else
    aws s3 cp podpac_deps_latest.zip s3://podpac-s3/podpac/podpac_deps_$TAG.zip
    aws s3 cp podpac_dist_latest.zip s3://podpac-s3/podpac/podpac_dist_$TAG.zip
fi
aws s3 cp podpac_deps_latest.zip s3://podpac-s3/podpac/podpac_deps_latest.zip
aws s3 cp podpac_dist_latest.zip s3://podpac-s3/podpac/podpac_dist_latest.zip

# Update lambda function to use the zips from S3 (uploaded above).
aws lambda update-function-code --dry-run --function-name podpac_lambda --s3-bucket podpac-s3 --s3-key podpac/podpac_dist_latest.zip
