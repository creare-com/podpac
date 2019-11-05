#!/bin/sh
#
# Upload podpac lambda distribution and dependencies
# 
# Currently, this uploads the zip archives and updates
# the specific lambda function
# 
# Usage:
# 
# $ bash upload_lambda.sh [s3-bucket]
# 
# Requires:
# - AWS CLI: https://docs.aws.amazon.com/cli/
# - AWS credentials must be configured using the `aws` cli.
#   See https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration
#
# Example usage:
# 
# $ bash upload_lambda.sh

BUCKET=podpac-dist
TAG="$(git describe --always)"

if [ ! -z "$1" ]
  then
    BUCKET=$1
fi

echo "Uploading podpac distribution to bucket: ${BUCKET}"

# Upload zips to S3
aws s3 cp podpac_deps.zip s3://$BUCKET/$TAG/podpac_deps.zip
aws s3 cp podpac_dist.zip s3://$BUCKET/$TAG/podpac_dist.zip
# rm podpac_deps.zip podpac_dist.zip

echo "Navigate to your bucket $BUCKET, select the zip archives you just uploaded and make them public"
