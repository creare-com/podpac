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
# $ bash upload_lambda.sh podpac-s3
# 

BUCKET=$1
TAG="$(git describe --always)"

if [ -z "$BUCKET" ]
  then
    echo "S3 bucket name required as first cli argument"
    exit 1
  else
    echo "Bucket: ${BUCKET}"
fi


# Upload zips to S3 with commit hash at the end
aws s3 cp podpac_deps.zip s3://$BUCKET/podpac/podpac_deps_$TAG.zip
aws s3 cp podpac_dist.zip s3://$BUCKET/podpac/podpac_dist_$TAG.zip
# rm podpac_deps.zip podpac_dist.zip

echo "Navigate to your S3 bucket, select the dependencies zip archives you just uploaded and make it public"
echo "In the future we will automate the access between the lambda function and s3 bucket"
