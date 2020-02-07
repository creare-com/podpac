#!/bin/sh
#
# Upload podpac lambda distribution and dependencies
# Change $BUCKET or $PATH to control the S3 Bucket and Bucket path
# where zip archives are uploaded.
# 
# Usage:
# 
# $ bash upload_lambda.sh
# 
# Requires:
# - AWS CLI: https://docs.aws.amazon.com/cli/
# - AWS credentials must be configured using the `aws` cli.
#   See https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration


BUCKET="podpac-dist"
# DIR="dev"
DIR="1.3.0"  # for releases, upload to release path by semantic version

AWSPATH="s3://$BUCKET/$DIR"
echo "Uploading podpac distribution to S3 path: ${AWSPATH}"

# Upload zips to S3
aws s3 cp podpac_deps.zip $AWSPATH/podpac_deps.zip
aws s3 cp podpac_dist.zip $AWSPATH/podpac_dist.zip

echo "Navigate to your bucket $BUCKET, select the zip archives you just uploaded and make them public"
