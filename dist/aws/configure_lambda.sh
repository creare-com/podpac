#!/bin/sh
#
# Configure AWS for podpac lambda function
# 
# Usage:
# 
# $ bash configure_lambda.sh [s3-bucket] [function-name]
# 
# Requires:
# - AWS CLI: https://docs.aws.amazon.com/cli/
# - AWS credentials must be configured using the `aws` cli.
#   See https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration
# - Dist and Dependencies uploaded using `upload_lambda.sh`
# - Function must be created from the AWS Dashboard
# - API Gateway must be created from the AWS Dashboard
#   - Note down the `rest-api-id` and `resource-id` in parentheses in the top bar of the API gatway dashboard
#   - You must Select the top level resource '/' and select "Create Method"
# Example usage:
# 
# $ bash configure_lambda.sh podpac-s3 podpac_lambda h827as06ji 1ya7h6
# 

# TODO: remove this in the future when function generation/update is automated elsewhere

BUCKET=$1
FUNCTION=$2
API_ID=$3
API_RESOURCE_ID=$4
TAG="$(git describe --always)"

if [ -z "$BUCKET" ]
  then
    echo "S3 bucket name required as first cli argument"
    exit 1
  else
    echo "Bucket: ${BUCKET}"
fi

if [ -z "$FUNCTION" ]
  then
    echo "Function name required as second cli argument"
    exit 1
  else
    echo "Function: ${FUNCTION}"
fi

if [ -z "$API_ID" ]
  then
    echo "Rest API ID required as third cli argument"
    exit 1
  else
    echo "REST API ID: ${API_ID}"
fi

if [ -z "$API_RESOURCE_ID" ]
  then
    echo "API Resource ID required as fourth cli argument"
    exit 1
  else
    echo "API Resource ID: ${API_RESOURCE_ID}"
fi

# Update lambda function to use the zips from S3 (uploaded above)
# aws lambda update-function-code --function-name $FUNCTION --s3-bucket $BUCKET --s3-key podpac/podpac_dist_$TAG.zip
# aws lambda update-function-configuration --function-name $FUNCTION --handler handler.handler --timeout 300 --memory-size 2048
# aws apigateway update-rest-api --rest-api-id $API_ID --patch-operations "op=replace,path=/binaryMediaTypes/*~1*,value='*/*'"
RESOURCE=$(aws apigateway create-resource --rest-api-id $API_ID --parent-id $API_RESOURCE_ID --path-part 'lambda' --output text)
RESOURCE_ID=$(echo "$(echo $RESOURCE | cut -d " " -f1)")
aws apigateway put-method --rest-api-id $API_ID --resource-id $RESOURCE_ID --http-method ANY --authorization-type NONE

echo "Log in to AWS and perform the following steps:"
echo "1. Navigate to your API in the API Gateway and select the resource /lambda HTTP Method (ANY)."
echo "2. Select Integration Request -> Lambda Function, Check Use Lambda Proxy Integration, Select your lambda function region and function name."
echo "3. Press the Actions dropdown and select Deploy API. Select [New Stage] and create a stage name (doesn't matter exact name)"
echo "4. Navigate to your lambda function console and confirm you see API Gateway as a trigger."

# LAMBDA_URI=`$(aws lambda)
# aws apigateway put-integration --rest-api-id $API_ID --resource-id $API_RESOURCE_ID --http-method ANY --type AWS --integration-http-method POST --uri
