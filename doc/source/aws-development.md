# AWS Development

This document provides details on how PODPAC integrates with AWS services for serverless cloud processing.
See the [AWS Quick Start Guide](aws.html) for a quick guide to building and using AWS services with PODPAC.

## AWS Architecture

## Creating PODPAC resources for AWS

> [Docker](https://www.docker.com/) is required for creating PODPAC resources for AWS services.

All files related to creating PODPAC resources for AWS live in [`dist/aws`](https://github.com/creare-com/podpac/tree/master/dist/aws).

- `handler.py`: [AWS Lambda function handler](https://docs.aws.amazon.com/lambda/latest/dg/python-programming-model-handler-types.html). Handles PODPAC Lambda trigger event, executes the pipeline, and returns the result back to the source of the trigger. Developers can override the default function handler for a [`Lambda`](api/podpac.managers.Lambda.html) Node using the [`function_handler`](api/podpac.managers.Lambda.html#podpac.managers.Lambda.function_handler) attribute.
- `DockerFile`: Docker instructions for creating PODPAC deployment package and dependencies using [Amazon's EC2 DockerHub](https://hub.docker.com/_/amazonlinux/) distribution.
- `build_lambda.sh`: Bash script to build PODPAC deployment package and dependencies using [Docker](https://www.docker.com/). Outputs `podpac_dist.zip` and `podpac_deps.zip` in the `dist/aws` directory.
  - `podpac_dist.zip`: PODPAC [deployment package](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-features.html#gettingstarted-features-package) ready to create a Lambda function.
  - `podpac_deps.zip`: PODPAC dependencies that are hosted on S3 and dynamically extracted during Lambda function execution. These files are seperate from `podpac_dist.zip` to circumvent the space limitations of AWS Lambda functions. 
- `upload_lambda.sh`: Convience script to upload deployment package and dependencies to an S3 bucket.

To create a custom deployment package or dependencies package:

- Edit the `Dockerfile` or `handler.py` with desired changes
  - To build using the local copy of the PODPAC repository,see comment in `Dockerfile` at ~L36.
- Build the deployment package and dependencies

```bash
$ bash build_lambda.sh
Creating docker image from podpac version master
podpac:master
...
Built podpac deployment package: podpac_dist.zip
Built podpac dependencies: podpac_deps.zip
```

- You can now use PODPAC to create a function from these local resources:

```python
import podpac

# configure settings
settings["AWS_ACCESS_KEY_ID"] = "access key id"
settings["AWS_SECRET_ACCESS_KEY"] = "secrect access key"
settings["AWS_REGION_NAME"] = "region name"
settings["S3_BUCKET_NAME"] = "bucket name"
settings["FUNCTION_NAME"] = "function name"
settings["FUNCTION_ROLE_NAME"] = "role name"

# define node
node = podpac.managers.aws.Lambda(function_source_dist_zip="dist/aws/podpac_dist.zip", 
                                  function_source_dependencies_zip="dist/aws/podpac_deps.zip"
                                  )

# build AWS resources
node.build()
```

- You can also upload `podpac_dist.zip` and `podpac_deps.zip` to a public or user-accessible S3 bucket and build PODPAC functions from the remote bucket. The bash script `upload_lambda.sh` can do this for you if the `BUCKET` variable is customized. 
We'll assume you copy the files to `s3://my-bucket/directory/podpac_dist.zip` and `s3://my-bucket/directory/podpac_deps.zip`:

```python
import podpac

# configure settings
settings["AWS_ACCESS_KEY_ID"] = "access key id"
settings["AWS_SECRET_ACCESS_KEY"] = "secrect access key"
settings["AWS_REGION_NAME"] = "region name"
settings["S3_BUCKET_NAME"] = "bucket name"
settings["FUNCTION_NAME"] = "function name"
settings["FUNCTION_ROLE_NAME"] = "role name"

# define node
node = podpac.managers.aws.Lambda(function_source_bucket="my-bucket", 
                                  function_source_dist_key="directory/podpac_dist.zip",
                                  function_source_dependencies_key="directory/podpac_deps.zip"
                                  )

# build AWS resources
node.build()
```
