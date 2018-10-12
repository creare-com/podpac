# AWS Lambda #

Podpac includes a package to create an Amazon Web Services Lambda function to execute nodes in a server-less environment. This package can be altered to handle events according to the developer's use case.

## AWS Architecture ##

All files related to creating a Lambda function are in `dist/aws`. The `DockerFile` is based on Amazon's EC2 DockerHub distribution, and creates a Podpac-friendly python 3.6 environment. A `.zip` file is extracted from this environment, which can be used to create a Lambda function. Conveniently, developers can also use this to create an EC2 instance, or work directly in the Docker container.

Our `handler.py` expects the Lambda event to include a pipeline definition in the form of (URI encoded) JSON. The handler then executes that pipeline accordingly. However, developers are encouraged to write their own handlers as needed.

## Creating Your Own Podpac Lambda Function ##

We're now set up to create an AWS Lambda function "out of the box". Assuming you've installed Docker, here are the steps to create a Lambda function:

- Run `docker build -f DockerFile --tag $NAME:$TAG .` from the `dist/aws` directory
- Create a Lambda using the resulting `podpac:latest/tmp/package.zip`
  - For example, we've chosen to do this as follows:
    - ```bash
      docker run --name lambda -itd $NAME:$TAG
      docker cp lambda:/tmp/package.zip package.zip
      docker stop lambda
      docker rm lambda
      ```
    - Upload package.zip (~63 MB) to S3.
    - Create a Lambda function from the AWS developer console
    - Copy the link address of package.zip from its S3 bucket, paste into the Lambda's "Function code" field
    - Set up any other Lambda properties you'd like. We use S3 triggers - the handler is triggered when pipeline JSON is uploaded to our S3 bucket
