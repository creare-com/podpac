# AWS Integration

PODPAC integrates with AWS to enable processing in the cloud. To process on the cloud you need to:

1. Obtain and AWS account
2. Generate and save the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (see [AWS documentation](https://aws.amazon.com/blogs/security/wheres-my-secret-access-key/))
3. Build the necessary AWS resources using PODPAC (see the [Setting up AWS Lambda Tutorial Notebook](https://github.com/creare-com/podpac-examples/blob/master/notebooks/4-advanced/aws-lambda.ipynb))

After these steps, nearly any PODPAC processing pipeline can be evaluated using AWS Lambda functions. 

```python
import podpac
...
output = node.eval(coords)  # Local evaluation of node
cloud_node = podpac.managers.Lambda(source=node)
cloud_output = cloud_node.eval(coords)
```

This functionality is documented in the following notebooks:
* [Running on AWS Lambda Tutorial Notebook](https://github.com/creare-com/podpac-examples/blob/master/notebooks/3-processing/running-on-aws-lambda.ipynb)
* [Setting up AWS Lambda Tutorial Notebook](https://github.com/creare-com/podpac-examples/blob/master/notebooks/4-advanced/aws-lambda.ipynb)
* [Budgeting with AWS Lambda Tutorial Notebook](https://github.com/creare-com/podpac-examples/blob/master/notebooks/4-advanced/aws-budget.ipynb)
