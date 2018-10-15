"""
Lambda is `Node` manager, which executes the given `Node` on an AWS Lambda
function.
"""
import json
from base64 import b64decode
from collections import OrderedDict

import boto3
import traitlets as tl

from podpac import settings
from podpac.core.node import COMMON_NODE_DOC, Node
from podpac.core.pipeline.output import ImageOutput, Output
# from podpac.core.pipeline import Pipeline
from podpac.core.utils import common_doc

COMMON_DOC = COMMON_NODE_DOC.copy()


class Lambda(Node):
    """A `Node` wrapper to evaluate node on AWS Lambda function

    Attributes
    ----------
    aws_access_key_id : string
        access key id from AWS credentials
    aws_secret_access_key : string`
        access key value from AWS credentials
    aws_region_name : string
        name of the AWS region
    source_node: Node
        node to be evaluated
    source_output: Output
        how to output the evaluated results of `source_node`
    """

    aws_access_key_id = tl.Unicode(
        allow_none=False, help="Access key ID from AWS for S3 bucket.")

    @tl.default('aws_access_key_id')
    def _aws_access_key_id_default(self):
        return settings.aws_access_key_id

    aws_secret_access_key = tl.Unicode(
        allow_none=False, help="Access key value from AWS for S3 bucket.")

    @tl.default('aws_secret_access_key')
    def _aws_secret_access_key_default(self):
        return settings.aws_secret_access_key

    aws_region_name = tl.Unicode(
        allow_none=False, help="Region name of AWS S3 bucket.")

    @tl.default('aws_region_name')
    def _aws_region_name_default(self):
        return settings.aws_region_name

    source_node = tl.Instance(Node, allow_none=False,
                              help="Node to evaluate in a Lambda function.")

    source_output = tl.Instance(Output, allow_none=False,
                                help="Image output information.")

    @tl.default('source_output')
    def _source_output_default(self):
        return ImageOutput(node=self.source_node)

    s3_bucket_name = tl.Unicode(
        allow_none=False, help="Name of AWS s3 bucket.")

    @tl.default('s3_bucket_name')
    def _s3_bucket_name_default(self):
        return settings.S3_BUCKET_NAME

    s3_json_folder = tl.Unicode(
        allow_none=False, help="S3 folder to put JSON in.")

    @tl.default('s3_json_folder')
    def _s3_json_folder_default(self):
        return settings.S3_JSON_FOLDER

    s3_output_folder = tl.Unicode(
        allow_none=False, help="S3 folder to put output in.")

    @tl.default('s3_output_folder')
    def _s3_output_folder_default(self):
        return settings.S3_OUTPUT_FOLDER

    # _pipeline = tl.Instance(Pipeline, allow_none=False)

    # s3_client = tl.Instance(boto3.client,
    #                         allow_none=False, help="S3 client from boto3.")
    #
    # @tl.default('s3_client')
    # def _s3_client_default(self):
    #     return boto3.client('s3')
    #
    def __init__(self, source_node):
        super().__init__()
        self.source_node = source_node
    #     self._pipeline = Pipeline(self.definition())
        try:
            self.s3 = boto3.client('s3')
        except Exception as e:
            print("Error when instantiating S3 boto3 client: %s" % str(e))
            raise e

    @property
    def definition(self):
        """
        TOOD: Fill this out.
        """
        return self.source_node.definition

    @property
    def pipeline_definition(self):
        return self.source_node.pipeline_definition

    @common_doc(COMMON_DOC)
    def execute(self, coordinates, output=None, method=None):
        """
        TODO: Docstring
        """
        lambda_json = OrderedDict()
        lambda_json['pipeline'] = self.pipeline_definition
        lambda_json['output'] = [self.source_output.pipeline_definition]
        lambda_json['coordinates'] = coordinates.pipeline_definition

        data = json.loads(b64decode(lambda_json))
        self.s3.put_object(
            Body=(bytes(json.dumps(data, indent=4).encode('UTF-8'))),
            Bucket=self.s3_bucket_name,
            Key=self.s3_json_folder + self.source_output.name + '.json'
        )
