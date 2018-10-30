"""
Lambda is `Node` manager, which executes the given `Node` on an AWS Lambda
function.
"""
import json
from collections import OrderedDict

import boto3
import traitlets as tl

from podpac import settings
from podpac.core.node import COMMON_NODE_DOC, Node
from podpac.core.pipeline.output import FileOutput, Output
# from podpac.core.pipeline import Pipeline
from podpac.core.utils import common_doc

COMMON_DOC = COMMON_NODE_DOC.copy()


class Lambda(Node):
    """A `Node` wrapper to evaluate source_node on AWS Lambda function

    Attributes
    ----------
    AWS_ACCESS_KEY_ID : string
        access key id from AWS credentials
    AWS_SECRET_ACCESS_KEY : string`
        access key value from AWS credentials
    AWS_REGION_NAME : string
        name of the AWS region
    source_node: Node
        node to be evaluated
    source_output: Output
        how to output the evaluated results of `source_node`
    """

    AWS_ACCESS_KEY_ID = tl.Unicode(
        allow_none=False, help="Access key ID from AWS for S3 bucket.")

    @tl.default('AWS_ACCESS_KEY_ID')
    def _AWS_ACCESS_KEY_ID_default(self):
        return settings.AWS_ACCESS_KEY_ID

    AWS_SECRET_ACCESS_KEY = tl.Unicode(
        allow_none=False, help="Access key value from AWS for S3 bucket.")

    @tl.default('AWS_SECRET_ACCESS_KEY')
    def _AWS_SECRET_ACCESS_KEY_default(self):
        return settings.AWS_SECRET_ACCESS_KEY

    AWS_REGION_NAME = tl.Unicode(
        allow_none=False, help="Region name of AWS S3 bucket.")

    @tl.default('AWS_REGION_NAME')
    def _AWS_REGION_NAME_default(self):
        return settings.AWS_REGION_NAME

    source_node = tl.Instance(Node, allow_none=False,
                              help="Node to evaluate in a Lambda function.")

    source_output = tl.Instance(Output, allow_none=False,
                                help="Image output information.")

    @tl.default('source_output')
    def _source_output_default(self):
        return FileOutput(node=self.source_node, name=self.source_node.__class__.__name__)

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

    def __init__(self, source_node):
        super().__init__()
        self.source_node = source_node
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
        _definition = OrderedDict()
        _definition['pipeline'] = self.source_node.pipeline_definition
        _definition['pipeline']['output'] = self.source_output.pipeline_definition
        return _definition

    @property
    def pipeline_json(self):
        return json.dumps(self.pipeline_definition, indent=4)

    @common_doc(COMMON_DOC)
    def execute(self, coordinates, output=None, method=None):
        """
        TODO: Docstring
        """
        _definition = self.pipeline_definition
        _definition['coordinates'] = json.loads(coordinates.json)
        self.s3.put_object(
            Body=(bytes(json.dumps(_definition, indent=4).encode('UTF-8'))),
            Bucket=self.s3_bucket_name,
            Key=self.s3_json_folder + self.source_output
            .name + '.json'
        )

        waiter = self.s3.get_waiter('object_exists')
        waiter.wait(Bucket=self.s3_bucket_name, Key=self.s3_output_folder +
                    self.source_output.name + '.' + self.source_output.format)
        output_object = self.s3.get_object(Bucket=self.s3_bucket_name, Key=self.s3_output_folder +
                    self.source_output.name + '.' + self.source_output.format)
