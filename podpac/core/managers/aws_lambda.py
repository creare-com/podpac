"""
Lambda is `Node` manager, which executes the given `Node` on an AWS Lambda
function.
"""
import json
from collections import OrderedDict

import boto3
import traitlets as tl

from io import BytesIO

from podpac.core.settings import settings
from podpac.core.node import COMMON_NODE_DOC, Node
from podpac.core.pipeline.output import FileOutput, Output
# from podpac.core.pipeline import Pipeline
from podpac.core.utils import common_doc, JSONEncoder

try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

COMMON_DOC = COMMON_NODE_DOC.copy()


class Lambda(Node):
    """A `Node` wrapper to evaluate source on AWS Lambda function

    Attributes
    ----------
    AWS_ACCESS_KEY_ID : string
        access key id from AWS credentials
    AWS_SECRET_ACCESS_KEY : string`
        access key value from AWS credentials
    AWS_REGION_NAME : string
        name of the AWS region
    source: Node
        node to be evaluated
    source_output: Output
        how to output the evaluated results of `source`
    attrs: dict
        additional attributes passed on to the Lambda definition of the base node
    """

    AWS_ACCESS_KEY_ID = tl.Unicode(
        allow_none=False, help="Access key ID from AWS for S3 bucket.")

    @tl.default('AWS_ACCESS_KEY_ID')
    def _AWS_ACCESS_KEY_ID_default(self):
        return settings['AWS_ACCESS_KEY_ID']

    AWS_SECRET_ACCESS_KEY = tl.Unicode(
        allow_none=False, help="Access key value from AWS for S3 bucket.")

    @tl.default('AWS_SECRET_ACCESS_KEY')
    def _AWS_SECRET_ACCESS_KEY_default(self):
        return settings['AWS_SECRET_ACCESS_KEY']

    AWS_REGION_NAME = tl.Unicode(
        allow_none=False, help="Region name of AWS S3 bucket.")

    @tl.default('AWS_REGION_NAME')
    def _AWS_REGION_NAME_default(self):
        return settings['AWS_REGION_NAME']

    source = tl.Instance(Node, allow_none=False,
                              help="Node to evaluate in a Lambda function.")

    source_output = tl.Instance(Output, allow_none=False,
                                help="Image output information.")

    attrs = tl.Dict()

    @tl.default('source_output')
    def _source_output_default(self):
        return FileOutput(node=self.source, name=self.source.__class__.__name__)

    s3_bucket_name = tl.Unicode(
        allow_none=False, help="Name of AWS s3 bucket.")

    @tl.default('s3_bucket_name')
    def _s3_bucket_name_default(self):
        return settings['S3_BUCKET_NAME']

    s3_json_folder = tl.Unicode(
        allow_none=False, help="S3 folder to put JSON in.")

    @tl.default('s3_json_folder')
    def _s3_json_folder_default(self):
        return settings['S3_JSON_FOLDER']

    s3_output_folder = tl.Unicode(
        allow_none=False, help="S3 folder to put output in.")

    @tl.default('s3_output_folder')
    def _s3_output_folder_default(self):
        return settings['S3_OUTPUT_FOLDER']

    @property
    def definition(self):
        """
        The definition of this manager is the aggregation of the source node
        and source output.
        """
        d = OrderedDict()
        d['pipeline'] = self.source.definition
        if self.attrs:
            out_node = next(reversed(d['pipeline']['nodes'].keys()))
            d['pipeline']['nodes'][out_node]['attrs'].update(self.attrs)
        d['pipeline']['output'] = self.source_output.definition
        return d

    @common_doc(COMMON_DOC)
    def eval(self, coordinates, output=None):
        """
        Evaluate the source node on the AWS Lambda Function at the given coordinates
        """

        d = self.definition
        d['coordinates'] = json.loads(coordinates.json)
        filename = '%s%s_%s_%s.%s' % (
            self.s3_json_folder,
            self.source_output.name,
            self.source.hash,
            coordinates.hash,
            'json')

        # create s3 client with credentials
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
        s3 = boto3.client('s3',
            region_name=self.AWS_REGION_NAME,
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY
        )

        # put pipeline into s3 bucket
        s3.put_object(
            Body=(bytes(json.dumps(d, indent=4, cls=JSONEncoder).encode('UTF-8'))),
            Bucket=self.s3_bucket_name,
            Key=filename
        )

        # wait for object to exist
        waiter = s3.get_waiter('object_exists')
        filename = '%s%s_%s_%s.%s' % (
            self.s3_output_folder,
            self.source_output.name,
            self.source.hash,
            coordinates.hash,
            self.source_output.format)
        waiter.wait(Bucket=self.s3_bucket_name, Key=filename)

        # After waiting, load the pickle file like this:
        resource = boto3.resource('s3')
        with BytesIO() as data:
            # Get the bucket and file name programmatically - see above...
            resource.Bucket(self.s3_bucket_name).download_fileobj(filename, data)
            data.seek(0)    # move back to the beginning after writing
            self._output = cPickle.load(data)
        return self._output
