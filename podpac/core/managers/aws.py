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
    aws_access_key_id : string
        access key id from AWS credentials
    aws_secret_access_key : string`
        access key value from AWS credentials
    aws_region_name : string
        name of the AWS region
    source: Node
        node to be evaluated
    source_output_format: str
        output format for the evaluated results of `source`
    source_output_name: str
        output name for the evaluated results of `source`
    attrs: dict
        additional attributes passed on to the Lambda definition of the base node
    download_result: Bool
        Flag that indicated whether node should wait to download the data.
    """

    # aws parameters
    aws_access_key_id = tl.Unicode(help="Access key ID from AWS for S3 bucket.")

    @tl.default("aws_access_key_id")
    def _aws_access_key_id_default(self):
        return settings["aws_access_key_id"]

    aws_secret_access_key = tl.Unicode(help="Access key value from AWS for S3 bucket.")

    @tl.default("aws_secret_access_key")
    def _aws_secret_access_key_default(self):
        return settings["aws_secret_access_key"]

    aws_region_name = tl.Unicode(help="Region name of AWS S3 bucket.")

    @tl.default("aws_region_name")
    def _aws_region_name_default(self):
        return settings["aws_region_name"]

    # s3 parameters
    s3_bucket_name = tl.Unicode(help="Name of AWS s3 bucket.")

    @tl.default("s3_bucket_name")
    def _s3_bucket_name_default(self):
        return settings["S3_BUCKET_NAME"] or "podpac"

    s3_json_folder = tl.Unicode(help="S3 folder to put JSON in.")

    @tl.default("s3_json_folder")
    def _s3_json_folder_default(self):
        return settings["S3_JSON_FOLDER"] or "input"

    s3_output_folder = tl.Unicode(help="S3 folder to put output in.")

    @tl.default("s3_output_folder")
    def _s3_output_folder_default(self):
        return settings["S3_OUTPUT_FOLDER"] or "output"

    # lambda function name
    name = tl.Unicode()

    def _name_default(self):
        return "podpac-lambda"

    # podpac source
    source = tl.Instance(Node, help="Node to evaluate in a Lambda function.")

    source_output_format = tl.Unicode(default_value="pkl", help="Output format.")
    source_output_name = tl.Unicode(help="Image output name.")

    attrs = tl.Dict()

    download_result = tl.Bool(True).tag(attr=True)

    @tl.default("source_output_name")
    def _source_output_name_default(self):
        return self.source.__class__.__name__

    @property
    def pipeline(self):
        """
        The pipeline of this manager is the aggregation of the source node definition and the output.
        """
        d = OrderedDict()
        d["pipeline"] = self.source.definition
        if self.attrs:
            out_node = next(reversed(d["pipeline"].keys()))
            d["pipeline"][out_node]["attrs"].update(self.attrs)
        d["output"] = {"format": self.source_output_format}
        return d

    @common_doc(COMMON_DOC)
    def eval(self, coordinates, output=None):
        """
        Evaluate the source node on the AWS Lambda Function at the given coordinates
        """

        # add coordinates to the pipeline
        pipeline = self.pipeline
        pipeline["coordinates"] = json.loads(coordinates.json)

        # filename
        filename = "{folder}{slash}{output}_{source}_{coordinates}.{suffix}".format(
            folder=self.s3_json_folder,
            slash="/" if not self.s3_json_folder.endswith("/") else "",
            output=self.source_output_name,
            source=self.source.hash,
            coordinates=coordinates.hash,
            suffix="json",
        )

        # create s3 client with credentials
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
        s3 = boto3.client(
            "s3",
            region_name=self.aws_region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )

        # put pipeline into s3 bucket
        s3.put_object(
            Body=(bytes(json.dumps(pipeline, indent=4, cls=JSONEncoder).encode("UTF-8"))),
            Bucket=self.s3_bucket_name,
            Key=filename,
        )

        # wait for object to exist
        if not self.download_result:
            return

        waiter = s3.get_waiter("object_exists")
        filename = "{folder}{slash}{output}_{source}_{coordinates}.{suffix}".format(
            folder=self.s3_output_folder,
            slash="/" if not self.s3_output_folder.endswith("/") else "",
            output=self.source_output_name,
            source=self.source.hash,
            coordinates=coordinates.hash,
            suffix=self.source_output_format,
        )
        waiter.wait(Bucket=self.s3_bucket_name, Key=filename)

        # After waiting, load the pickle file like this:
        response = s3.get_object(Key=filename, Bucket=self.s3_bucket_name)
        body = response["Body"].read()
        self._output = cPickle.loads(body)
        return self._output

    def __repr__(self):
        rep = "{}\n".format(str(self.__class__.__name__))
        rep += "\tName: {}\n".format(self.name)
        return rep


if __name__ == "__main__":
    node = Lambda(name="podpac-mls-test2")

    print (node)
