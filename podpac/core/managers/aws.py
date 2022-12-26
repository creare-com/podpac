"""
Lambda is `Node` manager, which executes the given `Node` on an AWS Lambda
function.
"""
import json
from collections import OrderedDict
import logging
import time
import re
from copy import deepcopy
import base64
from datetime import datetime

from six import string_types

import traitlets as tl
import numpy as np

from podpac.core.units import UnitsDataArray
from podpac.core.settings import settings
from podpac.core.node import COMMON_NODE_DOC, Node
from podpac.core.utils import common_doc, JSONEncoder
from podpac import version

# Optional imports
from lazy_import import lazy_module, lazy_class

try:
    import boto3
    import botocore
except:

    class err:
        def __init__(self, *args, **kwargs):
            raise ImportError("boto3 is not installed, please install to use this functionality.")

    class boto3:
        Session = err


# Set up logging
_log = logging.getLogger(__name__)

COMMON_DOC = COMMON_NODE_DOC.copy()


class LambdaException(Exception):
    """Exception during execution of a Lambda node"""

    pass


class Lambda(Node):
    """A `Node` wrapper to evaluate source on AWS Lambda function


    Attributes
    ----------
    aws_access_key_id : str, optional
        Access key id from AWS credentials. If :attr:`session` is provided, this attribute will be ignored. Overrides :attr:`podpac.settings`.
    aws_region_name : str, optional
        Name of the AWS region. If :attr:`session` is provided, this attribute will be ignored. Overrides :attr:`podpac.settings`.
    aws_secret_access_key : str
        Access key value from AWS credentials. If :attr:`session` is provided, this attribute will be ignored. Overrides :attr:`podpac.settings`.
    function_name : str, optional
        Name of the lambda function to use or create. Defaults to :attr:`podpac.settings["FUNCTION_NAME"]` or "podpac-lambda-autogen".
    function_timeout : int, optional
        Timeout of the lambda function, in seconds. Defaults to 600.
    function_triggers : list of str, optional
        Methods to trigger this function. May only include ["eval", "S3", "APIGateway"]. During the :meth:`self.build()` process, this list will determine which AWS resources are linked to Lambda function. Defaults to ["eval"].
    function_role_name : str, optional
        Name of the AWS role created for lambda function. Defaults to :attr:`podpac.settings["FUNCTION_ROLE_NAME"]` or "podpac-lambda-autogen".
    function_s3_bucket : str, optional
        S3 bucket name to use with lambda function. Defaults to :attr:`podpac.settings["S3_BUCKET_NAME"]` or "podpac-autogen-<timestamp>" with the timestamp to ensure uniqueness.
    eval_settings : dict, optional
        Default is podpac.settings. PODPAC settings that will be used to evaluate the Lambda function.
    eval_timeout : float, optional
        Default is None. The amount of time to wait for an eval to return. To get near asynchronous response, set this to a small number.

    Other Attributes
    ----------------
    node_attrs : dict
        Additional attributes passed on to the Lambda definition of the base node
    download_result : Bool
        Flag that indicated whether node should wait to download the data.
    function_api_description : str, optional
        Description for the AWS API Gateway resource
    function_api_endpoint : str, optional
        Endpoint path for API Gateway. Defaults to "eval".
    function_api_name : str, optional
        AWS resource name for the API Gateway. Defaults to :attr:`self.function_name` + "-api".
    function_api_stage : str, optional
        Stage name for the API gateway. Defaults to "prod".
    function_api_tags : dict, optional
        AWS Tags for API Gateway resource. Defaults to :attr:`self.function_tags`.
    function_api_version : str, optional
        API Gateway version. Defaults to :meth:`podpac.verions.semver()`.
    function_description : str, optional
        Description for the AWS Lambda function resource
    function_env_variables : dict, optional
        Environment variables to use within the lambda function.
    function_eval_trigger : str, optional
        Function trigger to use during node eval process. Must be on of "eval" (default), "S3", or "APIGateway".
    function_handler : str, optional
        Handler method in Lambda function. Defaults to "handler.handler".
    function_memory : int, optional
        Memory allocated for each Lambda function. Defaults to 2048 MB.
    function_restrict_pipelines : list, optional
        List of Node hashes (see :class:`podpac.Node.hash`).
        Restricts lambda function evaluation to specific Node definitions.
    function_role_assume_policy_document : dict, optional.
        Assume policy document for role created. Defaults to allowing role to assume Lambda function.
    function_role_description : str, optional
        Description for the AWS role resource
    function_role_policy_arns : list of str, optional
        Managed role policy ARNs to attach to role.
    function_role_policy_document : dict, optional
        Inline role policies to put in role.
    function_role_tags : dict, optional
        AWS Tags for role resource. Defaults to :attr:`self.function_tags`.
    function_s3_dependencies_key : str, optional
        S3 path to copy and reference podpac dependencies. Defaults to "podpac_deps_<semver>.zip".
    function_s3_input : str, optional
        Folder in :attr:`self.function_s3_bucket` to store input pipelines when "S3" is included in :attr:`self.function_triggers`. Defaults to "input".
    function_s3_output : str, optional
        Folder in :attr:`self.function_s3_bucket` to watch for output when "S3" is included in :attr:`self.function_triggers`. Defaults to "output".
    function_s3_tags : dict, optional
        AWS Tags for S3 bucket resource. Defaults to :attr:`self.function_tags`.
    function_source_bucket : str, optional
        S3 Bucket to use for released podpac distribution during :meth:`self.build()` process. Defaults to "podpac-dist". This bucket is managed by the PODPAC distribution team.
    function_source_dependencies_key : str, optional
        S3 path within :attr:`self.function_source_bucket` to source podpac dependencies archive during :meth:`self.build()` process. Defaults to "<semver>/podpac_deps.zip".
    function_source_dependencies_zip : str, optional
        Override :attr:`self.function_source_dependencies_key` and upload custom source podpac dependencies archive to :attr:`self.function_s3_bucket` during :meth:`self.build()` process.
    function_source_dist_key : str, optional
        S3 path within :attr:`self.function_source_bucket` to source podpac dist archive during :meth:`self.build()` process. Defaults to "<semver>/podpac_dist.zip".
    function_source_dist_zip : str, optional
        Override :attr:`self.function_source_dist_key` and create lambda function using custom source podpac dist archive to :attr:`self.function_s3_bucket` during :meth:`self.build()` process.
    function_tags : dict, optional
        AWS Tags for Lambda function resource. Defaults to :attr:`podpac.settings["AWS_TAGS"]` or {}.
    function_budget_amount : float, optional
        EXPERIMENTAL FEATURE
        Monthly budget for function and associated AWS resources.
        When usage reaches 80% of this amount, AWS will notify :attr:`function_budget_email`.
        Defaults to :attr:`podpac.settings["AWS_BUDGET_AMOUNT"]`.
    function_budget_email : str, optional
        EXPERIMENTAL FEATURE
        Email to notify when usage reaches 80% of :attr:`function_budget_amount`.
        Defaults to :attr:`podpac.settings["AWS_BUDGET_EMAIL"]`.
    function_budget_name : str, optional
        EXPERIMENTAL FEATURE
        Name for AWS budget
    function_budget_currency : str, optional
        EXPERIMENTAL FEATURE
        Currency type for the :attr:`function_budget_amount`.
        Defaults to "USD".
        See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/budgets.html#Budgets.Client.create_budget
        for currency (or Unit) options.
    output_format : dict, optional
        Definition for how output is saved after results are computed.
    session : :class:`podpac.managers.aws.Session`
        AWS Session to use for this node.
    source : :class:`podpac.Node`
        Node to be evaluated on the Lambda function.
    source_output_format : str
        Output format for the evaluated results of `source`
    source_output_name : str
        Output name for the evaluated results of `source`
    """

    # aws parameters - defaults are handled in Session
    aws_access_key_id = tl.Unicode(default_value=None, allow_none=True)
    aws_secret_access_key = tl.Unicode(default_value=None, allow_none=True)
    aws_region_name = tl.Unicode(default_value=None, allow_none=True)
    session = tl.Instance(boto3.Session)

    @tl.default("session")
    def _session_default(self):
        # defaults to "settings" if None
        return Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region_name,
        )

    # general function parameters
    function_eval_trigger = tl.Enum(["eval", "S3", "APIGateway"], default_value="eval").tag(attr=True)

    # lambda function parameters
    function_name = tl.Unicode().tag(attr=True, readonly=True)  # see default below
    function_triggers = tl.List(tl.Enum(["eval", "S3", "APIGateway"])).tag(readonly=True)
    function_handler = tl.Unicode(default_value="handler.handler").tag(readonly=True)
    function_description = tl.Unicode(default_value="PODPAC Lambda Function (https://podpac.org)").tag(readonly=True)
    function_env_variables = tl.Dict(default_value={}).tag(readonly=True)  # environment vars in function
    function_tags = tl.Dict().tag(readonly=True)  # key: value for tags on function (and any created roles)
    function_timeout = tl.Int(default_value=600).tag(readonly=True)
    function_memory = tl.Int(default_value=2048).tag(readonly=True)
    function_source_dist_zip = tl.Unicode(default_value=None, allow_none=True).tag(
        readonly=True
    )  # override published podpac archive with local file
    function_source_dependencies_zip = tl.Unicode(default_value=None, allow_none=True).tag(
        readonly=True
    )  # override published podpac deps archive with local file
    function_source_bucket = tl.Unicode(default_value="podpac-dist", allow_none=True).tag(readonly=True)
    function_source_dist_key = tl.Unicode().tag(readonly=True)  # see default below
    function_source_dependencies_key = tl.Unicode().tag(readonly=True)  # see default below
    function_allow_unsafe_eval = tl.Bool().tag(readonly=True)  # see default below
    function_restrict_pipelines = tl.List(tl.Unicode(), default_value=[]).tag(readonly=True)
    _function_arn = tl.Unicode(default_value=None, allow_none=True)
    _function_last_modified = tl.Unicode(default_value=None, allow_none=True)
    _function_version = tl.Unicode(default_value=None, allow_none=True)
    _function_code_sha256 = tl.Unicode(default_value=None, allow_none=True)
    _function_triggers = tl.Dict(default_value={}, allow_none=True)
    _function_valid = tl.Bool(default_value=False, allow_none=True)
    _function = tl.Dict(default_value=None, allow_none=True)  # raw response from AWS on "get_"

    output_format = tl.Dict(None, allow_none=True).tag(attr=True)

    @property
    def outputs(self):
        return self.source.outputs

    @tl.default("function_name")
    def _function_name_default(self):
        if settings["FUNCTION_NAME"] is None:
            settings["FUNCTION_NAME"] = "podpac-lambda-autogen"

        return settings["FUNCTION_NAME"]

    @tl.default("function_triggers")
    def _function_triggers_default(self):
        if self.function_eval_trigger != "eval":
            return ["eval", self.function_eval_trigger]
        else:
            return ["eval"]

    @tl.default("function_source_dist_key")
    def _function_source_dist_key_default(self):
        v = version.version()
        if "+" in v:
            v = "dev"

        return "{}/podpac_dist.zip".format(v)

    @tl.default("function_source_dependencies_key")
    def _function_source_dependencies_key_default(self):
        v = version.version()
        if "+" in v:
            v = "dev"

        return "{}/podpac_deps.zip".format(v)

    @tl.default("function_tags")
    def _function_tags_default(self):
        return settings["AWS_TAGS"] or {}

    @tl.default("function_allow_unsafe_eval")
    def _function_allow_unsafe_eval_default(self):
        return "UNSAFE_EVAL_HASH" in self.eval_settings and isinstance(
            self.eval_settings["UNSAFE_EVAL_HASH"], string_types
        )

    # role parameters
    function_role_name = tl.Unicode().tag(readonly=True)  # see default below
    function_role_description = tl.Unicode(default_value="PODPAC Lambda Role").tag(readonly=True)
    function_role_policy_document = tl.Dict(allow_none=True).tag(readonly=True)  # see default below - can be none
    function_role_policy_arns = tl.List(
        default_value=[
            "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        ]  # allows read/write to cloudwatch
    ).tag(readonly=True)
    function_role_assume_policy_document = tl.Dict().tag(readonly=True)  # see default below
    function_role_tags = tl.Dict().tag(readonly=True)  # see default below
    _function_role_arn = tl.Unicode(default_value=None, allow_none=True)
    _role = tl.Dict(default_value=None, allow_none=True)  # raw response from AWS on "get_"

    @tl.default("function_role_name")
    def _function_role_name_default(self):
        if settings["FUNCTION_ROLE_NAME"] is None:
            settings["FUNCTION_ROLE_NAME"] = "podpac-lambda-autogen"

        return settings["FUNCTION_ROLE_NAME"]

    @tl.default("function_role_policy_document")
    def _function_role_policy_document_default(self):
        # enable role to be run by lambda - this document is defined by AWS
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:PutObject",
                        "s3:GetObject",
                        "s3:DeleteObject",
                        "s3:ReplicateObject",
                        "s3:ListBucket",
                        "s3:ListMultipartUploadParts",
                        "s3:ListBucketByTags",
                        "s3:GetBucketTagging",
                        "s3:ListBucketVersions",
                        "s3:AbortMultipartUpload",
                        "s3:GetObjectTagging",
                        "s3:ListBucketMultipartUploads",
                        "s3:GetBucketLocation",
                        "s3:GetObjectVersion",
                    ],
                    "Resource": ["arn:aws:s3:::{}/*".format(self.function_s3_bucket)],
                }
            ],
        }

    @tl.default("function_role_assume_policy_document")
    def _function_role_assume_policy_document_default(self):
        # enable role to be run by lambda - this document is defined by AWS
        return {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}
            ],
        }

    @tl.default("function_role_tags")
    def _function_role_tags_default(self):
        return self.function_tags

    # s3 parameters
    function_s3_bucket = tl.Unicode().tag(attr=True, readonly=True)  # see default below
    function_s3_dependencies_key = tl.Unicode()  # see default below
    function_s3_input = tl.Unicode()  # see default below
    function_s3_output = tl.Unicode()  # see default below
    function_s3_tags = tl.Dict()  # see default below
    _bucket = tl.Dict(default_value=None, allow_none=True)  # raw response from AWS on "get_"

    @tl.default("function_s3_bucket")
    def _function_s3_bucket_default(self):
        return settings["S3_BUCKET_NAME"] or "podpac-autogen-{}".format(
            np.datetime64("now").astype(int)
        )  # must be globally unique

    @tl.default("function_s3_input")
    def _function_s3_input_default(self):
        if settings["FUNCTION_S3_INPUT"] is None:
            settings["FUNCTION_S3_INPUT"] = "input/"

        return settings["FUNCTION_S3_INPUT"]

    @tl.default("function_s3_output")
    def _function_s3_output_default(self):
        if settings["FUNCTION_S3_OUTPUT"] is None:
            settings["FUNCTION_S3_OUTPUT"] = "output/"

        return settings["FUNCTION_S3_OUTPUT"]

    @tl.default("function_s3_tags")
    def _function_s3_tags_default(self):
        return self.function_tags

    @tl.default("function_s3_dependencies_key")
    def _function_s3_dependencies_key_default(self):
        if settings["FUNCTION_DEPENDENCIES_KEY"] is None:
            settings["FUNCTION_DEPENDENCIES_KEY"] = "podpac_deps_{}.zip".format(version.semver())

        return settings["FUNCTION_DEPENDENCIES_KEY"]

    # api gateway parameters
    function_api_name = tl.Unicode().tag(readonly=True)  # see default below
    function_api_description = tl.Unicode().tag(readonly=True)  # see default below
    function_api_version = tl.Unicode(default_value="{}".format(version.semver())).tag(readonly=True)
    function_api_tags = tl.Dict().tag(readonly=True)  # see default below
    function_api_stage = tl.Unicode(default_value="prod").tag(readonly=True)
    function_api_endpoint = tl.Unicode(default_value="eval").tag(readonly=True)
    _function_api_id = tl.Unicode(default_value=None, allow_none=True)  # will create api if None
    _function_api_url = tl.Unicode(default_value=None, allow_none=True)
    _function_api_resource_id = tl.Unicode(default_value=None, allow_none=True)
    _api = tl.Dict(default_value=None, allow_none=True)  # raw response from AWS on "get_"

    @tl.default("function_api_name")
    def _function_api_name_default(self):
        return "{}-api".format(self.function_name)

    @tl.default("function_api_description")
    def _function_api_description_default(self):
        return "PODPAC Lambda REST API for {} function".format(self.function_name)

    @tl.default("function_api_tags")
    def _function_api_tags_default(self):
        return self.function_tags

    # budget parameters
    function_budget_amount = tl.Float(allow_none=True).tag(readonly=True)  # see default below
    function_budget_email = tl.Unicode(allow_none=True).tag(readonly=True)  # see default below
    function_budget_name = tl.Unicode().tag(readonly=True)  # see default below
    function_budget_currency = tl.Unicode(default_value="USD").tag(readonly=True)
    _budget = tl.Dict(default_value=None, allow_none=True)  # raw response from AWS on "get_"

    @tl.default("function_budget_amount")
    def _function_budget_amount_default(self):
        return settings["AWS_BUDGET_AMOUNT"] or None

    @tl.default("function_budget_email")
    def _function_budget_email_default(self):
        return settings["AWS_BUDGET_EMAIL"] or None

    @tl.default("function_budget_name")
    def _function_budget_name_default(self):
        return "{}-budget".format(self.function_name)

    # podpac node parameters
    source = tl.Instance(Node, allow_none=True).tag(attr=True)
    source_output_format = tl.Unicode(default_value="netcdf")
    source_output_name = tl.Unicode()
    node_attrs = tl.Dict()
    download_result = tl.Bool(True).tag(attr=True)
    force_compute = tl.Bool().tag(attr=True)
    eval_settings = tl.Dict().tag(attr=True)
    eval_timeout = tl.Float(610).tag(attr=True)

    @tl.default("source_output_name")
    def _source_output_name_default(self):
        return self.source.__class__.__name__

    @tl.default("force_compute")
    def _force_compute_default(self):
        if settings["FUNCTION_FORCE_COMPUTE"] is None:
            settings["FUNCTION_FORCE_COMPUTE"] = False

        return settings["FUNCTION_FORCE_COMPUTE"]

    @tl.default("eval_settings")
    def _eval_settings_default(self):
        return settings.copy()

    @property
    def pipeline(self):
        """
        The pipeline of this manager is the aggregation of the source node definition and the output.
        """
        d = OrderedDict()
        d["pipeline"] = self.source.definition
        if self.node_attrs:
            out_node = next(reversed(d["pipeline"].keys()))
            d["pipeline"][out_node]["attrs"].update(self.node_attrs)
        d["output"] = {"format": self.source_output_format}
        d["settings"] = self.eval_settings
        return d

    @common_doc(COMMON_DOC)
    def eval(self, coordinates, output=None, selector=None):
        """
        Evaluate the source node on the AWS Lambda Function at the given coordinates
        """
        if self.source is None:
            raise ValueError("'source' node must be defined to eval")

        if self.function_eval_trigger == "eval":
            return self._eval_invoke(coordinates, output)
        elif self.function_eval_trigger == "S3":
            return self._eval_s3(coordinates, output)
        elif self.function_eval_trigger == "APIGateway":
            raise NotImplementedError("APIGateway trigger not yet implemented through eval")
        else:
            raise ValueError("Function trigger is not one of 'eval', 'S3', or 'APIGateway'")

    def build(self):
        """Build Lambda function and associated resources on AWS
        to run PODPAC pipelines
        """

        # TODO: move towards an architecture where the "create_" functions repair on each build
        # and skip when the resources already exist

        # see if current setup is valid, if so just return
        valid = self.validate()
        if valid:
            _log.debug("Current cloud resources will support this PODPAC lambda function")
            return

        # TODO: how much "importing" do we want to do? Currently, this will see if the cloud resource is available
        # and if so, assume that it will work with the function setup

        # self.validate updates current properties (self.function, self.role, self.bucket, self.api)
        # create default role if it doesn't exist
        if self._role is None:
            self.create_role()

            # after creating a role, you need to wait ~10 seconds before its active and will work with the lambda function
            # this is not cool
            time.sleep(10)

        # create function
        if self._function is None:
            self.create_function()

            # after creating a role, you need to wait ~5 seconds before its active and will return an arn
            # this is also not cool
            time.sleep(5)

        # TODO: check to make sure function and role work together

        # create API gateway
        if self._api is None:
            self.create_api()

        # TODO: check to make sure function and API work together

        # create S3 bucket - this will skip pieces that already exist
        self.create_bucket()

        # create budget, if defined
        if self.function_budget_amount is not None:
            self.create_budget()

        # check to see if setup is valid after creation
        # TODO: remove this in favor of something more granular??
        self.validate(raise_exceptions=True)

        _log.info("Successfully built AWS resources to support function {}".format(self.function_name))

    def validate(self, raise_exceptions=False):
        """
        Validate cloud resources and interoperability of resources for
        PODPAC usage

        Parameters
        ----------
        raise_exceptions : bool, optional
            Raise validation errors when encountered
        """

        # TODO: I don't know if this is the right architecture to handle validation
        # perhaps we just want to improve the "create_" methods to be self-healing

        def _raise(msg):
            _log.debug(msg)
            if raise_exceptions:
                raise Exception(msg)
            else:
                return False

        # get currently defined resources
        self.get_role()
        self.get_function()
        self.get_api()
        self.get_bucket()
        self.get_budget()

        # check that each resource has a valid configuration
        if not self.validate_role():
            return _raise("Failed to validate role")

        if not self.validate_function():
            return _raise("Failed to validate function")

        if not self.validate_bucket():
            return _raise("Failed to validate bucket")

        if not self.validate_api():
            return _raise("Failed to validate API")

        if not self.validate_budget():
            return _raise("Failed to validate budget")

        # check that the integration of resources is correct

        # check that role_arn is the same as function configured role
        if self._function["Configuration"]["Role"] != self._function_role_arn:
            return _raise("Function role ARN is not the same as role ARN for {}".format(self.function_role_name))

        # if it makes it to the end, its valid
        self._function_valid = True
        return True

    def delete(self, confirm=False):
        """Remove all cloud resources associated with function

        Parameters
        ----------
        confirm : bool, optional
            Must pass in confirm paramter
        """
        _log.info("Removing all cloud resources associated with this Lamba node")

        if confirm:
            self.remove_triggers()
            self.delete_function()
            self.delete_role()
            self.delete_api()
            self.delete_bucket(delete_objects=True)
            self.delete_budget()
        else:
            raise ValueError("You must pass confirm=True to delete all AWS resources")

    def describe(self):
        """Show a description of the Lambda Utilities"""
        # TODO: change this to format strings when we deprecate py 2
        status = "(staged)" if not self._function_valid else "(built)"

        # source dist
        if not self._function_valid:
            source_dist = (
                self.function_source_dist_zip
                if self.function_source_dist_zip is not None
                else "s3://{}/{}".format(self.function_source_bucket, self.function_source_dist_key)
            )
        else:
            source_dist = self._function_code_sha256

        # source deps
        if not self._function_valid:
            source_deps = (
                self.function_source_dependencies_zip
                if self.function_source_dependencies_zip is not None
                else "s3://{}/{}".format(self.function_source_bucket, self.function_source_dependencies_key)
            )
        else:
            source_deps = "s3://{}/{}".format(self.function_s3_bucket, self.function_s3_dependencies_key)

        # only show API if built or if its proposed in triggers
        if self._api is not None or (not self._function_valid and "APIGatway" in self.function_triggers):
            api_output = """
    API
        Name: {function_api_name}
        Description: {function_api_description}
        ID: {_function_api_id}
        Resource ID: {_function_api_resource_id}
        Version: {function_api_version}
        Tags: {function_api_tags}
        Stage: {function_api_stage}
        Endpoint: {function_api_endpoint}
        URL: {_function_api_url}
            """.format(
                function_api_name=self.function_api_name,
                function_api_description=self.function_api_description,
                _function_api_id=self._function_api_id,
                _function_api_resource_id=self._function_api_resource_id,
                function_api_version=self.function_api_version,
                function_api_tags=self.function_api_tags,
                function_api_stage=self.function_api_stage,
                function_api_endpoint=self.function_api_endpoint,
                _function_api_url=self._function_api_url,
            )
        else:
            api_output = ""

        # only show budget if its defined
        if self.function_budget_amount is not None:
            budget_output = """
    Budget
        Name: {function_budget_name}
        Amount: {function_budget_amount}
        Currency: {function_budget_currency}
        E-mail: {function_budget_email}
        Spent: {function_budget_usage} {function_budget_usage_currency}
            """.format(
                function_budget_name=self.function_budget_name,
                function_budget_amount=self.function_budget_amount,
                function_budget_currency=self.function_budget_currency,
                function_budget_email=self.function_budget_email,
                function_budget_usage=self._budget["CalculatedSpend"]["ActualSpend"]["Amount"]
                if self._budget
                else None,
                function_budget_usage_currency=self._budget["CalculatedSpend"]["ActualSpend"]["Unit"]
                if self._budget
                else None,
            )
        else:
            budget_output = ""

        output = """
Lambda Node {status}
    Function
        Name: {function_name}
        Description: {function_description}
        ARN: {_function_arn}
        Triggers: {function_triggers}
        Handler: {function_handler}
        Environment Variables: {function_env_variables}
        Timeout: {function_timeout} seconds
        Memory: {function_memory} MB
        Tags: {function_tags}
        Source Dist: {source_dist}
        Source Dependencies: {source_deps}
        Last Modified: {_function_last_modified}
        Version: {_function_version}
        Restrict Evaluation: {function_restrict_pipelines}

    S3
        Bucket: {function_s3_bucket}
        Tags: {function_s3_tags}
        Input Folder: {function_s3_input}
        Output Folder: {function_s3_output}

    Role
        Name: {function_role_name}
        Description: {function_role_description}
        ARN: {_function_role_arn}
        Policy Document: {function_role_policy_document}
        Policy ARNs: {function_role_policy_arns}
        Assume Policy Document: {function_role_assume_policy_document}
        Tags: {function_role_tags}

        {api_output}

        {budget_output}
        """.format(
            status=status,
            function_name=self.function_name,
            function_description=self.function_description,
            function_triggers=self.function_triggers,
            function_handler=self.function_handler,
            function_env_variables=self.function_env_variables,
            function_timeout=self.function_timeout,
            function_memory=self.function_memory,
            function_tags=self.function_tags,
            source_dist=source_dist,
            source_deps=source_deps,
            _function_arn=self._function_arn,
            _function_last_modified=self._function_last_modified,
            _function_version=self._function_version,
            function_restrict_pipelines=self.function_restrict_pipelines,
            function_s3_bucket=self.function_s3_bucket,
            function_s3_tags=self.function_s3_tags,
            function_s3_input=self.function_s3_input,
            function_s3_output=self.function_s3_output,
            function_role_name=self.function_role_name,
            function_role_description=self.function_role_description,
            function_role_policy_document=self.function_role_policy_document,
            function_role_policy_arns=self.function_role_policy_arns,
            function_role_assume_policy_document=self.function_role_assume_policy_document,
            function_role_tags=self.function_role_tags,
            _function_role_arn=self._function_role_arn,
            api_output=api_output,
            budget_output=budget_output,
        )

        print(output)

    # Function
    def create_function(self):
        """Build Lambda function on AWS"""
        if self.function_name is None:
            raise AttributeError("Function name is not defined")

        if self.function_allow_unsafe_eval:
            _log.info("Lambda function will allow unsafe evaluation of Nodes with the current settings")
            self.function_env_variables["PODPAC_UNSAFE_EVAL"] = settings["UNSAFE_EVAL_HASH"]

        if self.function_restrict_pipelines:
            _log.info("Lambda function will only run for pipelines: {}".format(self.function_restrict_pipelines))
            self.function_env_variables["PODPAC_RESTRICT_PIPELINES"] = json.dumps(self.function_restrict_pipelines)

        # add special tag - value is hash, for lack of better value at this point
        self.function_tags["_podpac_resource_hash"] = self.hash

        # if function already exists, this will return existing function
        function = create_function(
            self.session,
            self.function_name,
            self._function_role_arn,
            self.function_handler,
            self.function_description,
            self.function_timeout,
            self.function_memory,
            self.function_env_variables,
            self.function_tags,
            self.function_source_dist_zip,
            self.function_source_bucket,
            self.function_source_dist_key,
        )

        # set class properties
        self._set_function(function)

    def update_function(self):
        """Update lambda function with new parameters"""
        if self.function_name is None:
            raise AttributeError("Function name is not defined")

        # if function already exists, this will return existing function
        function = update_function(
            self.session,
            self.function_name,
            self.function_source_dist_zip,
            self.function_source_bucket,
            self.function_source_dist_key,
        )

        # set class properties
        self._set_function(function)

    def get_function(self):
        """Get function definition from AWS

        Returns
        -------
        dict
            See :func:`podpac.managers.aws.get_function`
        """
        function = get_function(self.session, self.function_name)
        self._set_function(function)

        return function

    def validate_function(self):
        """
        Validate that function is configured properly

        This should only be run after running `self.get_function()`
        """
        # TOOD: implement

        if self._function is None:
            return False

        return True

    def delete_function(self):
        """Remove AWS Lambda function and associated resources on AWS"""

        self.get_function()

        delete_function(self.session, self.function_name)

        # reset internals
        self._function = None
        self._function_arn = None
        self._function_last_modified = None
        self._function_version = None
        self._function_code_sha256 = None

    def add_trigger(self, statement_id, principle, source_arn):
        """Add trigger (permission) to lambda function

        Parameters
        ----------
        statement_id : str
            Specific identifier for trigger
        principle : str
            Principle identifier from AWS
        source_arn : str
            Source ARN for trigger
        """
        add_function_trigger(self.session, self.function_name, statement_id, principle, source_arn)
        self._function_triggers[statement_id] = source_arn

    def remove_trigger(self, statement_id):
        """Remove trigger (permission) from lambda function

        Parameters
        ----------
        statement_id : str
            Specific identifier for trigger
        """

        remove_function_trigger(self.session, self.function_name, statement_id)

        # remove from local dict
        del self._function_triggers[statement_id]

    def remove_triggers(self):
        """
        Remove all triggers from function
        """
        triggers = deepcopy(self._function_triggers)  # to avoid changing the size of dict during iteration
        for trigger in triggers:
            self.remove_trigger(trigger)

    # IAM Roles
    def create_role(self):
        """Create IAM role to execute podpac lambda function"""

        # add special tag - value is hash
        self.function_role_tags["_podpac_resource_hash"] = self.hash

        role = create_role(
            self.session,
            self.function_role_name,
            self.function_role_description,
            self.function_role_policy_document,
            self.function_role_policy_arns,
            self.function_role_assume_policy_document,
            self.function_role_tags,
        )

        self._set_role(role)

    def get_role(self):
        """Get role definition from AWS

        See :attr:`self.function_role_name` for role_name

        Returns
        -------
        dict
            See :func:`podpac.managers.aws.get_role`
        """
        role = get_role(self.session, self.function_role_name)
        self._set_role(role)

        return role

    def validate_role(self):
        """
        Validate that role will work with function.

        This should only be run after running `self.get_role()`
        """
        # TODO: add constraints

        if self._role is None:
            return False

        # check role policy document
        document_valid = False
        valid_document = {
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
        for s in self.function_role_assume_policy_document["Statement"]:
            if json.dumps(s) == json.dumps(valid_document):
                document_valid = True

        if not document_valid:
            _log.error("Function role policy document does not allow lambda function to assume role")
            return False

        return True

    def delete_role(self):
        """Remove role from AWS resources

        See :attr:`self.function_role_name` for role_name
        """
        self.get_role()

        if self.function_role_name is None:
            _log.debug("No role name defined for this function")
            return

        delete_role(self.session, self.function_role_name)

        # reset members
        self._role = None
        self._function_role_arn = None

        # TODO: handle defaults after deletion

    # S3 Creation
    def create_bucket(self):
        """Create S3 bucket to work with function"""
        if self.function_name is None:
            raise AttributeError("Function name must be defined when creating S3 bucket and trigger")

        if self._function_arn is None:
            raise ValueError("Lambda function must be created before creating a bucket")

        if self._function_role_arn is None:
            raise ValueError("Function role must be created before creating a bucket")

        # add special tags - value is hash
        self.function_s3_tags["_podpac_resource_hash"] = self.hash

        # create bucket
        bucket = create_bucket(
            self.session, self.function_s3_bucket, bucket_policy=None, bucket_tags=self.function_s3_tags
        )
        self._set_bucket(bucket)

        # after creating a bucket, you need to wait ~2 seconds before its active and can be uploaded to
        # this is not cool
        time.sleep(5)

        # get reference to s3 client for session
        s3 = self.session.client("s3")

        # add podpac deps to bucket for version
        # see if the function depedencies exist in bucket
        try:
            s3.head_object(Bucket=self.function_s3_bucket, Key=self.function_s3_dependencies_key)
        except botocore.exceptions.ClientError:

            # copy from user supplied dependencies
            if self.function_source_dependencies_zip is not None:
                put_object(
                    self.session,
                    self.function_s3_bucket,
                    self.function_s3_dependencies_key,
                    file=self.function_source_dependencies_zip,
                )

            # copy resources from podpac dist
            else:
                s3resource = self.session.resource("s3")
                copy_source = {"Bucket": self.function_source_bucket, "Key": self.function_source_dependencies_key}
                s3resource.meta.client.copy(copy_source, self.function_s3_bucket, self.function_s3_dependencies_key)

        # Add S3 Function triggers, if they don't exist already
        # TODO: add validition to see if trigger already exists
        if "S3" in self.function_triggers:
            # add permission to invoke call lambda - this feels brittle due to source_arn
            statement_id = re.sub("[-_.]", "", self.function_s3_bucket)
            principle = "s3.amazonaws.com"
            source_arn = "arn:aws:s3:::{}".format(self.function_s3_bucket)
            self.add_trigger(statement_id, principle, source_arn)

            # lambda integration on object creation events
            s3.put_bucket_notification_configuration(
                Bucket=self.function_s3_bucket,
                NotificationConfiguration={
                    "LambdaFunctionConfigurations": [
                        {
                            "Id": "{}".format(np.datetime64("now").astype(int)),
                            "LambdaFunctionArn": self._function_arn,
                            "Events": ["s3:ObjectCreated:*"],
                            "Filter": {"Key": {"FilterRules": [{"Name": "prefix", "Value": self.function_s3_input}]}},
                        }
                    ]
                },
            )
        else:
            _log.debug("Skipping S3 trigger because 'S3' not in the function triggers")

        return bucket

    def get_bucket(self):
        """Get S3 Bucket for function

        Returns
        -------
        dict
            See :func:`podpac.managers.aws.get_bucket`
        """
        bucket = get_bucket(self.session, self.function_s3_bucket)
        self._set_bucket(bucket)

        return bucket

    def validate_bucket(self):
        """
        Validate that bucket will work with function.

        This should only be run after running `self.get_bucket()`
        """
        if self._bucket is None:
            return False

        s3 = self.session.client("s3")

        # make sure dependencies are in there
        try:
            s3.head_object(Bucket=self.function_s3_bucket, Key=self.function_s3_dependencies_key)
        except botocore.exceptions.ClientError:
            _log.error("Failed to find PODPAC dependencies in bucket")
            return False

        # TODO: make sure trigger exists
        if "S3" in self.function_triggers:
            pass

        return True

    def delete_bucket(self, delete_objects=False):
        """Delete bucket associated with this function

        Parameters
        ----------
        delete_objects : bool, optional
            Delete all objects in the bucket while deleting bucket. Defaults to False.
        """

        self.get_bucket()

        # delete bucket
        delete_bucket(self.session, self.function_s3_bucket, delete_objects=delete_objects)

        # TODO: update manage attributes here?
        self._bucket = None

    # API Gateway
    def create_api(self):
        """Create API Gateway API for lambda function"""

        if "APIGateway" not in self.function_triggers:
            _log.debug("Skipping API creation because 'APIGateway' not in the function triggers")
            return

        if self.function_name is None:
            raise AttributeError("Function name must be defined when creating API Gateway")

        if self._function_arn is None:
            raise ValueError("Lambda function must be created before creating an API bucket")

        # add special tag - value is hash
        self.function_api_tags["_podpac_resource_hash"] = self.hash

        # create api and resource
        api = create_api(
            self.session,
            self.function_api_name,
            self.function_api_description,
            self.function_api_version,
            self.function_api_tags,
            self.function_api_endpoint,
        )
        self._set_api(api)

        # lambda proxy integration - this feels pretty brittle due to uri
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigateway.html#APIGateway.Client.put_integration
        aws_lambda_uri = "arn:aws:apigateway:{}:lambda:path/2015-03-31/functions".format(self.session.region_name)
        uri = "{}/{}/invocations".format(aws_lambda_uri, self._function_arn)

        apigateway = self.session.client("apigateway")
        apigateway.put_integration(
            restApiId=api["id"],
            resourceId=api["resource"]["id"],
            httpMethod="ANY",
            integrationHttpMethod="POST",
            type="AWS_PROXY",
            uri=uri,
            passthroughBehavior="WHEN_NO_MATCH",
            contentHandling="CONVERT_TO_TEXT",
            timeoutInMillis=29000,
        )

        # get responses back
        apigateway.put_integration_response(
            restApiId=api["id"],
            resourceId=api["resource"]["id"],
            httpMethod="ANY",
            statusCode="200",
            selectionPattern="",  # bug, see https://github.com/aws/aws-sdk-ruby/issues/1080
        )

        # deploy the api. this has to happen after creating the integration
        deploy_api(self.session, self._function_api_id, self.function_api_stage)

        # add permission to invoke call lambda - this feels brittle due to source_arn
        statement_id = api["id"]
        principle = "apigateway.amazonaws.com"
        source_arn = "arn:aws:execute-api:{}:{}:{}/*/*/*".format(
            self.session.region_name, self.session.get_account_id(), api["id"]
        )
        self.add_trigger(statement_id, principle, source_arn)

    def get_api(self):
        """Get API Gateway definition for function

        Returns
        -------
        dict
            See :func:`podpac.managers.aws.get_api`
        """
        if "APIGateway" not in self.function_triggers:
            _log.debug("Skipping API get because 'APIGateway' not in the function triggers")
            return None

        api = get_api(self.session, self.function_api_name, self.function_api_endpoint)
        self._set_api(api)

        return api

    def validate_api(self):
        """
        Validate that API will work with function.

        This should only be run after running `self.get_api()`
        """

        if "APIGateway" not in self.function_triggers:
            _log.debug("Skipping API validation because 'APIGateway' not in the function triggers")
            return True

        # TOOD: implement
        if self._api is None:
            return False

        return True

    def delete_api(self):
        """Delete API Gateway for Function"""

        self.get_api()

        # remove API
        delete_api(self.session, self.function_api_name)

        # reset
        self._api = None
        self._function_api_id = None
        self._function_api_url = None
        self._function_api_resource_id = None

    # Function budget
    def create_budget(self):
        """
        EXPERIMENTAL FEATURE
        Create budget for lambda function based on node hash.
        """
        # skip if no budget provided
        if self.function_budget_amount is None:
            _log.debug("Skipping Budget creation because function budget is not defined")
            return

        _log.warning(
            "Creating an AWS Budget with PODPAC is an experimental feature. Please continue to monitor AWS usage costs seperately."
        )

        budget = create_budget(
            self.session,
            self.function_budget_amount,
            self.function_budget_email,
            budget_name=self.function_budget_name,
            budget_currency=self.function_budget_currency,
            budget_filter_tags={"_podpac_resource_hash": self.hash},
        )

        self._set_budget(budget)

    def get_budget(self):
        """
        EXPERIMENTAL FEATURE
        Get budget definition for function

        Returns
        -------
        dict
            See :func:`podpac.managers.aws.get_budget`
        """

        # skip if no budget provided
        if self.function_budget_amount is None:
            _log.debug("Skipping Budget request because function budget is not defined")
            return None

        budget = get_budget(self.session, self.function_budget_name)
        self._set_budget(budget)

        return budget

    def validate_budget(self):
        """Validate budget definition for function

        Returns
        -------
        dict
            See :func:`podpac.managers.aws.get_budget`
        """

        # skip if no budget provided
        if self.function_budget_amount is None:
            _log.debug("Skipping Budget validation because function budget is not defined")
            return True

        if self._budget is None:
            return False

        return True

    def delete_budget(self):
        """Delete budget associated with function"""
        self.get_budget()

        # delete budget
        delete_budget(self.session, self.function_budget_name)

        # reset class attributes
        self._budget = None

    # Logs
    def get_logs(self, limit=5, start=None, end=None):
        """Get Cloudwatch logs from lambda function execution

        See :func:`podpac.managers.aws.get_logs`

        Parameters
        ----------
        limit : int, optional
            Limit logs to the most recent N logs
        start : str, optional
            Datetime string. Must work as input to np.datetime64 (i.e np.datetime64(start))
            Defaults to 1 hour prior to ``end``.
        end : str, optional
            Datetime string. Must work as input to np.datetime64 (i.e np.datetime64(end))
            Defaults to now.

        Returns
        -------
        list
            list of log events
        """
        if self.function_name is None:
            raise AttributeError("Function name must be defined to get logs")

        log_group_name = "/aws/lambda/{}".format(self.function_name)
        return get_logs(self.session, log_group_name, limit=limit, start=start, end=end)

    # -----------------------------------------------------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------------------------------------------------

    def _set_function(self, function):
        """Set function class members

        Parameters
        ----------
        function : dict
        """
        # update all class members with return
        # this allows a new Lambda instance to fill in class members from input function_name
        if function is not None:
            self.set_trait("function_handler", function["Configuration"]["Handler"])
            self.set_trait("function_description", function["Configuration"]["Description"])
            self.set_trait("function_env_variables", function["Configuration"]["Environment"]["Variables"])
            self.set_trait("function_timeout", function["Configuration"]["Timeout"])
            self.set_trait("function_memory", function["Configuration"]["MemorySize"])
            self.set_trait("function_tags", function["tags"])
            self._function_arn = function["Configuration"]["FunctionArn"]
            self._function_last_modified = function["Configuration"]["LastModified"]
            self._function_version = function["Configuration"]["Version"]
            self._function_code_sha256 = function["Configuration"][
                "CodeSha256"
            ]  # TODO: is this the best way to determine S3 source bucket and dist zip?

            # store a copy of the whole response from AWS
            self._function = function

    def _set_role(self, role):
        """Set role class members

        Parameters
        ----------
        role : dict
        """
        if role is not None:
            self.set_trait("function_role_name", role["RoleName"])
            self.set_trait("function_role_description", role["Description"])
            self.set_trait("function_role_assume_policy_document", role["AssumeRolePolicyDocument"])
            self.set_trait("function_role_policy_arns", role["policy_arns"])
            self.set_trait("function_role_policy_document", role["policy_document"])
            self.set_trait("function_role_tags", role["tags"])
            self._function_role_arn = role["Arn"]

            # store a copy of the whole response from AWS
            self._role = role

    def _set_bucket(self, bucket):
        """Set bucket class members

        Parameters
        ----------
        bucket : dict
        """
        if bucket is not None:
            self.set_trait("function_s3_bucket", bucket["name"])
            self.set_trait("function_s3_tags", bucket["tags"])

            # store a copy of the whole response from AWS
            self._bucket = bucket

    def _set_api(self, api):
        """Set api class members

        Parameters
        ----------
        api : dict
        """
        if api is not None:
            self.set_trait("function_api_name", api["name"])
            self.set_trait("function_api_description", api["description"])
            self.set_trait("function_api_version", api["version"])
            self.set_trait("function_api_tags", api["tags"])
            self._function_api_id = api["id"]

            if "stage" in api and api["stage"] is not None:
                self.set_trait("function_api_stage", api["stage"])

            if "resource" in api and api["resource"] is not None:
                self._function_api_resource_id = api["resource"]["id"]
                self.set_trait("function_api_endpoint", api["resource"]["pathPart"])

            # set api url
            self._function_api_url = self._get_api_url()

            # store a copy of the whole response from AWS
            self._api = api

    def _set_budget(self, budget):
        """Set budget class members

        Parameters
        ----------
        budget : dict
        """
        if budget is not None:
            budget_filter_tags = {"_podpac_resource_hash": self.hash}

            self.set_trait("function_budget_amount", float(budget["BudgetLimit"]["Amount"]))
            self.set_trait("function_budget_name", budget["BudgetName"])
            self.set_trait("function_budget_currency", budget["BudgetLimit"]["Unit"])

            # TODO
            # self.set_trait("function_budget_email", budget["BudgetLimit"]["Amount"]))

            # store a copy of the whole response from AWS
            self._budget = budget

    def _create_eval_pipeline(self, coordinates):
        """shorthand to create pipeline on eval"""

        # add coordinates to the pipeline
        pipeline = self.pipeline  # contains "pipeline" and "output" keys
        pipeline["coordinates"] = json.loads(coordinates.json)

        # TODO: should we move this to `self.pipeline`?
        pipeline["settings"] = self.eval_settings
        pipeline["settings"][
            "FUNCTION_DEPENDENCIES_KEY"
        ] = self.function_s3_dependencies_key  # overwrite in case this is specified explicitly by class
        if self.output_format:
            pipeline["output"] = self.output_format

        return pipeline

    def _eval_invoke(self, coordinates, output=None):
        """eval node through invoke trigger"""

        # create eval pipeline
        pipeline = self._create_eval_pipeline(coordinates)

        # create lambda client
        config = botocore.config.Config(
            read_timeout=self.eval_timeout, max_pool_connections=1001, retries={"max_attempts": 0}
        )
        awslambda = self.session.client("lambda", config=config)

        # pipeline payload
        payload = bytes(json.dumps(pipeline, indent=4, cls=JSONEncoder).encode("UTF-8"))

        if self.download_result:
            _log.debug("Evaluating pipeline via invoke synchronously")
            response = awslambda.invoke(
                FunctionName=self.function_name,
                LogType="Tail",  # include the execution log in the response.
                Payload=payload,
            )
        else:
            # async invocation
            _log.debug("Evaluating pipeline via invoke asynchronously")
            awslambda.invoke(
                FunctionName=self.function_name,
                InvocationType="Event",
                LogType="Tail",  # include the execution log in the response.
                Payload=payload,
            )

            return

        _log.debug("Received response from lambda function")

        if "FunctionError" in response:
            _log.error("Unhandled error from lambda function")
            # logs = base64.b64decode(response["LogResult"]).decode("UTF-8").split('\n')
            payload = json.loads(response["Payload"].read().decode("UTF-8"))
            raise LambdaException(
                "Error in lambda function evaluation:\n\nError Type: {}\nError Message: {}\nStack Trace: {}".format(
                    payload["errorType"], payload["errorMessage"], "\n".join(payload["stackTrace"])
                )
            )

        # After waiting, load the pickle file like this:
        payload = response["Payload"].read()
        try:
            self._output = UnitsDataArray.open(payload)
        except ValueError:
            # Not actually a data-array, returning a string instead
            return payload.decode("utf-8")
        return self._output

    def _eval_s3(self, coordinates, output=None):
        """Evaluate node through s3 trigger"""

        _log.debug("Evaluating pipeline via S3")

        input_folder = "{}{}".format(self.function_s3_input, "/" if not self.function_s3_input.endswith("/") else "")
        output_folder = "{}{}".format(self.function_s3_output, "/" if not self.function_s3_output.endswith("/") else "")

        # create eval pipeline
        pipeline = self._create_eval_pipeline(coordinates)
        pipeline["settings"]["FUNCTION_FORCE_COMPUTE"] = self.force_compute
        pipeline["settings"][
            "FUNCTION_S3_INPUT"
        ] = input_folder  # overwrite in case this is specified explicitly by class
        pipeline["settings"][
            "FUNCTION_S3_OUTPUT"
        ] = output_folder  # overwrite in case this is specified explicitly by class

        # filename
        filename = "{folder}{output}_{source}_{coordinates}.{suffix}".format(
            folder=input_folder,
            output=self.source_output_name,
            source=self.source.hash,
            coordinates=coordinates.hash,
            suffix="json",
        )

        # create s3 client
        s3 = self.session.client("s3")

        # put pipeline into s3 bucket
        s3.put_object(
            Body=(bytes(json.dumps(pipeline, indent=4, cls=JSONEncoder).encode("UTF-8"))),
            Bucket=self.function_s3_bucket,
            Key=filename,
        )

        _log.debug("Successfully put pipeline into S3 bucket")

        # wait for object to exist
        if not self.download_result:
            return

        # TODO: handle the "force_compute" parameter
        waiter = s3.get_waiter("object_exists")
        filename = "{folder}{output}_{source}_{coordinates}.{suffix}".format(
            folder=output_folder,
            output=self.source_output_name,
            source=self.source.hash,
            coordinates=coordinates.hash,
            suffix=self.source_output_format,
        )

        _log.debug("Starting to wait for output")
        waiter.wait(Bucket=self.function_s3_bucket, Key=filename)

        # After waiting, load the pickle file like this:
        _log.debug("Received response from lambda function")
        response = s3.get_object(Key=filename, Bucket=self.function_s3_bucket)
        body = response["Body"].read()
        self._output = UnitsDataArray.open(body)
        return self._output

    def _eval_api(self, coordinates, output=None):
        # TODO: implement and pass in settings in the REST API
        pass

    def _get_api_url(self):
        """Generated API url"""
        if (
            self._function_api_id is not None
            and self.function_api_stage is not None
            and self.function_api_endpoint is not None
        ):
            return "https://{}.execute-api.{}.amazonaws.com/{}/{}".format(
                self._function_api_id, self.session.region_name, self.function_api_stage, self.function_api_endpoint
            )
        else:
            return None

    def __repr__(self):
        rep = "{} {}\n".format(str(self.__class__.__name__), "(staged)" if not self._function_valid else "(built)")
        rep += "\tName: {}\n".format(self.function_name)
        rep += "\tSource: {}\n".format(self.source.__class__.__name__ if self.source is not None else "")
        rep += "\tBucket: {}\n".format(self.function_s3_bucket)
        rep += "\tTriggers: {}\n".format(self.function_triggers)
        rep += "\tRole: {}\n".format(self.function_role_name)

        # Bucket

        # API
        if "APIGateway" in self.function_triggers:
            rep += "\tAPI: {}\n".format(self.function_api_name)
            rep += "\tAPI Url: {}\n".format(self._function_api_url)

        return rep


class Session(boto3.Session):
    """Wrapper for :class:`boto3.Session`
    See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html

    We wrap the Session class to provide access to the podpac settings for the
    aws_access_key_id, aws_secret_access_key, and region_name and to check the credentials
    on session creation.
    """

    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
        aws_access_key_id = settings["AWS_ACCESS_KEY_ID"] if aws_access_key_id is None else aws_access_key_id
        aws_secret_access_key = (
            settings["AWS_SECRET_ACCESS_KEY"] if aws_secret_access_key is None else aws_secret_access_key
        )
        region_name = settings["AWS_REGION_NAME"] if region_name is None else region_name

        super(Session, self).__init__(
            aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name
        )

        try:
            _ = self.get_account_id()
        except botocore.exceptions.ClientError as e:
            _log.error(
                "AWS credential check failed. Confirm aws access key id and aws secred access key are valid. Credential check exception: {}".format(
                    str(e)
                )
            )
            raise ValueError(
                "AWS credential check failed. Confirm aws access key id and aws secred access key are valid."
            )

    def get_account_id(self):
        """Return the account ID assciated with this AWS session. The credentials will determine the account ID.

        Returns
        -------
        str
            account id associated with credentials
        """
        return self.client("sts").get_caller_identity()["Account"]


# -----------------------------------------------------------------------------------------------------------------
# S3
# -----------------------------------------------------------------------------------------------------------------
def create_bucket(session, bucket_name, bucket_region=None, bucket_policy=None, bucket_tags={}):
    """Create S3 bucket

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    bucket_name : str
        Bucket name
    bucket_region : str, optional
        Location constraint for bucket. Defaults to no location constraint
    bucket_policy : dict, optional
        Bucket policy document as dict. For parameters, see https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutBucketPolicy.html#API_PutBucketPolicy_RequestSyntax
    bucket_tags : dict, optional
        Bucket tags

    Returns
    -------
    dict
        See :func:`podpac.managers.aws.get_bucket`

    Raises
    ------
    ValueError
        Description
    """

    bucket = get_bucket(session, bucket_name)

    # TODO: add checks to make sure bucket parameters match
    if bucket is not None:
        _log.debug("S3 bucket '{}' already exists. Using existing bucket.".format(bucket_name))
        return bucket

    if bucket_name is None:
        raise ValueError("`bucket_name` is None in create_bucket")

    # add special podpac tags for billing id
    bucket_tags["_podpac_resource"] = "true"

    # bucket configuration
    bucket_config = {"ACL": "private", "Bucket": bucket_name}
    if bucket_region is not None:
        bucket_config["LocationConstraint"] = bucket_region

    _log.debug("Creating S3 bucket {}".format(bucket_name))
    s3 = session.client("s3")

    # create bucket
    s3.create_bucket(**bucket_config)

    # add tags
    # for some reason the tags API is different here
    tags = []
    for key in bucket_tags.keys():
        tags.append({"Key": key, "Value": bucket_tags[key]})

    s3.put_bucket_tagging(Bucket=bucket_name, Tagging={"TagSet": tags})

    # set bucket policy
    if bucket_policy is not None:
        s3.put_bucket_policy(Bucket=bucket_name, Policy=json.dumps(bucket_policy))

    # get finalized bucket
    bucket = get_bucket(session, bucket_name)
    _log.debug("Successfully created S3 bucket '{}'".format(bucket_name))

    return bucket


def get_object(session, bucket_name, bucket_path):
    """Get an object from an S3 bucket

    See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.get_object

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    bucket_name : str
        Bucket name
    bucket_path : str
        Path to object in bucket
    """

    if bucket_name is None or bucket_path is None:
        return None

    _log.debug("Getting object {} from S3 bucket {}".format(bucket_path, bucket_name))
    s3 = session.client("s3")

    # see if the object exists
    try:
        s3.head_object(Bucket=bucket_name, Key=bucket_path)
    except botocore.exceptions.ClientError:
        return None

    # get the object
    return s3.get_object(Bucket=bucket_name, Key=bucket_path)


def put_object(session, bucket_name, bucket_path, file=None, object_acl="private", object_metadata=None):
    """Simple wrapper to put an object in an S3 bucket

    See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.put_object

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    bucket_name : str
        Bucket name
    bucket_path : str
        Path in bucket to put object
    file : str | bytes, optional
        Path to local object or b'bytes'. If none, this will create a directory in bucket.
    object_acl : str, optional
        Object ACL. Defaults to 'private'
        One of: 'private'|'public-read'|'public-read-write'|'authenticated-read'|'aws-exec-read'|'bucket-owner-read'|'bucket-owner-full-control'
    object_metadata : dict, optional
        Metadata to add to object
    """

    if bucket_name is None or bucket_path is None:
        return None

    _log.debug("Putting object {} into S3 bucket {}".format(bucket_path, bucket_name))
    s3 = session.client("s3")

    object_config = {"ACL": object_acl, "Bucket": bucket_name, "Key": bucket_path}

    object_body = None
    if isinstance(file, string_types):
        with open(file, "rb") as f:
            object_body = f.read()
    else:
        object_body = file

    if object_body is not None:
        object_config["Body"] = object_body

    if object_metadata is not None:
        object_config["Metadata"] = object_metadata

    s3.put_object(**object_config)


def get_bucket(session, bucket_name):
    """Get S3 bucket parameters

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    bucket_name : str
        Bucket name

    Returns
    -------
    dict
        Bucket dict containing keys: "name", region", "policy", "tags"
    """
    if bucket_name is None:
        return None

    _log.debug("Getting S3 bucket {}".format(bucket_name))
    s3 = session.client("s3")

    # see if the bucket exists
    try:
        s3.head_bucket(Bucket=bucket_name)
    except botocore.exceptions.ClientError:
        return None

    # init empty object
    bucket = {"name": bucket_name}

    # TODO: this is usually none, even though the bucket has a region. It could either be a bug
    # in getting the region/LocationConstraint, or just misleading
    # get location constraint. this will be None for no location constraint
    bucket["region"] = s3.get_bucket_location(Bucket=bucket_name)["LocationConstraint"]

    try:
        bucket["policy"] = s3.get_bucket_policy(Bucket=bucket_name)["Policy"]
    except botocore.exceptions.ClientError:
        bucket["policy"] = None

    # reverse tags into dict
    tags = {}
    try:
        tag_set = s3.get_bucket_tagging(Bucket=bucket_name)["TagSet"]
        for tag in tag_set:
            tags[tag["Key"]] = tag["Value"]
    except botocore.exceptions.ClientError:
        pass

    bucket["tags"] = tags

    return bucket


def delete_bucket(session, bucket_name, delete_objects=False):
    """Remove S3 bucket from AWS resources

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    bucket_name : str
        Bucket name to delete
    delete_objects : bool, optional
        Must be set to True if the bucket contains files. This helps avoid deleting buckets inadvertantly
    """
    if bucket_name is None:
        _log.error("`bucket_name` not defined in delete_bucket")
        return

    # make sure bucket exists
    bucket = get_bucket(session, bucket_name)
    if bucket is None:
        _log.debug("S3 bucket '{}' does not exist".format(bucket_name))
        return

    _log.debug("Removing S3 bucket '{}'".format(bucket_name))
    s3 = session.client("s3")

    # need to remove all objects before it can be removed. Only do this if delete_objects is TRue
    if delete_objects:
        s3resource = session.resource("s3")
        bucket = s3resource.Bucket(bucket_name)
        bucket.object_versions.delete()  # delete objects that are versioned
        bucket.objects.all().delete()  # delete objects that are not versioned

    # now delete bucket
    s3.delete_bucket(Bucket=bucket_name)
    _log.debug("Successfully removed S3 bucket '{}'".format(bucket_name))


# -----------------------------------------------------------------------------------------------------------------
# Lambda
# -----------------------------------------------------------------------------------------------------------------
def create_function(
    session,
    function_name,
    function_role_arn,
    function_handler,
    function_description="PODPAC function",
    function_timeout=600,
    function_memory=2048,
    function_env_variables={},
    function_tags={},
    function_source_dist_zip=None,
    function_source_bucket=None,
    function_source_dist_key=None,
):
    """Build Lambda function and associated resources on AWS

    Parameters
    ----------
    session : :class:`Session`
        AWS boto3 Session. See :class:`Session` for creation.
    function_name : str
        Function name
    function_role_arn : str
        Role ARN for the function.
        Generate a role for lambda function execution with :func:`podpac.managers.aws.create_role`.
        The "Arn" key in the output of this function can be used and this input.
    function_handler : str
        Handler module and method (i.e. "module.method")
    function_description : str, optional
        Function description
    function_timeout : int, optional
        Function timeout
    function_memory : int, optional
        Function memory limit
    function_env_variables : dict, optional
        Environment variables for function
    function_tags : dict, optional
        Function tags
    function_source_dist_zip : str, optional
        Path to .zip archive containg the function source.
    function_source_bucket : str
        S3 Bucket containing .zip archive of the function source. If defined, :attr:`function_source_dist_key` must be defined.
    function_source_dist_key : str
        If :attr:`function_source_bucket` is defined, this is the path to the .zip archive of the function source.

    Returns
    -------
    dict
        See :func:`podpac.managers.aws.get_function`
    """

    function = get_function(session, function_name)

    # TODO: add checks to make sure role parameters match
    if function is not None:
        _log.debug("AWS lambda function '{}' already exists. Using existing function.".format(function_name))
        return function

    if function_name is None:
        raise ValueError("`function_name` is None in create_function")

    # add special podpac tags for billing id
    function_tags["_podpac_resource"] = "true"

    _log.debug("Creating lambda function {}".format(function_name))
    awslambda = session.client("lambda")

    lambda_config = {
        "Runtime": "python3.7",
        "FunctionName": function_name,
        "Publish": True,
        "Role": function_role_arn,
        "Handler": function_handler,
        "Code": {},
        "Description": function_description,
        "Timeout": function_timeout,
        "MemorySize": function_memory,
        "Environment": {"Variables": function_env_variables},
        "Tags": function_tags,
    }

    # read function from zip file
    if function_source_dist_zip is not None:
        raise NotImplementedError("Supplying a source dist zip from a local file is not yet supported")
        # TODO: this fails when the file size is over a certain limit
        # with open(function_source_dist_zip, "rb") as f:
        #     lambda_config["Code"]["ZipFile"] = f.read()

    # read function from S3 (Default)
    elif function_source_bucket is not None and function_source_dist_key is not None:
        lambda_config["Code"]["S3Bucket"] = function_source_bucket
        lambda_config["Code"]["S3Key"] = function_source_dist_key

    else:
        raise ValueError("Function source is not defined")

    # create function
    awslambda.create_function(**lambda_config)

    # get function after created
    function = get_function(session, function_name)

    _log.debug("Successfully created lambda function '{}'".format(function_name))
    return function


def get_function(session, function_name):
    """Get function definition from AWS

    Parameters
    ----------
    function_name : str
        Function name

    Returns
    -------
    dict
        Dict returned from Boto3 get_function
        Based on value returned by https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda.html#Lambda.Client.get_function
        Adds "tags" key to list function tags
        Returns None if no function role is found
    """
    if function_name is None:
        return None

    _log.debug("Getting lambda function {}".format(function_name))

    awslambda = session.client("lambda")
    try:
        function = awslambda.get_function(FunctionName=function_name)
        del function["ResponseMetadata"]  # remove response details from function
    except awslambda.exceptions.ResourceNotFoundException as e:
        _log.debug("Failed to get lambda function {} with exception: {}".format(function_name, e))
        return None

    # get tags
    try:
        function["tags"] = awslambda.list_tags(Resource=function["Configuration"]["FunctionArn"])["Tags"]
    except botocore.exceptions.ClientError:
        function["tags"] = {}

    return function


def update_function(
    session, function_name, function_source_dist_zip=None, function_source_bucket=None, function_source_dist_key=None
):
    """Update function on AWS

    Parameters
    ----------
    session : :class:`Session`
        AWS boto3 Session. See :class:`Session` for creation.
    function_name : str
        Function name
    function_source_dist_zip : str, optional
        Path to .zip archive containg the function source.
    function_source_bucket : str
        S3 Bucket containing .zip archive of the function source. If defined, :attr:`function_source_dist_key` must be defined.
    function_source_dist_key : str
        If :attr:`function_source_bucket` is defined, this is the path to the .zip archive of the function source.

    Returns
    -------
    dict
        See :func:`podpac.managers.aws.get_function`
    """
    function = get_function(session, function_name)

    if function is None:
        raise ValueError("AWS lambda function {} does not exist".format(function_name))

    _log.debug("Updating lambda function {} code".format(function_name))
    awslambda = session.client("lambda")

    lambda_config = {"FunctionName": function_name, "Publish": True}

    # read function from S3 (Default)
    if function_source_bucket is not None and function_source_dist_key is not None:
        lambda_config["S3Bucket"] = function_source_bucket
        lambda_config["S3Key"] = function_source_dist_key

    # read function from zip file
    elif function_source_dist_zip is not None:
        raise NotImplementedError("Supplying a source dist zip from a local file is not yet supported")
        # TODO: this fails when the file size is over a certain limit
        # with open(function_source_dist_zip, "rb") as f:
        #     lambda_config["ZipFile"] = f.read()

    else:
        raise ValueError("Function source is not defined")

    # create function
    awslambda.update_function_code(**lambda_config)

    # get function after created
    function = get_function(session, function_name)

    _log.debug("Successfully updated lambda function code '{}'".format(function_name))
    return function


def delete_function(session, function_name):
    """Remove AWS Lambda function

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    function_name : str
        Lambda function name
    """
    if function_name is None:
        _log.error("`function_name` not defined in delete_function")
        return

    # make sure function exists
    function = get_function(session, function_name)
    if function is None:
        _log.debug("Lambda function '{}' does not exist".format(function_name))
        return

    _log.debug("Removing lambda function '{}'".format(function_name))

    awslambda = session.client("lambda")
    awslambda.delete_function(FunctionName=function_name)

    _log.debug("Removed lambda function '{}'".format(function_name))


def add_function_trigger(session, function_name, statement_id, principle, source_arn):
    """Add trigger (permission) to lambda function

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    function_name : str
        Function name
    statement_id : str
        Specific identifier for trigger
    principle : str
        Principle identifier from AWS
    source_arn : str
        Source ARN for trigger
    """
    if function_name is None or statement_id is None or principle is None or source_arn is None:
        raise ValueError(
            "`function_name`, `statement_id`, `principle`, and `source_arn` are required to add function trigger"
        )

    awslambda = session.client("lambda")
    awslambda.add_permission(
        FunctionName=function_name,
        StatementId=statement_id,
        Action="lambda:InvokeFunction",
        Principal=principle,
        SourceArn=source_arn,
    )


def remove_function_trigger(session, function_name, statement_id):
    """Remove trigger (permission) from lambda function

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    function_name : str
        Function name
    statement_id : str
        Specific identifier for trigger
    """
    if function_name is None or statement_id is None:
        _log.error("`api_id` or `statement_id` not defined in remove_function_trigger")
        return

    awslambda = session.client("lambda")
    try:
        awslambda.remove_permission(FunctionName=function_name, StatementId=statement_id)
    except awslambda.exceptions.ResourceNotFoundException:
        _log.warning("Failed to remove permission {} on function {}".format(statement_id, function_name))


# -----------------------------------------------------------------------------------------------------------------
# IAM Roles
# -----------------------------------------------------------------------------------------------------------------


def create_role(
    session,
    role_name,
    role_description="PODPAC Role",
    role_policy_document=None,
    role_policy_arns=[],
    role_assume_policy_document=None,
    role_tags={},
):
    """Create IAM role

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    role_name : str
        Role name to create
    role_description : str, optional
        Role description
    role_policy_document : dict, optional
        Role policy document allowing role access to AWS resources
        See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/iam.html#IAM.Client.put_role
    role_policy_arns : list, optional
        List of role policy ARN's to attach to role
    role_assume_policy_document : None, optional
        Role policy document.
        Defaults to trust policy allowing role to execute lambda functions.
        See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/iam.html#IAM.Client.create_role
    role_tags : dict, optional
        Role tags


    Returns
    -------
    dict
        See :func:`podpac.managers.aws.get_role`
    """

    role = get_role(session, role_name)

    # TODO: add checks to make sure role parameters match
    if role is not None:
        _log.debug("IAM role '{}' already exists. Using existing role.".format(role_name))
        return role

    if role_name is None:
        raise ValueError("`role_name` is None in create_role")

    # default role_assume_policy_document is lambda
    if role_assume_policy_document is None:
        role_assume_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}
            ],
        }

    # add special podpac tags for billing id
    role_tags["_podpac_resource"] = "true"

    _log.debug("Creating IAM role {}".format(role_name))
    iam = session.client("iam")

    iam_config = {
        "RoleName": role_name,
        "Description": role_description,
        "AssumeRolePolicyDocument": json.dumps(role_assume_policy_document),
    }

    # for some reason the tags API is different here
    tags = []
    for key in role_tags.keys():
        tags.append({"Key": key, "Value": role_tags[key]})
    iam_config["Tags"] = tags

    # create role
    iam.create_role(**iam_config)

    # add role policy document
    if role_policy_document is not None:
        policy_name = "{}-policy".format(role_name)
        iam.put_role_policy(RoleName=role_name, PolicyName=policy_name, PolicyDocument=json.dumps(role_policy_document))

    # attached role polcy ARNs
    for policy in role_policy_arns:
        iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)

    # get finalized role
    role = get_role(session, role_name)
    _log.debug("Successfully created IAM role '{}'".format(role_name))

    return role


def get_role(session, role_name):
    """Get role definition from AWS

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    role_name : str
        Role name

    Returns
    -------
    dict
        Dict returned from AWS defining role.
        Based on the 'Role' key in https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/iam.html#IAM.Client.get_role
        Adds "policy_document" key to show inline policy document.
        Adds "policy_arns" key to list attached policies.
        Adds "tags" key to list function tags
        Returns None if no role is found
    """
    if role_name is None:
        return None

    _log.debug("Getting IAM role with name {}".format(role_name))
    iam = session.client("iam")

    try:
        response = iam.get_role(RoleName=role_name)
        role = response["Role"]
    except iam.exceptions.NoSuchEntityException as e:
        _log.debug("Failed to get IAM role for name {} with exception: {}".format(role_name, e))
        return None

    # get inline policies
    try:
        policy_name = "{}-policy".format(role_name)
        response = iam.get_role_policy(RoleName=role_name, PolicyName=policy_name)
        role["policy_document"] = response["PolicyDocument"]
    except botocore.exceptions.ClientError:
        role["policy_document"] = None

    # get attached policies
    try:
        response = iam.list_attached_role_policies(RoleName=role_name)
        role["policy_arns"] = [policy["PolicyArn"] for policy in response["AttachedPolicies"]]
    except botocore.exceptions.ClientError:
        role["policy_arns"] = []

    # get tags - reverse tags into dict
    tags = {}
    try:
        response = iam.list_role_tags(RoleName=role_name)
        for tag in response["Tags"]:
            tags[tag["Key"]] = tag["Value"]
    except botocore.exceptions.ClientError:
        pass

    role["tags"] = tags

    return role


def get_role_name(session, role_arn):
    """
    Get function role name based on role_arn

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    role_arn : str
        Role arn

    Returns
    -------
    str
        Role name.
        Returns None if no role name is found for role arn.
    """
    if role_arn is None:
        return None

    iam = session.client("iam")
    roles = iam.list_roles()
    role = [role for role in roles["Roles"] if role["Arn"] == role_arn]
    role_name = role[0] if len(role) else None
    return role_name


def delete_role(session, role_name):
    """Remove role from AWS resources

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    role_name : str
        Role name to delete
    """
    if role_name is None:
        _log.error("`role_name` not defined in delete_role")
        return

    # make sure function exists
    role = get_role(session, role_name)
    if role is None:
        _log.debug("IAM role '{}' does not exist".format(role_name))
        return

    _log.debug("Removing IAM role '{}'".format(role_name))
    iam = session.client("iam")

    # need to remove inline policies first, if they exist
    try:
        policy_name = "{}-policy".format(role_name)
        iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
    except botocore.exceptions.ClientError:
        pass

    # need to detach policies first
    response = iam.list_attached_role_policies(RoleName=role_name)
    for policy in response["AttachedPolicies"]:
        iam.detach_role_policy(RoleName=role_name, PolicyArn=policy["PolicyArn"])

    iam.delete_role(RoleName=role_name)
    _log.debug("Successfully removed IAM role '{}'".format(role_name))


# -----------------------------------------------------------------------------------------------------------------
# API Gateway
# -----------------------------------------------------------------------------------------------------------------


def create_api(
    session, api_name="podpac-api", api_description="PODPAC API", api_version=None, api_tags={}, api_endpoint="eval"
):
    """Create API Gateway REST API

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    api_name : str
        API Name
    api_description : str, optional
        API Description. Defaults to "PODPAC API"
    api_version : str, optional
        API Version. Defaults to PODPAC version.
    api_tags : dict, optional
        API tags. Defaults to {}.
    api_endpoint : str, optional
        API endpoint. Defaults to "eval".

    Returns
    -------
    dict
        See :func:`podpac.managers.aws.get_api`
    """

    # set version to podpac version, if None
    api = get_api(session, api_name, api_endpoint)

    # TODO: add checks to make sure api parameters match
    if api is not None and ("resource" in api and api["resource"] is not None):
        _log.debug(
            "API '{}' and API resource {} already exist. Using existing API ID and resource.".format(
                api_name, api_endpoint
            )
        )
        return api

    # add special podpac tags for billing id
    api_tags["_podpac_resource"] = "true"

    apigateway = session.client("apigateway")

    if api is None:
        _log.debug("Creating API gateway with name {}".format(api_name))

        # set version default
        if api_version is None:
            api_version = version.semver()

        api = apigateway.create_rest_api(
            name=api_name,
            description=api_description,
            version=api_version,
            binaryMediaTypes=["*/*"],
            apiKeySource="HEADER",
            endpointConfiguration={"types": ["REGIONAL"]},
            tags=api_tags,
        )

    # create resource
    _log.debug("Creating API endpoint {} for API ID {}".format(api_endpoint, api["id"]))

    # get resources to get access to parentId ("/" path)
    resources = apigateway.get_resources(restApiId=api["id"])
    parent_id = resources["items"][0]["id"]  # TODO - make this based on path == "/" ?

    # create resource
    resource = apigateway.create_resource(restApiId=api["id"], parentId=parent_id, pathPart=api_endpoint)

    # put method for resource
    apigateway.put_method(
        restApiId=api["id"],
        resourceId=resource["id"],
        httpMethod="ANY",
        authorizationType="NONE",  # TODO: support "AWS_IAM"
        apiKeyRequired=False,  # TODO: create "generate_key()" method
    )

    # save resource on api
    api["resource"] = resource

    return api


def get_api(session, api_name, api_endpoint=None):
    """Get API Gateway definition

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    api_name : str
        API Name
    api_endpoint : str, optional
        API endpoint path, defaults to returning the first endpoint it finds

    Returns
    -------
    dict
        (Returns output of Boto3 API Gateway creation
        Equivalent to https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigateway.html#APIGateway.Client.create_rest_api
        Contains extra key "resource" with output of of Boto3 API Resource. Set to None if API Resource ID is not found)
        Returns None if API Id is not found
    """
    if api_name is None:
        return None

    _log.debug("Getting API Gateway with name {}".format(api_name))
    apigateway = session.client("apigateway")

    try:
        response = apigateway.get_rest_apis(limit=200)
        apis = [api for api in response["items"] if api["name"] == api_name]
        api_id = apis[0]["id"] if len(apis) else None

        if api_id is None:
            return None

        api = apigateway.get_rest_api(restApiId=api_id)
        del api["ResponseMetadata"]
    except (botocore.exceptions.ParamValidationError, apigateway.exceptions.NotFoundException) as e:
        _log.error("Failed to get API Gateway with name {} with exception: {}".format(api_name, e))
        return None

    # try to get stage
    try:
        response = apigateway.get_stages(restApiId=api["id"])
        api["stage"] = response["item"][0]["stageName"] if len(response["item"]) else None
    except Exception:  # TODO: make this more specific?
        pass

    # get resources
    resources = apigateway.get_resources(restApiId=api["id"])
    if api_endpoint is not None:
        resources_filtered = [r for r in resources["items"] if api_endpoint in r["path"]]
    else:
        resources_filtered = [
            r for r in resources["items"] if "pathPart" in r
        ]  # filter resources by ones with a "pathPart"

    # grab the first one, if it exists
    resource = resources_filtered[0] if len(resources_filtered) else None

    # save resource on api
    api["resource"] = resource

    return api


def deploy_api(session, api_id, api_stage):
    """Deploy API gateway definition

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    api_id : str
        API ID. Generated during API creation or returned from :func:`podpac.manager.aws.get_api`
    api_stage : str
        API Stage
    """
    if api_id is None or api_stage is None:
        raise ValueError("`api_id` and `api_stage` must be defined to deploy API")

    _log.debug("Deploying API Gateway with ID {} and stage {}".format(api_id, api_stage))

    apigateway = session.client("apigateway")
    apigateway.create_deployment(
        restApiId=api_id, stageName=api_stage, stageDescription="Deployment of PODPAC API", description="PODPAC API"
    )
    _log.debug("Deployed API Gateway with ID {} and stage {}".format(api_id, api_stage))


def delete_api(session, api_name):
    """Delete API Gateway API

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    api_id : str
        API ID. Generated during API Creation.
    """
    if api_name is None:
        _log.error("`api_id` not defined in delete_api")
        return

    # make sure api exists
    api = get_api(session, api_name, None)
    if api is None:
        _log.debug("API Gateway '{}' does not exist".format(api_name))
        return

    _log.debug("Removing API Gateway with ID {}".format(api["id"]))

    apigateway = session.client("apigateway")
    apigateway.delete_rest_api(restApiId=api["id"])

    _log.debug("Successfully removed API Gateway with ID {}".format(api["id"]))


# -----------------------------------------------------------------------------------------------------------------
# Budget
# -----------------------------------------------------------------------------------------------------------------

# Budget
def create_budget(
    session,
    budget_amount,
    budget_email=None,
    budget_name="podpac-resource-budget",
    budget_currency="USD",
    budget_threshold=80.0,
    budget_filter_tags={"_podpac_resource": "true"},
):
    """
    EXPERIMENTAL FEATURE
    Create a budget for podpac AWS resources based on tags.
    By default, this creates a budget for all podpac created resources with the tag: {"_podpac_resource": "true"}

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    budget_amount : int
        Budget amount
    budget_email : str, optional
        Notification e-mail for budget alerts.
        If no e-mail is provided, the budget must be monitored through the AWS interface.
    budget_name : str, optional
        Budget name
    budget_currency : str, optional
        Currency unit for budget. Defaults to "USD".
        See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/budgets.html#Budgets.Client.describe_budget
        for Unit types.
    budget_threshold : float, optional
        Percent of the budget at which an e-mail notification is sent.
    budget_filter_tags : dict, optional
        Create budget for specific set of resource tags.
        By default, the budget is created for all podpac created resources.

    Returns
    -------
    dict
        Returns Boto3 budget description
        Equivalent to https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/budgets.html#Budgets.Client.describe_budget
    """

    # see if budget already exists
    budget = get_budget(session, budget_name)

    if budget is not None:
        _log.debug("Budget '{}' already exists. Using existing budget".format(budget_name))
        return budget

    # format tags - this is a strange syntax discovered through inspection of boto3
    filter_tags = ["user:{}${}".format(k, v) for (k, v) in budget_filter_tags.items()]

    budgets = session.client("budgets")

    # budget definition
    budget_definition = {
        "AccountId": session.get_account_id(),
        "Budget": {
            "BudgetName": budget_name,
            "BudgetLimit": {"Amount": str(budget_amount), "Unit": budget_currency},
            "CostFilters": {"TagKeyValue": filter_tags},
            "CostTypes": {
                "IncludeTax": True,
                "IncludeSubscription": True,
                "UseBlended": False,
                "IncludeRefund": False,
                "IncludeCredit": False,
                "IncludeUpfront": True,
                "IncludeRecurring": True,
                "IncludeOtherSubscription": True,
                "IncludeSupport": True,
                "IncludeDiscount": True,
                "UseAmortized": False,
            },
            "TimeUnit": "MONTHLY",  # only support monthly for now
            "BudgetType": "COST",
        },
    }

    # handle e-mail and notifications
    if budget_email is None:
        _log.warning(
            "No notification e-mail provided for AWS Budget. This budget must be monitored through the AWS interface."
        )
    else:
        budget_definition["NotificationsWithSubscribers"] = [
            {
                "Notification": {
                    "NotificationType": "ACTUAL",
                    "ComparisonOperator": "GREATER_THAN",
                    "Threshold": budget_threshold,
                    "ThresholdType": "PERCENTAGE",
                    "NotificationState": "ALARM",
                },
                "Subscribers": [{"SubscriptionType": "EMAIL", "Address": budget_email}],
            }
        ]

    # create budget
    budgets.create_budget(**budget_definition)

    # alert the user that they must activate tags
    print(
        "To finalize budget creation, you must visit https://console.aws.amazon.com/billing/home#/preferences/tags and 'Activate' the following User Defined Cost Allocation tags: {}.\nBudget tracking will not work if these User Defined Cost Allocation tags are not active.\nBudget creation and usage updates may take 24 hours to take effect.".format(
            list(budget_filter_tags.keys())
        )
    )

    # get finalized budget
    budget = get_budget(session, budget_name)
    _log.debug("Successfully created budget '{}'".format(budget_name))

    return budget


def get_budget(session, budget_name):
    """
    EXPERIMENTAL FEATURE
    Get a budget by name

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    budget_name : str
        Budget name

    Returns
    -------
    dict
        Returns Boto3 budget description
        Equivalent to https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/budgets.html#Budgets.Client.describe_budget
        Returns None if budget is not found.
    """

    _log.debug("Getting budget with name {}".format(budget_name))
    budgets = session.client("budgets")

    try:
        response = budgets.describe_budget(AccountId=session.get_account_id(), BudgetName=budget_name)
        budget = response["Budget"]
    except (botocore.exceptions.ParamValidationError, budgets.exceptions.NotFoundException) as e:
        _log.error("Failed to get budget with name {} with exception: {}".format(budget_name, e))
        return None

    return budget


def delete_budget(session, budget_name):
    """Delete a budget by name

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    budget_name : str
        Budget name

    Returns
    -------
    None
    """

    budgets = session.client("budgets")

    try:
        budgets.delete_budget(AccountId=session.get_account_id(), BudgetName=budget_name)
    except budgets.exceptions.NotFoundException as e:
        pass

    _log.debug("Successfully removed budget with name '{}'".format(budget_name))


# -----------------------------------------------------------------------------------------------------------------
# Cloudwatch Logs
# -----------------------------------------------------------------------------------------------------------------
def get_logs(session, log_group_name, limit=100, start=None, end=None):
    """Get logs from cloudwatch from specific log groups

    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    log_group_name : str
        Log group name
    limit : int, optional
        Limit logs to the most recent N logs
    start : str, optional
        Datetime string. Must work as input to np.datetime64 (i.e np.datetime64(start))
        Defaults to 1 hour prior to ``end``.
    end : str, optional
        Datetime string. Must work as input to np.datetime64 (i.e np.datetime64(end))
        Defaults to now.

    Returns
    -------
    list
        list of log events
    """

    # default is now
    if end is None:
        end = np.datetime64("now")
    else:
        end = np.datetime64(end)

    # default is 1 hour prior to now
    if start is None:
        start = end - np.timedelta64(1, "h")
    else:
        start = np.datetime64(start)

    # convert to float and add precision for comparison with AWS response
    start = start.astype(float) * 1000
    end = end.astype(float) * 1000

    # get client
    cloudwatchlogs = session.client("logs")  # cloudwatch logs

    try:
        log_streams = cloudwatchlogs.describe_log_streams(
            logGroupName=log_group_name, orderBy="LastEventTime", descending=True
        )
    except cloudwatchlogs.exceptions.ResourceNotFoundException:
        _log.debug("No log streams found for log group name: {}".format(log_group_name))
        return []

    streams = [
        s for s in log_streams["logStreams"] if (s["firstEventTimestamp"] < end and s["lastEventTimestamp"] > start)
    ]
    logs = []
    for stream in streams:
        response = cloudwatchlogs.get_log_events(
            logGroupName=log_group_name,
            logStreamName=stream["logStreamName"],
            startTime=int(start),
            endTime=int(end) + 1000,
            limit=limit,
        )
        logs += response["events"]

    # sort logs
    logs.sort(key=lambda k: k["timestamp"])

    # take only the last "limit"
    logs = logs[-limit:]

    # add time easier to read
    for log in logs:
        log["time"] = "{}.{:03d}".format(
            datetime.fromtimestamp(log["timestamp"] / 1000).strftime("%Y-%m-%d %H:%M:%S"), log["timestamp"] % 1000
        )

    return logs
