"""
Lambda is `Node` manager, which executes the given `Node` on an AWS Lambda
function.
"""
import json
from collections import OrderedDict
import logging
import time

import boto3
import botocore
import traitlets as tl
import numpy as np

from podpac.core.settings import settings
from podpac.core.node import COMMON_NODE_DOC, Node
from podpac.core.utils import common_doc, JSONEncoder
from podpac import version

# Set up logging
_log = logging.getLogger(__name__)

try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

COMMON_DOC = COMMON_NODE_DOC.copy()


class Lambda(Node):
    """A `Node` wrapper to evaluate source on AWS Lambda function
    
    Attributes
    ----------
    attrs : dict
        additional attributes passed on to the Lambda definition of the base node
    aws_access_key_id : string
        access key id from AWS credentials
    aws_region_name : string
        name of the AWS region
    aws_secret_access_key : string
        access key value from AWS credentials
    download_result : Bool
        flag that indicated whether node should wait to download the data.
    function_name : string
        name of the lambda function to use or create
    function_s3_bucket : string
        s3 bucket name to use with lambda function
    function_s3_input : TYPE
        folder in `function_s3_bucket` to store pipelines
    function_s3_output : TYPE
        folder in `function_s3_bucket` to watch for output
    source : :class:`podpac.Node`
        node to be evaluated
    source_output_format : str
        output format for the evaluated results of `source`
    source_output_name : str
        output name for the evaluated results of `source`
    """

    # aws parameters - defaults are handled in Session
    aws_access_key_id = tl.Unicode(default_value=None, allow_none=True)
    aws_secret_access_key = tl.Unicode(default_value=None, allow_none=True)
    aws_region_name = tl.Unicode(default_value=None, allow_none=True)
    session = tl.Instance(boto3.Session)

    @tl.default("session")
    def _session_default(self):
        return Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region_name,
        )

    # general function parameters
    function_eval_trigger = tl.Enum(["S3", "APIGateway"], default_value="S3")

    # lambda function parameters
    function_name = tl.Unicode(default_value="podpac-lambda-autogen")
    function_handler = tl.Unicode(default_value="handler.handler")
    function_description = tl.Unicode(default_value="PODPAC Lambda Function (https://podpac.org)")
    function_env_variables = tl.Dict(default_value={})  # environment vars in function
    function_tags = tl.Dict(default_value={})  # key: value for tags on function (and any created roles)
    function_timeout = tl.Int(default_value=600)
    function_memory = tl.Int(default_value=2048)
    function_zip_file = tl.Unicode(default_value=None, allow_none=True)  # override zip archive with local file
    function_source_bucket = tl.Unicode(default_value="podpac-dist")
    function_source_key = tl.Unicode(default_value="{}/podpac_dist.zip".format(version.semver()))

    # role parameters
    function_role_name = tl.Unicode(default_value="podpac-lambda-autogen")
    function_role_description = tl.Unicode(default_value="PODPAC Lambda Role")
    function_role_policies = tl.List(default_value=["arn:aws:iam::aws:policy/AWSLambdaExecute"])
    function_role_tags = tl.Dict()  # see default below

    @tl.default("function_role_tags")
    def _function_role_tags_default(self):
        return self.function_tags

    # s3 parameters
    function_s3_bucket = tl.Unicode()  # see default below
    function_s3_input = tl.Unicode()  # see default below
    function_s3_output = tl.Unicode()  # see default below

    @tl.default("function_s3_bucket")
    def _function_s3_bucket_default(self):
        return settings["S3_BUCKET_NAME"] or "podpac-autogen"

    @tl.default("function_s3_input")
    def _function_s3_input_default(self):
        return settings["S3_JSON_FOLDER"] or "input"

    @tl.default("function_s3_output")
    def _function_s3_output_default(self):
        return settings["S3_OUTPUT_FOLDER"] or "output"

    # api gateway parameters
    function_api_id = tl.Unicode(default_value=None, allow_none=True)  # will create api if None
    function_api_name = tl.Unicode()  # see default below
    function_api_description = tl.Unicode()  # see default below
    function_api_version = tl.Unicode(default_value="{}".format(version.semver()))
    function_api_tags = tl.Dict()  # see default below
    function_api_stage = tl.Unicode(default_value="prod")
    function_api_endpoint = tl.Unicode(default_value="eval")

    @tl.default("function_api_name")
    def _function_api_name_default(self):
        return "{}-api".format(self.function_name)

    @tl.default("function_api_description")
    def _function_api_description_default(self):
        return "PODPAC Lambda REST API for {} function".format(self.function_name)

    @tl.default("function_api_tags")
    def _function_api_tags_default(self):
        return self.function_tags

    # readonly properties
    function_arn = tl.Unicode(default_value=None, allow_none=True)
    function_role_arn = tl.Unicode(default_value=None, allow_none=True)
    function_last_modified = tl.Unicode(default_value=None, allow_none=True)
    function_version = tl.Unicode(default_value=None, allow_none=True)
    function_code_sha256 = tl.Unicode(default_value=None, allow_none=True)
    function_api_url = tl.Unicode(default_value=None, allow_none=True)
    function_resource_id = tl.Unicode(default_value=None, allow_none=True)
    function_triggers = tl.Dict(default_value={})

    # podpac node parameters
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

        if self.function_eval_trigger == "S3":
            return self._eval_s3(coordinates, output=None)
        else:
            raise NotImplementedError("APIGateway trigger not yet implemented through eval")

    def create_function(self, recreate=False):
        """Build Lambda function and associated resources on AWS
        
        Parameters
        ----------
        recreate : bool, optional
            If function already exists on AWS, remove function and re-add
        """

        function = self.get_function()  # gets default self.function_name
        if function is not None and recreate:
            self.delete_function()  # TODO: autoremove?

        # create default role if it doesn't exist
        role = self.get_role()  # gets default self.function_role_name
        if role is None:
            self.create_role()

            # after creating a role, you need to wait ~10 seconds before its active and will work with the lambda function
            # this is not cool
            time.sleep(10)

        function = create_function(
            self.session,
            self.function_name,
            self.function_role_arn,
            self.function_handler,
            self.function_description,
            self.function_timeout,
            self.function_memory,
            self.function_env_variables,
            self.function_tags,
            self.function_zip_file,
            self.function_source_bucket,
            self.function_source_key,
        )

        # create API gateway
        self.create_api()

        # create S3 bucket
        self.create_s3()

        # set function parameters to class
        self._set_function(function)

    def get_function(self):
        """Get function definition from AWS
            
        Returns
        -------
        dict
            See :func:`podpac.managers.aws.get_function`
        """
        return get_function(self.session, self.function_name)

    def delete_function(self, autoremove=False):
        """Remove AWS Lambda function and associated resources on AWS
        
        Parameters
        ----------
        autoremove : bool, optional
            Remove all associated AWS resources. Defaults to False.
        """
        delete_function(self.session, self.function_name)

        # reset members
        self.function_arn = None
        self.function_last_modified = None
        self.function_version = None
        self.function_code_sha256 = None

        # remove all dependent resources
        if autoremove:
            self.delete_role()
            self.delete_api()
            self.remove_triggers()

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
        add_trigger(self.session, self.function_name, statement_id, principle, source_arn)
        self.function_triggers[statement_id] = source_arn

    def remove_trigger(self, statement_id):
        """Remove trigger (permission) from lambda function
        
        Parameters
        ----------
        statement_id : str
            Specific identifier for trigger
        """

        remove_function_trigger(self.session, self.function_name, statement_id)

        # remove from local dict
        del self.function_triggers[statement_id]

    def remove_triggers(self):
        """
        Remove all triggers from function
        """
        for trigger in self.function_triggers:
            self.remove_trigger(trigger)

    # IAM Roles
    def create_role(self):
        """Create IAM role to execute podpac lambda function
        
        Returns
        -------
        dict
            See :func:`podpac.managers.aws.get_role`
        """

        # enable role to be run by lambda - this document is defined by AWS
        lambda_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}
            ],
        }

        role = create_role(
            self.session,
            self.function_role_name,
            self.function_role_description,
            lambda_policy_document,
            self.function_role_policies,
            self.function_role_tags,
        )

        if role is not None:
            self.function_role_arn = role["Arn"]

    def get_role(self):
        """Get role definition from AWS
        
        See :attr:`self.function_role_name` for role_name
        
        Returns
        -------
        dict
            See :func:`podpac.managers.aws.get_role`
        """
        return get_role(self.session, self.function_role_name)

    def delete_role(self):
        """Remove role from AWS resources

        See :attr:`self.function_role_name` for role_name
        """
        if self.function_role_name is None:
            _log.debug("No role name defined for this function")
            return

        delete_role(self.session, self.function_role_name)

        # reset members
        self.function_role_arn = None

    # S3 Creation
    # TODO: LAST PIECE
    def create_s3(self):
        pass

    def get_s3(self):
        pass

    def delete_s3(self):
        pass

    # API Gateway
    def create_api(self):
        """Create API Gateway API for lambda function
        """
        if self.function_name is None:
            raise ValueError("Function name must be defined when creating API Gateway")

        function = self.get_function()
        function_arn = function["Configuration"]["FunctionArn"]

        # create api and resource
        api, resource = create_api(
            self.session,
            self.function_api_id,
            self.function_api_name,
            self.function_api_description,
            self.function_api_version,
            self.function_api_tags,
            self.function_api_endpoint,
        )

        # set class values
        if api is not None:
            self.function_api_id = api["id"]

        if resource is not None:
            self.function_resource_id = resource["id"]

        # lambda proxy integration - this feels pretty brittle due to uri
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigateway.html#APIGateway.Client.put_integration
        aws_lambda_uri = "arn:aws:apigateway:{}:lambda:path/2015-03-31/functions".format(self.session.region_name)
        uri = "{}/{}/invocations".format(aws_lambda_uri, function_arn)

        try:
            apigateway = self.session.client("apigateway")
            apigateway.put_integration(
                restApiId=api["id"],
                resourceId=resource["id"],
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
                resourceId=resource["id"],
                httpMethod="ANY",
                statusCode="200",
                selectionPattern="",  # bug, see https://github.com/aws/aws-sdk-ruby/issues/1080
            )

            # deploy the api. this has to happen after creating the integration
            deploy_api(self.session, self.function_api_id, self.function_api_stage)
            self.function_api_url = self._get_api_url(api, self.function_api_stage, self.function_api_endpoint)

            # add permission to invoke call lambda - this feels brittle due to source_arn
            statement_id = api["id"]
            principle = "apigateway.amazonaws.com"
            source_arn = "arn:aws:execute-api:{}:{}:{}/*/*/*".format(
                self.session.region_name, self.session.get_account_id(), api["id"]
            )
            self.add_trigger(statement_id, principle, source_arn)

        # clean up if build API function fails
        except Exception as e:
            if api is not None:
                try:
                    self.delete_api()
                except:
                    pass

            _log.error("Failed to build API gateway with exception: {}".format(e))

    def get_api(self):
        """Get API Gateway definition for function
        
        Returns
        -------
        dict
            See :func:`podpac.managers.aws.get_api`
        """
        if self.function_api_id is None:
            _log.debug("No API ID defined for function")
            return None

        return get_api(self.session, self.function_api_id)

    def delete_api(self):
        """Delete API Gateway for Function"""
        if self.function_api_id is None:
            _log.debug("No API ID defined for function")
            return

        # remove API
        delete_api(self.session, self.function_api_id)

        # reset
        self.function_api_id = None
        self.function_api_url = None
        self.function_resource_id = None

    # -----------------------------------------------------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------------------------------------------------

    def _eval_s3(self, coordinates, output=None):
        """Evaluate node through s3 trigger"""

        # add coordinates to the pipeline
        pipeline = self.pipeline
        pipeline["coordinates"] = json.loads(coordinates.json)

        # filename
        filename = "{folder}{slash}{output}_{source}_{coordinates}.{suffix}".format(
            folder=self.function_s3_input,
            slash="/" if not self.function_s3_input.endswith("/") else "",
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

        # wait for object to exist
        if not self.download_result:
            return

        waiter = s3.get_waiter("object_exists")
        filename = "{folder}{slash}{output}_{source}_{coordinates}.{suffix}".format(
            folder=self.function_s3_output,
            slash="/" if not self.function_s3_output.endswith("/") else "",
            output=self.source_output_name,
            source=self.source.hash,
            coordinates=coordinates.hash,
            suffix=self.source_output_format,
        )
        waiter.wait(Bucket=self.function_s3_bucket, Key=filename)

        # After waiting, load the pickle file like this:
        response = s3.get_object(Key=filename, Bucket=self.function_s3_bucket)
        body = response["Body"].read()
        self._output = cPickle.loads(body)
        return self._output

    def _eval_api(self, coordinates, output=None):
        # TODO: implement
        pass

    def _get_api_url(self, api, api_stage, api_endpoint):
        """Generated API url
        
        Parameters
        ----------
        api : dict
            API dict returned from :meth:`self.get_api`
        api_stage : str
            API Stage
        api_endpoint : str
            API resource path
        """
        return "https://{}.execute-api.{}.amazonaws.com/{}/{}".format(
            api["id"], self.session.region_name, api_stage, api_endpoint
        )

    def _set_function(self, function):
        """Set function parameters based on function response from AWS
        
        Parameters
        ----------
        function : dict
            dict returned from AWS defining role.
            Equivalent to value returned in the 'Role' key in https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/iam.html#IAM.Client.get_role 
        """
        self.function_role_name = function["Configuration"]["Role"].split("/")[-1]  # get the last part of the role
        self.function_handler = function["Configuration"]["Handler"]
        self.function_description = function["Configuration"]["Description"]
        self.function_env_variables = function["Configuration"]["Environment"]["Variables"]
        self.function_timeout = function["Configuration"]["Timeout"]
        self.function_memory = function["Configuration"]["MemorySize"]

        # properties
        self.function_arn = function["Configuration"]["FunctionArn"]
        self.function_last_modified = function["Configuration"]["LastModified"]
        self.function_version = function["Configuration"]["Version"]
        self.function_code_sha256 = function["Configuration"][
            "CodeSha256"
        ]  # TODO: is this the best way to determine S3 source bucket and dist zip?

        awslambda = self.session.client("lambda")
        self.function_tags = awslambda.list_tags(Resource=self.function_arn)["Tags"]

    def __repr__(self):
        rep = "{}\n".format(str(self.__class__.__name__))
        rep += "\tName: {}\n".format(self.function_name)

        if self.function_api_url is not None:
            rep += "\tAPI Url: {}\n".format(self.function_api_url)
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
        Description
    
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


def put_object(session, bucket_name, bucket_path, file, object_acl="private", object_metadata=None):
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
    file : str | bytes
        Path to local object or b'bytes'
    object_acl : str, optional
        Object ACL.
        One of: 'private'|'public-read'|'public-read-write'|'authenticated-read'|'aws-exec-read'|'bucket-owner-read'|'bucket-owner-full-control'
    object_metadata : dict, optional
        Metadata to add to object
    """

    if bucket_name is None or bucket_path is None or file is None:
        return None

    _log.debug("Putting object {} into S3 bucket {}".format(bucket_path, bucket_name))
    s3 = session.client("s3")

    if isinstance(file, str):
        with open(file, "rb") as f:
            object_body = f.read()
    else:
        object_body = file

    object_config = {"ACL": object_acl, "Bucket": bucket_name, "Body": object_body, "Key": bucket_path}

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
    function_zip_file=None,
    function_source_bucket=None,
    function_source_key=None,
):
    """Build Lambda function and associated resources on AWS
    
    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
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
    function_zip_file : str, optional
        Path to .zip archive containg the function source.
    function_source_bucket : str
        S3 Bucket containing .zip archive of the function source. If defined, :attr:`function_source_key` must be defined.
    function_source_key : TYPE
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

    _log.debug("Creating lambda function {}".format(function_name))
    awslambda = session.client("lambda")

    # read function from S3 (Default)
    if function_source_bucket is not None:
        code = {"S3Bucket": function_source_bucket, "S3Key": function_source_key}

    # read function from zip file
    if function_zip_file is not None:
        with open(function_zip_file, "rb") as f:
            code = {"ZipFile": f.read()}

    # create function
    awslambda.create_function(
        FunctionName=function_name,
        Runtime="python3.7",
        Role=function_role_arn,
        Handler=function_handler,
        Code=code,
        Description=function_description,
        Timeout=function_timeout,
        MemorySize=function_memory,
        Publish=True,
        Environment={"Variables": function_env_variables},
        Tags=function_tags,
    )

    # get function after created
    function = get_function(session, function_name)  # gets default self.function_name after creation

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
        Equivalent to value returned by https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda.html#Lambda.Client.get_function
        Returns None if no function role is found
    """
    if function_name is None:
        return None

    _log.debug("Getting lambda function {}".format(function_name))

    awslambda = session.client("lambda")
    try:
        function = awslambda.get_function(FunctionName=function_name)
        del function["ResponseMetadata"]  # remove response details from function
        return function
    except awslambda.exceptions.ResourceNotFoundException as e:
        _log.debug("Failed to get lambda function {} with exception: {}".format(function_name, e))
        return None


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


def add_trigger(session, function_name, statement_id, principle, source_arn):
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
    role_policies=["arn:aws:iam::aws:policy/AWSLambdaExecute"],
    role_tags=None,
):
    """Create IAM role
    
    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    role_name : str
        Role name to create
    role_policy_document : dict, optional
        Role policy document. 
        Defaults to trust policy allowing role to execute lambda functions.
        See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/iam.html#IAM.Client.create_role
    role_description : str, optional
        Role description
    role_policies : list, optional
        Role policies
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

    # default role_policy_document is lambda
    if role_policy_document is None:
        role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}
            ],
        }

    _log.debug("Creating IAM role {} with policies {}".format(role_name, role_policies))
    iam = session.client("iam")

    iam_config = {
        "RoleName": role_name,
        "Description": role_description,
        "AssumeRolePolicyDocument": json.dumps(role_policy_document),
    }

    # for some reason the tags API is different here
    if role_tags is not None:
        tags = []
        for key in role_tags.keys():
            tags.append({"Key": key, "Value": role_tags[key]})
        iam_config["Tags"] = tags

    # create role
    iam.create_role(**iam_config)

    # attached lambda execution policy
    for policy in role_policies:
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
        Equivalent to value returned in the 'Role' key in https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/iam.html#IAM.Client.get_role 
        Returns None if no role is found
    """
    if role_name is None:
        return None

    _log.debug("Getting IAM role with name {}".format(role_name))
    iam = session.client("iam")
    try:
        response = iam.get_role(RoleName=role_name)
        return response["Role"]
    except iam.exceptions.NoSuchEntityException as e:
        _log.debug("Failed to get IAM role for name {} with exception: {}".format(role_name, e))
        return None


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
    session,
    api_id,
    api_name="podpac-api",
    api_description="PODPAC API",
    api_version=None,
    api_tags={},
    api_endpoint="eval",
):
    """Create API Gateway REST API
    
    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    api_id : str
        API ID. If None, the API will be created. If the `api_id` is not None, this method will try to return the API associated with the ID
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
    
    No Longer Returned
    ------------------
    (dict, dict)
        (api, resource) dicts returned from API gateway creation
        ``api`` equivalent to https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigateway.html#APIGateway.Client.create_rest_api
        ``resource`` equivalent to https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigateway.html#APIGateway.Client.create_resource
    """
    # set version to podpac version, if None
    api = get_api(session, api_id)

    # TODO: add checks to make sure api parameters match
    if api is not None:
        resource = get_api_resource(session, api["id"], api_endpoint)
        _log.debug(
            "API '{}' and API resource {} already exist. Using existing API ID and resource.".format(
                api_name, api_endpoint
            )
        )
        return api, resource

    _log.debug("Creating API gateway with name {}".format(api_name))
    apigateway = session.client("apigateway")

    # set version default
    if api_version is None:
        api_version = version.semver()

    try:
        api = None  # in case commands fail, want to have this set to None
        resource = None
        api = apigateway.create_rest_api(
            name=api_name,
            description=api_description,
            version=api_version,
            binaryMediaTypes=["*/*"],
            apiKeySource="HEADER",
            endpointConfiguration={"types": ["REGIONAL"]},
            tags=api_tags,
        )

        resource = create_api_resource(session, api["id"], api_endpoint)

        # put method
        apigateway.put_method(
            restApiId=api["id"],
            resourceId=resource["id"],
            httpMethod="ANY",
            authorizationType="NONE",  # TODO: support "AWS_IAM"
            apiKeyRequired=False,  # TODO: create "generate_key()" method
        )

        return api, resource

    # clean up if any API methods fail
    except Exception as e:
        if api is not None:
            try:
                delete_api(session, api["id"])
            except:
                pass

        _log.error("Failed to build API gateway with name {} with exception: {}".format(api_name, e))
        raise e


def deploy_api(session, api_id, api_stage):
    """Deploy API gateway definition
    
    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    api_id : str
        API ID. Generated during API Creation.
    api_stage : str
        API Stage
    """
    if api_id is None or api_stage is None:
        raise ValueError("`api_id` and `api_stage` must be defined to deploy API")

    _log.debug("Deploying API Gateway with ID {} and stage {}".format(api_id, api_stage))

    apigateway = session.client("apigateway")
    apigateway.create_deployment(
        restApiId=api_id,
        stageName=api_stage,
        stageDescription="Deployment of Lambda function API",
        description="PODPAC Lambda Function API",
    )
    _log.debug("Deployed API Gateway with ID {} and stage {}".format(api_id, api_stage))


def get_api(session, api_id):
    """Get API Gateway definition
    
    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    api_id : str
        API ID. Generated during API Creation.
    
    Returns
    -------
    dict
        Returns output of Boto3 API Gateway creation
        Equivalent to https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigateway.html#APIGateway.Client.create_rest_api
        Returns None if API Id is not found
    """
    if api_id is None:
        return None

    _log.debug("Getting API Gateway with ID {}".format(api_id))
    apigateway = session.client("apigateway")

    try:
        api = apigateway.get_rest_api(restApiId=api_id)
        return api
    except (botocore.exceptions.ParamValidationError, apigateway.exceptions.NotFoundException) as e:
        _log.error("Failed to get API Gateway with ID {} with exception: {}".format(api_id, e))
        return None


def create_api_resource(session, api_id, api_endpoint):
    """Create API Resource (endpoint)
    
    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    api_id : str
        API ID. Generated during API Creation.
    api_endpoint : str
        API endpoint path
    
    Returns
    -------
    dict
        Returns output of Boto3 API Resource
        Equivalent to https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigateway.html#APIGateway.Client.get_resource
        Returns None if API Resource ID is not found
    """
    resource = get_api_resource(session, api_id, api_endpoint)
    if resource is not None:
        _log.debug("API resource '{}' already exists. Using existing resource.".format(api_endpoint))
        return resource

    _log.debug("Creating API Resource for  {}".format(api_id))
    apigateway = session.client("apigateway")

    # get resources to get access to parentId ("/" path)
    resources = apigateway.get_resources(restApiId=api_id)
    parent_id = resources["items"][0]["id"]  # TODO - make this based on path == "/" ?

    # create resource
    return apigateway.create_resource(restApiId=api_id, parentId=parent_id, pathPart=api_endpoint)


def get_api_resource(session, api_id, api_endpoint):
    """Get API Resource definition
    
    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    api_id : str
        API ID. Generated during API Creation.
    api_endpoint : str
        API endpoint path

    Returns
    -------
    dict
        Returns output of Boto3 API Resource
        Equivalent to https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigateway.html#APIGateway.Client.get_resource
        Returns None if API Resource ID is not found
    """
    if api_id is None or api_endpoint is None:
        return None

    _log.debug("Getting API resource {} for API ID {}".format(api_endpoint, api_id))
    apigateway = session.client("apigateway")

    try:
        resources = apigateway.get_resources(restApiId=api_id)
        resources_filtered = [r for r in resources["items"] if api_endpoint in r["path"]]
        resource = resources_filtered[0] if len(resources_filtered) else None
        return resource
    except (botocore.exceptions.ParamValidationError, apigateway.exceptions.NotFoundException) as e:
        _log.error("Failed to get API Gateway with ID {} with exception: {}".format(api_id, e))
        return None


def delete_api(session, api_id):
    """Delete API Gateway API
    
    Parameters
    ----------
    session : :class:`Session`
        AWS Boto3 Session. See :class:`Session` for creation.
    api_id : str
        API ID. Generated during API Creation.
    """
    if api_id is None:
        _log.error("`api_id` not defined in delete_api")
        return

    # make sure api exists
    api = get_api(session, api_id)
    if api is None:
        _log.debug("API Gateway '{}' does not exist".format(api_id))
        return

    _log.debug("Removing API Gateway with ID {}".format(api_id))

    apigateway = session.client("apigateway")
    apigateway.delete_rest_api(restApiId=api_id)

    _log.debug("Successfully removed API Gateway with ID {}".format(api_id))


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

    return logs
