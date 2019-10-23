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

from io import BytesIO

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
    s3_bucket_name : string
        s3 bucket name to use with lambda function
    s3_json_folder : TYPE
        folder in `s3_bucket_name` to store pipelines
    s3_output_folder : TYPE
        folder in `s3_bucket_name` to watch for output
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

    # lambda function parameters
    function_name = tl.Unicode(default_value="podpac-lambda-autogen")
    function_role_name = tl.Unicode(default_value=None, allow_none=True)  # will create role if None
    function_handler = tl.Unicode(default_value="handler.handler")
    function_description = tl.Unicode(default_value="PODPAC Lambda Function (https://podpac.org)")
    function_env_variables = tl.Dict(default_value={})  # environment vars in function
    function_tags = tl.Dict(default_value={})  # key: value for tags on function (and any created roles)
    function_timeout = tl.Int(default_value=600)
    function_memory = tl.Int(default_value=2048)
    function_source_bucket = tl.Unicode(default_value="podpac-dist")
    function_source_key = tl.Unicode()

    @tl.default("function_source_key")
    def _function_source_key_default(self):
        return "{}/podpac_dist.zip".format(version.semver())

    # readonly properties
    function_arn = tl.Unicode()
    function_last_modified = tl.Unicode()
    function_version = tl.Unicode()
    function_code_sha256 = tl.Unicode()
    function_api_url = tl.Unicode()
    function_api_id = tl.Unicode()
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

    def create_function(self, recreate=False):
        """Build Lambda function and associated resources on AWS
        
        Parameters
        ----------
        recreate : bool, optional
            If function already exists on AWS, remove function and re-add
        """

        function = self.get_function()  # gets default self.function_name
        if function is not None:
            _log.debug("AWS lambda function '{}' already exists. Using existing function.".format(self.function_name))

            if recreate:
                _log.debug("Recreating lambda function '{}'".format(self.function_name))
                self.delete_function()  # TODO: autoremove?

            else:
                # set function to class
                self._set_function(function)

                # try function? # compare function?
                # TODO: Add comparison function here to see if function has changed
                return

        awslambda = self.session.client("lambda")
        _log.debug("No AWS lambda function '{}' exists. Creating function...".format(self.function_name))

        # create default role if it doesn't exist
        role = self.get_role()  # gets default self.function_role_name
        if role is None:
            role = self.create_role(
                role_tags=self.function_tags
            )  # create role with the same tags as the function, by default

            # after creating a role, you need to wait ~10 seconds before its active and will work with the lambda function
            # this is not cool
            time.sleep(10)

        # create function
        awslambda.create_function(
            FunctionName=self.function_name,
            Runtime="python3.7",
            Role=role["Arn"],
            Handler=self.function_handler,
            Code={"S3Bucket": self.function_source_bucket, "S3Key": self.function_source_key},
            Description=self.function_description,
            Timeout=self.function_timeout,
            MemorySize=self.function_memory,
            Publish=True,
            Environment={"Variables": self.function_env_variables},
            Tags=self.function_tags,
        )

        # get function after created
        function = self.get_function()  # gets default self.function_name after creation
        if function is None:
            raise ValueError("Failed to get created function from AWS lambda")

        # create API gateway
        function["api"] = self.create_api(function_name=self.function_name)

        # set function to class
        self._set_function(function)

    def get_function(self, function_name=None):
        """Get function definition from AWS
        
        Parameters
        ----------
        function_name : str, optional
            function name to get. Defaults to :attr:`self.function_name`
            
        Returns
        -------
        dict
            dict returned from AWS defining function
            Equivalent to value returned by https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lambda.html#Lambda.Client.get_function
            Returns None if not role is found
        """
        if function_name is None:
            if self.function_name is not None:
                function_name = self.function_name
            else:
                return None

        awslambda = self.session.client("lambda")
        try:
            function = awslambda.get_function(FunctionName=function_name)
            del function["ResponseMetadata"]  # remove response details from function
            return function
        except awslambda.exceptions.ResourceNotFoundException as e:
            _log.debug("failed to get function {} with exception: {}".format(function_name, e))
            return None

    def delete_function(self, function_name=None, autoremove=False):
        """Remove AWS Lambda function and associated resources on AWS
        """

        if function_name is None:
            if self.function_name is not None:
                function_name = self.function_name
            else:
                return

        _log.debug("Removing lambda function '{}'".format(function_name))
        awslambda = self.session.client("lambda")
        awslambda.delete_function(FunctionName=function_name)

        # TODO add autoremove
        if autoremove:
            self.delete_role()
            self.delete_api()
            self._remove_triggers()

    def create_role(
        self,
        role_name="podpac-lambda-role-autogen",
        role_description="PODPAC Lambda Role",
        role_policies=["arn:aws:iam::aws:policy/AWSLambdaExecute"],
        role_tags={},
    ):
        """Create IAM role to execute podpac lambda function
        
        Parameters
        ----------
        role_name : str, optional
            Description
        role_description : str, optional
            Description
        role_policies : list, optional
            Description
        role_tags : dict, optional
            Description
        
        Returns
        -------
        dict
            dict returned from AWS defining role.
            Equivalent to value returned in the 'Role' key in https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/iam.html#IAM.Client.get_role 
        """

        role = self.get_role(role_name=role_name)
        if role is not None:
            _log.debug("AWS role '{}' already exists. Using existing role.".format(role_name))
            return role

        # create role
        iam = self.session.client("iam")
        _log.debug("No AWS role '{}' exists. Creating role...".format(role_name))

        # for some reason the tags API is different here
        tags = []
        for key in role_tags.keys():
            tags.append({"Key": key, "Value": role_tags[key]})

        # enable role to be run by lambda - this document is defined by AWS
        lambda_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}
            ],
        }

        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(lambda_policy_document),
            Description=role_description,
            Tags=tags,
        )

        # attached lambda execution policy
        for policy in role_policies:
            response = iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)

        # get finalized role
        role = self.get_role(role_name=role_name)
        _log.debug("Successfully created AWS role '{}'".format(role_name))

        return role

    def get_role(self, role_name=None):
        """Get role definition from AWS
        
        Parameters
        ----------
        role_name : str, optional
            Role name to get. Defaults to :attr:`self.function_role_name`
        
        Returns
        -------
        dict
            dict returned from AWS defining role.
            Equivalent to value returned in the 'Role' key in https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/iam.html#IAM.Client.get_role 
            Returns None if not role is found
        """
        if role_name is None:
            if self.function_role_name is not None:
                role_name = self.function_role_name
            else:
                return None

        iam = self.session.client("iam")
        try:
            response = iam.get_role(RoleName=role_name)
            return response["Role"]
        except iam.exceptions.NoSuchEntityException as e:
            _log.debug("failed to get role {} with exception: {}".format(role_name, e))
            return None

    def get_role_name(self, role_arn):
        """
        CURRENTLY NOT USED

        Get function role name based on role_arn
        
        Parameters
        ----------
        role_arn : str
            role arn
        
        Returns
        -------
        str
            role name
        """
        iam = self.session.client("iam")
        roles = iam.list_roles()
        role = [role for role in roles["Roles"] if role["Arn"] == role_arn]
        role_name = role[0] if len(role) else None
        return role_name

    def delete_role(self, role_name=None):
        """Remove role from AWS resources
        
        Parameters
        ----------
        role_name : str, optional
            Role name to delete. Defaults to :attr:`self.function_role_name`
        """
        if role_name is None:
            if self.function_role_name is not None:
                role_name = self.function_role_name
            else:
                return

        _log.debug("Removing AWS role '{}'".format(role_name))
        iam = self.session.client("iam")

        # need to detach policies first
        response = iam.list_attached_role_policies(RoleName=role_name)
        for policy in response["AttachedPolicies"]:
            iam.detach_role_policy(RoleName=role_name, PolicyArn=policy["PolicyArn"])

        iam.delete_role(RoleName=role_name)
        _log.debug("Successfully removed AWS role '{}'".format(role_name))

    def create_api(
        self,
        function_name=None,
        api_name=None,
        api_description=None,
        api_version=None,
        api_tags={},
        api_stage="prod",
        api_resource_path="eval",
    ):
        """Create API Gateway API for lambda function
        
        Parameters
        ----------
        function_name : None, optional
            Defaults to :attr:`self.function_name`
        api_name : None, optional
            API Name
        api_description : None, optional
            API Description
        api_version : str, optional
            API Version. Defaults to PODPAC version.
        api_tags : dict, optional
            API tags
        api_stage : str, optional
            API Stage
        api_resource_path : str, optional
            API endpoint
        
        Returns
        -------
        dict
            dict returned from API gateway creation
            Equivalent to https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigateway.html#APIGateway.Client.create_rest_api
            with the addition of a "url" key with the API url
        """
        if function_name is None:
            if self.function_name is not None:
                function_name = self.function_name
            else:
                raise ValueError("Function name must be defined when creating api")

        function = self.get_function(function_name=function_name)
        function_arn = function["Configuration"]["FunctionArn"]

        # set name and description
        if api_name is None:
            api_name = "{}-api".format(function_name)

        if api_description is None:
            api_description = "PODPAC Lambda REST API for {} function".format(function_name)

        if api_version is None:
            api_version = version.semver()

        _log.debug("Creating API gateway for function {}".format(function_name))
        apigateway = self.session.client("apigateway")

        try:
            api = None  # in case first command fails
            api = apigateway.create_rest_api(
                name=api_name,
                description=api_description,
                version=api_version,
                binaryMediaTypes=["*/*"],
                apiKeySource="HEADER",
                endpointConfiguration={"types": ["REGIONAL"]},
                tags=api_tags,
            )

            # get resources to get access to parentId ("/" path)
            resources = apigateway.get_resources(restApiId=api["id"])
            parent_id = resources["items"][0]["id"]  # to do - make this based on path == "/" ?

            # create resource
            resource = apigateway.create_resource(restApiId=api["id"], parentId=parent_id, pathPart=api_resource_path)

            # put method
            apigateway.put_method(
                restApiId=api["id"],
                resourceId=resource["id"],
                httpMethod="ANY",
                authorizationType="NONE",  # TODO: support "AWS_IAM"
                apiKeyRequired=False,  # TODO: create "generate_key()" method
            )

            # lambda proxy integration - this feels pretty brittle due to uri
            # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigateway.html#APIGateway.Client.put_integration
            aws_lambda_uri = "arn:aws:apigateway:{}:lambda:path/2015-03-31/functions".format(self.session.region_name)
            uri = "{}/{}/invocations".format(aws_lambda_uri, function_arn)
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

            # deploy
            apigateway.create_deployment(
                restApiId=api["id"],
                stageName=api_stage,
                stageDescription="Deployment of Lambda function API",
                description="PODPAC Lambda Function API",
            )

            # add permission to invoke call lambda - this feels brittle due to source_arn
            statement_id = "APIGateway"
            principle = "apigateway.amazonaws.com"
            source_arn = "arn:aws:execute-api:{}:{}:{}/*/*/*".format(
                self.session.region_name, self.session.get_account_id(), api["id"]
            )
            self._add_trigger(statement_id, principle, source_arn)
            api["url"] = self._get_api_url(api, api_stage, api_resource_path)

            return api

        except Exception as e:
            # clean up
            if api is not None:
                try:
                    self.delete_api(api["id"])
                except:
                    pass

            _log.error("Failed to build API gateway with exception: {}".format(e))
            return None

    def get_api(self, api_id=None):
        """Get API Gateway definition
        
        Parameters
        ----------
        api_id : None, optional
            API ID. Defaults to :attr:`self.function_api_id
        
        Returns
        -------
        dict
            dict returned from API gateway creation
            Equivalent to https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigateway.html#APIGateway.Client.create_rest_api
            returns None if no API found
        """
        if api_id is None:
            if self.function_api_id is not None:
                api_id = self.function_api_id
            else:
                return None

        apigateway = self.session.client("apigateway")

        try:
            api = apigateway.get_rest_api(restApiId=api_id)
            return api
        except (botocore.exceptions.ParamValidationError, apigateway.exceptions.NotFoundException) as e:
            _log.debug("failed to get API {} with exception: {}".format(api_id, e))
            return None

    def delete_api(self, api_id=None):
        """Delete API Gateway API
        
        Parameters
        ----------
        api_id : None, optional
            Description
        """
        if api_id is None:
            if self.function_api_id is not None:
                api_id = self.function_api_id
            else:
                return

        _log.debug("Removing API gateway '{}'".format(api_id))
        apigateway = self.session.client("apigateway")
        apigateway.delete_rest_api(restApiId=api_id)
        _log.debug("Successfully removed API gateway '{}'".format(api_id))

    # -----------------------------------------------------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------------------------------------------------

    def _get_api_url(self, api, api_stage, api_resource_path):
        """Generated API url
        
        Parameters
        ----------
        api : dict
            API dict returned from :meth:`self.get_api`
        api_stage : str
            API Stage
        api_resource_path : str
            API resource path
        """
        return "https://{}.execute-api.{}.amazonaws.com/{}/{}".format(
            api["id"], self.session.region_name, api_stage, api_resource_path
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

        if "api" in function:
            self.function_api_id = function["api"]["id"]
            self.function_api_url = function["api"]["url"]

    def _add_trigger(self, statement_id, principle, source_arn):
        """Add trigger (permission) to lambda function
        
        Parameters
        ----------
        statement_id : str
            Description
        principle : str
            Description
        source_arn : str
            Description
        
        Returns
        -------
        dict
            Description
        """
        awslambda = self.session.client("lambda")
        awslambda.add_permission(
            FunctionName=self.function_name,
            StatementId=statement_id,
            Action="lambda:InvokeFunction",
            Principal=principle,
            SourceArn=source_arn,
        )
        self.function_triggers[statement_id] = source_arn

    def _remove_trigger(self, statement_id):
        """Remove trigger (permission) from lambda function
        
        Parameters
        ----------
        statement_id : str
            Description
        """
        awslambda = self.session.client("lambda")
        awslambda.remove_permission(FunctionName=self.function_name, StatementId=statement_id)
        del self.function_triggers[statement_id]

    def _remove_triggers(self):
        """Remove all triggers from function
        """
        for trigger in self.function_triggers:
            self._remove_trigger(trigger)

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
