"""
Unit tests for podpac.core.managers.aws

All tests mock AWS API calls — no real AWS credentials needed.
"""

import io
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

boto3_real = pytest.importorskip("boto3")
botocore = pytest.importorskip("botocore")
import botocore.exceptions

from podpac import settings
from podpac.core.node import Node
from podpac.core.managers.aws import (
    Lambda,
    LambdaException,
    Session,
    get_function,
    create_function,
    update_function,
    delete_function,
    add_function_trigger,
    remove_function_trigger,
    get_role,
    create_role,
    delete_role,
    get_bucket,
    create_bucket,
    delete_bucket,
    get_object,
    put_object,
    get_api,
    deploy_api,
    delete_api,
    get_budget,
    create_budget,
    delete_budget,
    get_logs,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class MockNode(Node):
    """Minimal concrete Node for use as Lambda source."""

    def eval(self, coordinates, output=None, selector=None):  # noqa: A003
        pass


class _MockBoto3Session(boto3_real.Session):
    """boto3.Session subclass that returns MagicMock clients — no real AWS calls."""

    def __init__(self):
        super().__init__(
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region_name="us-east-1",
        )
        self._clients = {}

    def client(self, service_name, **kwargs):
        if service_name not in self._clients:
            self._clients[service_name] = MagicMock()
        return self._clients[service_name]

    def resource(self, service_name, **kwargs):
        return MagicMock()

    def get_account_id(self):
        return "123456789012"


def _mock_session():
    """Plain MagicMock shaped like a Session, for standalone function tests."""
    session = MagicMock()
    session.region_name = "us-east-1"
    session.get_account_id.return_value = "123456789012"
    return session


def _function_dict(name="test-fn"):
    return {
        "Configuration": {
            "FunctionArn": "arn:aws:lambda:us-east-1:123456789012:function:{}".format(name),
            "Handler": "handler.handler",
            "Description": "PODPAC Lambda Function",
            "Environment": {"Variables": {}},
            "Timeout": 600,
            "MemorySize": 2048,
            "LastModified": "2021-01-01T00:00:00.000+0000",
            "Version": "$LATEST",
            "CodeSha256": "abc123",
            "Role": "arn:aws:iam::123456789012:role/test-role",
        },
        "tags": {},
    }


def _role_dict(name="test-role"):
    return {
        "RoleName": name,
        "Arn": "arn:aws:iam::123456789012:role/{}".format(name),
        "Description": "Test role",
        "AssumeRolePolicyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        },
        "policy_document": None,
        "policy_arns": [],
        "tags": {},
    }


# ---------------------------------------------------------------------------
# TestSession
# ---------------------------------------------------------------------------


class TestSession(object):
    @patch("boto3.Session.__init__", return_value=None)
    def test_init_uses_settings_defaults(self, mock_boto_init):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        with patch.object(Session, "client", return_value=mock_sts):
            with settings:
                settings["AWS_ACCESS_KEY_ID"] = "env-key"
                settings["AWS_SECRET_ACCESS_KEY"] = "env-secret"
                settings["AWS_REGION_NAME"] = "eu-west-1"
                Session()
        mock_boto_init.assert_called_once_with(
            aws_access_key_id="env-key",
            aws_secret_access_key="env-secret",
            region_name="eu-west-1",
        )

    @patch("boto3.Session.__init__", return_value=None)
    def test_init_explicit_credentials_override_settings(self, mock_boto_init):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        with patch.object(Session, "client", return_value=mock_sts):
            with settings:
                settings["AWS_ACCESS_KEY_ID"] = "should-not-use"
                Session(
                    aws_access_key_id="explicit-key",
                    aws_secret_access_key="explicit-secret",
                    region_name="ap-east-1",
                )
        mock_boto_init.assert_called_once_with(
            aws_access_key_id="explicit-key",
            aws_secret_access_key="explicit-secret",
            region_name="ap-east-1",
        )

    @patch("boto3.Session.__init__", return_value=None)
    def test_init_invalid_credentials_raises_value_error(self, _):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "InvalidClientTokenId", "Message": "Invalid token"}},
            "GetCallerIdentity",
        )
        with patch.object(Session, "client", return_value=mock_sts):
            with pytest.raises(ValueError, match="credential check failed"):
                Session()

    @patch("boto3.Session.__init__", return_value=None)
    def test_get_account_id(self, _):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "555444333222"}
        with patch.object(Session, "client", return_value=mock_sts):
            s = Session()
            result = s.get_account_id()
        assert result == "555444333222"


# ---------------------------------------------------------------------------
# TestGetFunction
# ---------------------------------------------------------------------------


class TestGetFunction(object):
    def test_returns_none_for_none_name(self):
        assert get_function(_mock_session(), None) is None

    def test_returns_none_when_not_found(self):
        session = _mock_session()
        mock_lambda = MagicMock()
        ResourceNotFound = type("ResourceNotFoundException", (Exception,), {})
        mock_lambda.exceptions.ResourceNotFoundException = ResourceNotFound
        mock_lambda.get_function.side_effect = ResourceNotFound("not found")
        session.client.return_value = mock_lambda

        assert get_function(session, "missing-fn") is None

    def test_returns_function_with_tags(self):
        session = _mock_session()
        mock_lambda = MagicMock()
        ResourceNotFound = type("ResourceNotFoundException", (Exception,), {})
        mock_lambda.exceptions.ResourceNotFoundException = ResourceNotFound
        mock_lambda.get_function.return_value = {
            "ResponseMetadata": {},
            "Configuration": {"FunctionArn": "arn:aws:lambda:us-east-1:123:function:fn"},
        }
        mock_lambda.list_tags.return_value = {"Tags": {"env": "test"}}
        session.client.return_value = mock_lambda

        result = get_function(session, "test-fn")
        assert result is not None
        assert result["tags"] == {"env": "test"}
        assert "ResponseMetadata" not in result

    def test_tags_default_empty_on_client_error(self):
        session = _mock_session()
        mock_lambda = MagicMock()
        ResourceNotFound = type("ResourceNotFoundException", (Exception,), {})
        mock_lambda.exceptions.ResourceNotFoundException = ResourceNotFound
        mock_lambda.get_function.return_value = {
            "ResponseMetadata": {},
            "Configuration": {"FunctionArn": "arn:aws:lambda:us-east-1:123:function:fn"},
        }
        mock_lambda.list_tags.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": ""}}, "ListTags"
        )
        session.client.return_value = mock_lambda

        result = get_function(session, "test-fn")
        assert result["tags"] == {}


# ---------------------------------------------------------------------------
# TestCreateFunction
# ---------------------------------------------------------------------------


class TestCreateFunction(object):
    def test_returns_existing_function_without_creating(self):
        session = _mock_session()
        existing = _function_dict()
        with patch("podpac.core.managers.aws.get_function", return_value=existing):
            result = create_function(session, "test-fn", "arn:role", "handler.handler")
        assert result is existing

    def test_raises_when_no_source_defined(self):
        session = _mock_session()
        with patch("podpac.core.managers.aws.get_function", return_value=None):
            with pytest.raises(ValueError, match="source is not defined"):
                create_function(session, "test-fn", "arn:role", "handler.handler")

    def test_creates_function_from_s3(self):
        session = _mock_session()
        created = _function_dict()
        mock_lambda = MagicMock()
        session.client.return_value = mock_lambda

        with patch("podpac.core.managers.aws.get_function", side_effect=[None, created]):
            result = create_function(
                session,
                "test-fn",
                "arn:role",
                "handler.handler",
                function_source_bucket="my-bucket",
                function_source_dist_key="dist.zip",
            )
        mock_lambda.create_function.assert_called_once()
        assert result is created

    def test_raises_not_implemented_for_local_zip(self):
        session = _mock_session()
        with patch("podpac.core.managers.aws.get_function", return_value=None):
            with pytest.raises(NotImplementedError):
                create_function(
                    session,
                    "test-fn",
                    "arn:role",
                    "handler.handler",
                    function_source_dist_zip="my.zip",
                )


# ---------------------------------------------------------------------------
# TestUpdateFunction
# ---------------------------------------------------------------------------


class TestUpdateFunction(object):
    def test_raises_when_function_not_found(self):
        session = _mock_session()
        with patch("podpac.core.managers.aws.get_function", return_value=None):
            with pytest.raises(ValueError, match="does not exist"):
                update_function(session, "missing-fn", function_source_bucket="b", function_source_dist_key="k")

    def test_updates_from_s3(self):
        session = _mock_session()
        existing = _function_dict()
        updated = _function_dict()
        mock_lambda = MagicMock()
        session.client.return_value = mock_lambda

        with patch("podpac.core.managers.aws.get_function", side_effect=[existing, updated]):
            result = update_function(session, "test-fn", function_source_bucket="b", function_source_dist_key="k")
        mock_lambda.update_function_code.assert_called_once()
        assert result is updated

    def test_raises_not_implemented_for_local_zip(self):
        session = _mock_session()
        with patch("podpac.core.managers.aws.get_function", return_value=_function_dict()):
            with pytest.raises(NotImplementedError):
                update_function(session, "test-fn", function_source_dist_zip="my.zip")


# ---------------------------------------------------------------------------
# TestDeleteFunction
# ---------------------------------------------------------------------------


class TestDeleteFunction(object):
    def test_noop_for_none_name(self):
        delete_function(_mock_session(), None)

    def test_noop_when_not_found(self):
        session = _mock_session()
        with patch("podpac.core.managers.aws.get_function", return_value=None):
            delete_function(session, "missing-fn")

    def test_deletes_existing_function(self):
        session = _mock_session()
        mock_lambda = MagicMock()
        session.client.return_value = mock_lambda
        with patch("podpac.core.managers.aws.get_function", return_value=_function_dict()):
            delete_function(session, "test-fn")
        mock_lambda.delete_function.assert_called_once_with(FunctionName="test-fn")


# ---------------------------------------------------------------------------
# TestAddRemoveFunctionTrigger
# ---------------------------------------------------------------------------


class TestAddRemoveFunctionTrigger(object):
    def test_add_trigger_raises_when_required_args_missing(self):
        with pytest.raises(ValueError):
            add_function_trigger(_mock_session(), None, "sid", "principal", "arn")

    def test_add_trigger_calls_add_permission(self):
        session = _mock_session()
        mock_lambda = MagicMock()
        session.client.return_value = mock_lambda
        add_function_trigger(session, "test-fn", "sid-1", "s3.amazonaws.com", "arn:aws:s3:::bucket")
        mock_lambda.add_permission.assert_called_once()

    def test_remove_trigger_noop_for_none(self):
        remove_function_trigger(_mock_session(), None, None)

    def test_remove_trigger_calls_remove_permission(self):
        session = _mock_session()
        mock_lambda = MagicMock()
        ResourceNotFound = type("ResourceNotFoundException", (Exception,), {})
        mock_lambda.exceptions.ResourceNotFoundException = ResourceNotFound
        session.client.return_value = mock_lambda
        remove_function_trigger(session, "test-fn", "sid-1")
        mock_lambda.remove_permission.assert_called_once_with(FunctionName="test-fn", StatementId="sid-1")

    def test_remove_trigger_swallows_not_found(self):
        session = _mock_session()
        mock_lambda = MagicMock()
        ResourceNotFound = type("ResourceNotFoundException", (Exception,), {})
        mock_lambda.exceptions.ResourceNotFoundException = ResourceNotFound
        mock_lambda.remove_permission.side_effect = ResourceNotFound("not found")
        session.client.return_value = mock_lambda
        remove_function_trigger(session, "test-fn", "sid-1")  # should not raise


# ---------------------------------------------------------------------------
# TestGetRole
# ---------------------------------------------------------------------------


class TestGetRole(object):
    def test_returns_none_for_none_name(self):
        assert get_role(_mock_session(), None) is None

    def test_returns_none_when_not_found(self):
        session = _mock_session()
        mock_iam = MagicMock()
        NoSuchEntity = type("NoSuchEntityException", (Exception,), {})
        mock_iam.exceptions.NoSuchEntityException = NoSuchEntity
        mock_iam.get_role.side_effect = NoSuchEntity("no such entity")
        session.client.return_value = mock_iam

        assert get_role(session, "missing-role") is None

    def test_returns_role_with_policy_and_tags(self):
        session = _mock_session()
        mock_iam = MagicMock()
        NoSuchEntity = type("NoSuchEntityException", (Exception,), {})
        mock_iam.exceptions.NoSuchEntityException = NoSuchEntity
        mock_iam.get_role.return_value = {
            "Role": {
                "RoleName": "test-role",
                "Arn": "arn:aws:iam::123:role/test-role",
                "Description": "desc",
                "AssumeRolePolicyDocument": {},
            }
        }
        mock_iam.get_role_policy.return_value = {"PolicyDocument": {"Version": "2012-10-17"}}
        mock_iam.list_attached_role_policies.return_value = {
            "AttachedPolicies": [{"PolicyArn": "arn:aws:iam::aws:policy/AWSLambdaBasicExecutionRole"}]
        }
        mock_iam.list_role_tags.return_value = {"Tags": [{"Key": "env", "Value": "test"}]}
        session.client.return_value = mock_iam

        result = get_role(session, "test-role")
        assert result["RoleName"] == "test-role"
        assert "policy_document" in result
        assert result["policy_arns"] == ["arn:aws:iam::aws:policy/AWSLambdaBasicExecutionRole"]
        assert result["tags"] == {"env": "test"}


# ---------------------------------------------------------------------------
# TestCreateRole
# ---------------------------------------------------------------------------


class TestCreateRole(object):
    def test_returns_existing_role_without_creating(self):
        session = _mock_session()
        existing = _role_dict()
        with patch("podpac.core.managers.aws.get_role", return_value=existing):
            result = create_role(session, "test-role")
        assert result is existing

    def test_creates_role_when_not_found(self):
        session = _mock_session()
        created = _role_dict()
        mock_iam = MagicMock()
        session.client.return_value = mock_iam

        with patch("podpac.core.managers.aws.get_role", side_effect=[None, created]):
            result = create_role(session, "test-role")
        mock_iam.create_role.assert_called_once()
        assert result is created

    def test_attaches_policy_arns(self):
        session = _mock_session()
        mock_iam = MagicMock()
        session.client.return_value = mock_iam

        with patch("podpac.core.managers.aws.get_role", side_effect=[None, _role_dict()]):
            create_role(session, "test-role", role_policy_arns=["arn:aws:iam::aws:policy/SomePolicy"])
        mock_iam.attach_role_policy.assert_called_once_with(
            RoleName="test-role", PolicyArn="arn:aws:iam::aws:policy/SomePolicy"
        )

    def test_puts_inline_policy_when_provided(self):
        session = _mock_session()
        mock_iam = MagicMock()
        session.client.return_value = mock_iam
        policy_doc = {"Version": "2012-10-17", "Statement": []}

        with patch("podpac.core.managers.aws.get_role", side_effect=[None, _role_dict()]):
            create_role(session, "test-role", role_policy_document=policy_doc)
        mock_iam.put_role_policy.assert_called_once()


# ---------------------------------------------------------------------------
# TestDeleteRole
# ---------------------------------------------------------------------------


class TestDeleteRole(object):
    def test_noop_for_none_name(self):
        delete_role(_mock_session(), None)

    def test_noop_when_role_not_found(self):
        session = _mock_session()
        with patch("podpac.core.managers.aws.get_role", return_value=None):
            delete_role(session, "missing-role")

    def test_deletes_role_and_detaches_policies(self):
        session = _mock_session()
        mock_iam = MagicMock()
        mock_iam.list_attached_role_policies.return_value = {
            "AttachedPolicies": [{"PolicyArn": "arn:aws:iam::aws:policy/SomePolicy"}]
        }
        session.client.return_value = mock_iam
        with patch("podpac.core.managers.aws.get_role", return_value=_role_dict()):
            delete_role(session, "test-role")
        mock_iam.detach_role_policy.assert_called_once_with(
            RoleName="test-role", PolicyArn="arn:aws:iam::aws:policy/SomePolicy"
        )
        mock_iam.delete_role.assert_called_once_with(RoleName="test-role")


# ---------------------------------------------------------------------------
# TestGetBucket
# ---------------------------------------------------------------------------


class TestGetBucket(object):
    def test_returns_none_for_none_name(self):
        assert get_bucket(_mock_session(), None) is None

    def test_returns_none_when_not_found(self):
        session = _mock_session()
        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "404", "Message": "Not found"}}, "HeadBucket"
        )
        session.client.return_value = mock_s3
        assert get_bucket(session, "missing-bucket") is None

    def test_returns_bucket_dict_with_tags(self):
        session = _mock_session()
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}
        mock_s3.get_bucket_location.return_value = {"LocationConstraint": "us-east-1"}
        mock_s3.get_bucket_policy.return_value = {"Policy": '{"Version":"2012-10-17"}'}
        mock_s3.get_bucket_tagging.return_value = {"TagSet": [{"Key": "env", "Value": "test"}]}
        session.client.return_value = mock_s3

        result = get_bucket(session, "test-bucket")
        assert result["name"] == "test-bucket"
        assert result["region"] == "us-east-1"
        assert result["tags"] == {"env": "test"}


# ---------------------------------------------------------------------------
# TestCreateBucket
# ---------------------------------------------------------------------------


class TestCreateBucket(object):
    def test_returns_existing_bucket_without_creating(self):
        session = _mock_session()
        existing = {"name": "test-bucket", "region": "us-east-1", "policy": None, "tags": {}}
        with patch("podpac.core.managers.aws.get_bucket", return_value=existing):
            result = create_bucket(session, "test-bucket")
        assert result is existing

    def test_creates_new_bucket_with_tags(self):
        session = _mock_session()
        created = {"name": "test-bucket", "region": None, "policy": None, "tags": {"env": "test"}}
        mock_s3 = MagicMock()
        session.client.return_value = mock_s3

        with patch("podpac.core.managers.aws.get_bucket", side_effect=[None, created]):
            result = create_bucket(session, "test-bucket", bucket_tags={"env": "test"})
        mock_s3.create_bucket.assert_called_once()
        mock_s3.put_bucket_tagging.assert_called_once()
        assert result is created

    def test_sets_bucket_policy_when_provided(self):
        session = _mock_session()
        created = {"name": "test-bucket", "region": None, "policy": None, "tags": {}}
        mock_s3 = MagicMock()
        session.client.return_value = mock_s3
        policy = {"Version": "2012-10-17", "Statement": []}

        with patch("podpac.core.managers.aws.get_bucket", side_effect=[None, created]):
            create_bucket(session, "test-bucket", bucket_policy=policy)
        mock_s3.put_bucket_policy.assert_called_once()


# ---------------------------------------------------------------------------
# TestDeleteBucket
# ---------------------------------------------------------------------------


class TestDeleteBucket(object):
    def test_noop_for_none_name(self):
        delete_bucket(_mock_session(), None)

    def test_noop_when_not_found(self):
        session = _mock_session()
        with patch("podpac.core.managers.aws.get_bucket", return_value=None):
            delete_bucket(session, "missing-bucket")

    def test_deletes_bucket(self):
        session = _mock_session()
        mock_s3 = MagicMock()
        session.client.return_value = mock_s3
        with patch("podpac.core.managers.aws.get_bucket", return_value={"name": "test-bucket", "tags": {}}):
            delete_bucket(session, "test-bucket")
        mock_s3.delete_bucket.assert_called_once_with(Bucket="test-bucket")

    def test_deletes_objects_before_bucket_when_requested(self):
        session = _mock_session()
        mock_s3_resource = MagicMock()
        mock_bucket_resource = MagicMock()
        mock_s3_resource.Bucket.return_value = mock_bucket_resource
        session.resource.return_value = mock_s3_resource
        session.client.return_value = MagicMock()

        with patch("podpac.core.managers.aws.get_bucket", return_value={"name": "test-bucket", "tags": {}}):
            delete_bucket(session, "test-bucket", delete_objects=True)
        mock_bucket_resource.object_versions.delete.assert_called_once()
        mock_bucket_resource.objects.all.return_value.delete.assert_called_once()


# ---------------------------------------------------------------------------
# TestGetObject / TestPutObject
# ---------------------------------------------------------------------------


class TestGetObject(object):
    def test_returns_none_for_none_bucket(self):
        assert get_object(_mock_session(), None, "key") is None

    def test_returns_none_for_none_path(self):
        assert get_object(_mock_session(), "bucket", None) is None

    def test_returns_none_when_object_not_found(self):
        session = _mock_session()
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "404", "Message": "Not found"}}, "HeadObject"
        )
        session.client.return_value = mock_s3
        assert get_object(session, "bucket", "key") is None

    def test_returns_s3_object_on_success(self):
        session = _mock_session()
        mock_s3 = MagicMock()
        mock_s3.head_object.return_value = {}
        expected = {"Body": b"data"}
        mock_s3.get_object.return_value = expected
        session.client.return_value = mock_s3

        result = get_object(session, "bucket", "key")
        assert result is expected


class TestPutObject(object):
    def test_noop_for_none_bucket(self):
        put_object(_mock_session(), None, "key")

    def test_noop_for_none_path(self):
        put_object(_mock_session(), "bucket", None)

    def test_puts_bytes_directly(self):
        session = _mock_session()
        mock_s3 = MagicMock()
        session.client.return_value = mock_s3
        put_object(session, "bucket", "key", file=b"some-bytes")
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["Body"] == b"some-bytes"

    def test_reads_file_from_path(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_bytes(b"file-contents")
        session = _mock_session()
        mock_s3 = MagicMock()
        session.client.return_value = mock_s3
        put_object(session, "bucket", "key", file=str(path))
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["Body"] == b"file-contents"

    def test_no_body_when_file_is_none(self):
        session = _mock_session()
        mock_s3 = MagicMock()
        session.client.return_value = mock_s3
        put_object(session, "bucket", "key/")
        call_kwargs = mock_s3.put_object.call_args[1]
        assert "Body" not in call_kwargs


# ---------------------------------------------------------------------------
# TestGetAPI / TestCreateAPI / TestDeployAPI / TestDeleteAPI
# ---------------------------------------------------------------------------


class TestGetAPI(object):
    def test_returns_none_for_none_name(self):
        assert get_api(_mock_session(), None, None) is None

    def test_returns_none_when_api_not_found(self):
        session = _mock_session()
        mock_apigw = MagicMock()
        mock_apigw.get_rest_apis.return_value = {"items": []}
        session.client.return_value = mock_apigw
        assert get_api(session, "missing-api", "eval") is None

    def _make_apigw_mock(self):
        mock_apigw = MagicMock()
        NotFound = type("NotFoundException", (Exception,), {})
        mock_apigw.exceptions.NotFoundException = NotFound
        return mock_apigw

    def test_returns_api_with_matching_resource(self):
        session = _mock_session()
        mock_apigw = self._make_apigw_mock()
        mock_apigw.get_rest_apis.return_value = {"items": [{"id": "abc123", "name": "test-api"}]}
        mock_apigw.get_rest_api.return_value = {
            "ResponseMetadata": {},
            "id": "abc123",
            "name": "test-api",
            "description": "test",
            "version": "1",
            "tags": {},
        }
        mock_apigw.get_stages.return_value = {"item": [{"stageName": "prod"}]}
        mock_apigw.get_resources.return_value = {
            "items": [
                {"id": "root", "path": "/"},
                {"id": "res1", "path": "/eval", "pathPart": "eval"},
            ]
        }
        session.client.return_value = mock_apigw

        result = get_api(session, "test-api", "eval")
        assert result is not None
        assert result["resource"]["pathPart"] == "eval"

    def test_resource_is_none_when_endpoint_not_found(self):
        session = _mock_session()
        mock_apigw = self._make_apigw_mock()
        mock_apigw.get_rest_apis.return_value = {"items": [{"id": "abc123", "name": "test-api"}]}
        mock_apigw.get_rest_api.return_value = {
            "ResponseMetadata": {},
            "id": "abc123",
            "name": "test-api",
            "description": "test",
            "version": "1",
            "tags": {},
        }
        mock_apigw.get_stages.return_value = {"item": []}
        mock_apigw.get_resources.return_value = {"items": [{"id": "root", "path": "/"}]}
        session.client.return_value = mock_apigw

        result = get_api(session, "test-api", "eval")
        assert result["resource"] is None


class TestDeployAPI(object):
    def test_raises_when_api_id_is_none(self):
        with pytest.raises(ValueError):
            deploy_api(_mock_session(), None, "prod")

    def test_raises_when_stage_is_none(self):
        with pytest.raises(ValueError):
            deploy_api(_mock_session(), "abc123", None)

    def test_calls_create_deployment(self):
        session = _mock_session()
        mock_apigw = MagicMock()
        session.client.return_value = mock_apigw
        deploy_api(session, "abc123", "prod")
        mock_apigw.create_deployment.assert_called_once_with(
            restApiId="abc123",
            stageName="prod",
            stageDescription="Deployment of PODPAC API",
            description="PODPAC API",
        )


class TestDeleteAPI(object):
    def test_noop_for_none_name(self):
        delete_api(_mock_session(), None)

    def test_noop_when_api_not_found(self):
        session = _mock_session()
        with patch("podpac.core.managers.aws.get_api", return_value=None):
            delete_api(session, "missing-api")

    def test_deletes_api(self):
        session = _mock_session()
        mock_apigw = MagicMock()
        session.client.return_value = mock_apigw
        with patch("podpac.core.managers.aws.get_api", return_value={"id": "abc123", "name": "test-api"}):
            delete_api(session, "test-api")
        mock_apigw.delete_rest_api.assert_called_once_with(restApiId="abc123")


# ---------------------------------------------------------------------------
# TestGetBudget / TestCreateBudget / TestDeleteBudget
# ---------------------------------------------------------------------------


class TestGetBudget(object):
    def test_returns_none_on_not_found(self):
        session = _mock_session()
        mock_budgets = MagicMock()
        NotFound = type("NotFoundException", (Exception,), {})
        mock_budgets.exceptions.NotFoundException = NotFound
        mock_budgets.describe_budget.side_effect = NotFound("not found")
        session.client.return_value = mock_budgets

        assert get_budget(session, "missing-budget") is None

    def test_returns_budget_on_success(self):
        session = _mock_session()
        mock_budgets = MagicMock()
        NotFound = type("NotFoundException", (Exception,), {})
        mock_budgets.exceptions.NotFoundException = NotFound
        budget_data = {"BudgetName": "test-budget", "BudgetLimit": {"Amount": "100.0", "Unit": "USD"}}
        mock_budgets.describe_budget.return_value = {"Budget": budget_data}
        session.client.return_value = mock_budgets

        result = get_budget(session, "test-budget")
        assert result["BudgetName"] == "test-budget"


class TestCreateBudget(object):
    def test_returns_existing_budget_without_creating(self):
        session = _mock_session()
        existing = {"BudgetName": "test-budget", "BudgetLimit": {"Amount": "100.0", "Unit": "USD"}}
        with patch("podpac.core.managers.aws.get_budget", return_value=existing):
            result = create_budget(session, 100.0)
        assert result is existing

    def test_creates_budget_without_email(self):
        session = _mock_session()
        created = {"BudgetName": "test-budget", "BudgetLimit": {"Amount": "100.0", "Unit": "USD"}}
        mock_budgets = MagicMock()
        session.client.return_value = mock_budgets

        with patch("podpac.core.managers.aws.get_budget", side_effect=[None, created]):
            create_budget(session, 100.0, budget_name="test-budget")
        call_kwargs = mock_budgets.create_budget.call_args[1]
        assert "NotificationsWithSubscribers" not in call_kwargs

    def test_creates_budget_with_email_notification(self):
        session = _mock_session()
        created = {"BudgetName": "test-budget", "BudgetLimit": {"Amount": "100.0", "Unit": "USD"}}
        mock_budgets = MagicMock()
        session.client.return_value = mock_budgets

        with patch("podpac.core.managers.aws.get_budget", side_effect=[None, created]):
            create_budget(session, 100.0, budget_email="test@example.com", budget_name="test-budget")
        call_kwargs = mock_budgets.create_budget.call_args[1]
        assert "NotificationsWithSubscribers" in call_kwargs


class TestDeleteBudget(object):
    def test_deletes_budget(self):
        session = _mock_session()
        mock_budgets = MagicMock()
        NotFound = type("NotFoundException", (Exception,), {})
        mock_budgets.exceptions.NotFoundException = NotFound
        session.client.return_value = mock_budgets
        delete_budget(session, "test-budget")
        mock_budgets.delete_budget.assert_called_once()

    def test_noop_when_budget_not_found(self):
        session = _mock_session()
        mock_budgets = MagicMock()
        NotFound = type("NotFoundException", (Exception,), {})
        mock_budgets.exceptions.NotFoundException = NotFound
        mock_budgets.delete_budget.side_effect = NotFound("not found")
        session.client.return_value = mock_budgets
        delete_budget(session, "missing-budget")  # should not raise


# ---------------------------------------------------------------------------
# TestGetLogs
# ---------------------------------------------------------------------------


class TestGetLogs(object):
    def test_returns_empty_list_when_log_group_not_found(self):
        session = _mock_session()
        mock_logs = MagicMock()
        ResourceNotFound = type("ResourceNotFoundException", (Exception,), {})
        mock_logs.exceptions.ResourceNotFoundException = ResourceNotFound
        mock_logs.describe_log_streams.side_effect = ResourceNotFound("no group")
        session.client.return_value = mock_logs

        result = get_logs(session, "/aws/lambda/test-fn")
        assert result == []

    def test_returns_sorted_log_events(self):
        import numpy as np

        session = _mock_session()
        mock_logs = MagicMock()
        ResourceNotFound = type("ResourceNotFoundException", (Exception,), {})
        mock_logs.exceptions.ResourceNotFoundException = ResourceNotFound

        now_ms = int(np.datetime64("now").astype(float) * 1000)
        mock_logs.describe_log_streams.return_value = {
            "logStreams": [
                {
                    "logStreamName": "stream-1",
                    "firstEventTimestamp": now_ms - 200000,
                    "lastEventTimestamp": now_ms + 200000,
                }
            ]
        }
        mock_logs.get_log_events.return_value = {
            "events": [
                {"timestamp": now_ms, "message": "second"},
                {"timestamp": now_ms - 1000, "message": "first"},
            ]
        }
        session.client.return_value = mock_logs

        result = get_logs(session, "/aws/lambda/test-fn", limit=10)
        assert len(result) == 2
        assert result[0]["timestamp"] < result[1]["timestamp"]


# ---------------------------------------------------------------------------
# TestLambdaClass
# ---------------------------------------------------------------------------


class TestLambdaClass(object):
    """Tests for the Lambda Node class — uses _MockBoto3Session to avoid real AWS calls."""

    def _make_lambda(self, **kwargs):
        session = _MockBoto3Session()
        return Lambda(session=session, **kwargs)

    # -- Defaults / attributes --

    def test_function_name_from_settings(self):
        with settings:
            settings["FUNCTION_NAME"] = "my-custom-function"
            node = self._make_lambda()
            assert node.function_name == "my-custom-function"

    def test_function_name_autogen_when_setting_is_none(self):
        with settings:
            settings["FUNCTION_NAME"] = None
            node = self._make_lambda()
            assert node.function_name == "podpac-lambda-autogen"

    def test_function_triggers_default_to_eval_only(self):
        node = self._make_lambda()
        assert node.function_triggers == ["eval"]

    def test_function_triggers_include_s3_when_eval_trigger_is_s3(self):
        node = self._make_lambda(function_eval_trigger="S3")
        assert "S3" in node.function_triggers
        assert "eval" in node.function_triggers

    def test_pipeline_property_has_required_keys(self):
        node = self._make_lambda(source=MockNode())
        pipeline = node.pipeline
        assert "pipeline" in pipeline
        assert "output" in pipeline
        assert "settings" in pipeline

    # -- eval() dispatch --

    def test_eval_raises_without_source(self):
        node = self._make_lambda()
        with pytest.raises(ValueError, match="source"):
            node.eval(MagicMock())

    def test_eval_dispatches_to_invoke_for_eval_trigger(self):
        node = self._make_lambda(source=MockNode(), function_eval_trigger="eval")
        with patch.object(node, "_eval_invoke", return_value=None) as mock_invoke:
            node.eval(MagicMock())
        mock_invoke.assert_called_once()

    def test_eval_dispatches_to_s3_for_s3_trigger(self):
        node = self._make_lambda(source=MockNode(), function_eval_trigger="S3")
        with patch.object(node, "_eval_s3", return_value=None) as mock_s3:
            node.eval(MagicMock())
        mock_s3.assert_called_once()

    def test_eval_raises_not_implemented_for_apigw_trigger(self):
        node = self._make_lambda(source=MockNode(), function_eval_trigger="APIGateway")
        with pytest.raises(NotImplementedError):
            node.eval(MagicMock())

    # -- _eval_invoke() --

    def test_eval_invoke_async_returns_none(self):
        node = self._make_lambda(source=MockNode(), download_result=False)
        mock_lambda_client = node.session.client("lambda")
        mock_lambda_client.invoke.return_value = {}
        mock_pipeline = {"pipeline": {}, "output": {}, "settings": {}, "coordinates": {}}

        with patch.object(node, "_create_eval_pipeline", return_value=mock_pipeline):
            result = node._eval_invoke(MagicMock())

        assert result is None
        assert mock_lambda_client.invoke.call_args.kwargs["InvocationType"] == "Event"

    def test_eval_invoke_raises_lambda_exception_on_function_error(self):
        node = self._make_lambda(source=MockNode(), download_result=True)
        mock_lambda_client = node.session.client("lambda")
        error_payload = json.dumps(
            {
                "errorType": "RuntimeError",
                "errorMessage": "Something went wrong",
                "stackTrace": ["line 1"],
            }
        ).encode("utf-8")
        mock_lambda_client.invoke.return_value = {
            "FunctionError": "Unhandled",
            "Payload": io.BytesIO(error_payload),
        }
        mock_pipeline = {"pipeline": {}, "output": {}, "settings": {}, "coordinates": {}}

        with patch.object(node, "_create_eval_pipeline", return_value=mock_pipeline):
            with pytest.raises(LambdaException, match="RuntimeError"):
                node._eval_invoke(MagicMock())

    def test_eval_invoke_returns_string_for_non_netcdf_payload(self):
        node = self._make_lambda(source=MockNode(), download_result=True)
        mock_lambda_client = node.session.client("lambda")
        mock_lambda_client.invoke.return_value = {"Payload": io.BytesIO(b"plain text result")}
        mock_pipeline = {"pipeline": {}, "output": {}, "settings": {}, "coordinates": {}}

        with patch.object(node, "_create_eval_pipeline", return_value=mock_pipeline):
            with patch("podpac.core.managers.aws.UnitsDataArray.open", side_effect=ValueError("not netcdf")):
                result = node._eval_invoke(MagicMock())
        assert result == "plain text result"

    # -- _eval_s3() --

    def test_eval_s3_no_download_returns_none_after_put(self):
        node = self._make_lambda(source=MockNode(), download_result=False)
        mock_s3 = node.session.client("s3")
        mock_pipeline = {"pipeline": {}, "output": {}, "settings": {}, "coordinates": {}}

        with patch.object(node, "_create_eval_pipeline", return_value=mock_pipeline):
            result = node._eval_s3(MagicMock())

        assert result is None
        mock_s3.put_object.assert_called_once()

    def test_eval_s3_puts_pipeline_json_to_input_folder(self):
        node = self._make_lambda(source=MockNode(), download_result=False)
        mock_s3 = node.session.client("s3")
        mock_pipeline = {"pipeline": {}, "output": {}, "settings": {}, "coordinates": {}}

        with patch.object(node, "_create_eval_pipeline", return_value=mock_pipeline):
            node._eval_s3(MagicMock())

        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert node.function_s3_bucket == call_kwargs["Bucket"]
        assert call_kwargs["Key"].endswith(".json")

    # -- validate_role() --

    def test_validate_role_returns_false_when_role_is_none(self):
        node = self._make_lambda()
        node._role = None
        assert node.validate_role() is False

    def test_validate_role_returns_false_when_lambda_principal_missing(self):
        node = self._make_lambda()
        node._role = {"RoleName": "test-role"}
        node.set_trait(
            "function_role_assume_policy_document",
            {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}
                ],
            },
        )
        assert node.validate_role() is False

    def test_validate_role_returns_true_with_valid_default_policy(self):
        node = self._make_lambda()
        node._role = {"RoleName": "test-role"}
        assert node.validate_role() is True

    # -- _set_*() internals --

    def test_set_function_populates_attributes(self):
        node = self._make_lambda()
        fn = _function_dict()
        node._set_function(fn)
        assert node._function_arn == fn["Configuration"]["FunctionArn"]
        assert node._function_code_sha256 == "abc123"
        assert node._function is fn

    def test_set_function_noop_on_none(self):
        node = self._make_lambda()
        original_arn = node._function_arn
        node._set_function(None)
        assert node._function_arn == original_arn

    def test_set_role_populates_attributes(self):
        node = self._make_lambda()
        role = _role_dict()
        node._set_role(role)
        assert node._function_role_arn == role["Arn"]
        assert node.function_role_name == role["RoleName"]
        assert node._role is role

    def test_set_role_noop_on_none(self):
        node = self._make_lambda()
        node._set_role(None)
        assert node._role is None

    def test_set_bucket_populates_attributes(self):
        node = self._make_lambda()
        bucket = {"name": "my-bucket", "tags": {"env": "test"}}
        node._set_bucket(bucket)
        assert node.function_s3_bucket == "my-bucket"
        assert node._bucket is bucket

    def test_set_api_populates_attributes(self):
        node = self._make_lambda()
        api = {
            "id": "abc123",
            "name": "my-api",
            "description": "test api",
            "version": "1.0",
            "tags": {},
            "stage": "prod",
            "resource": {"id": "res1", "pathPart": "eval"},
        }
        node._set_api(api)
        assert node._function_api_id == "abc123"
        assert node.function_api_stage == "prod"
        assert node.function_api_endpoint == "eval"
        assert node._api is api

    def test_set_budget_populates_attributes(self):
        node = self._make_lambda()
        budget = {
            "BudgetName": "test-budget",
            "BudgetLimit": {"Amount": "100.0", "Unit": "USD"},
        }
        node._set_budget(budget)
        assert np.isclose(node.function_budget_amount, 100.0, rtol=1e-09, atol=1e-09)
        assert node.function_budget_currency == "USD"
        assert node._budget is budget

    def test_repr_includes_function_name_and_bucket(self):
        node = self._make_lambda()
        r = repr(node)
        assert node.function_name in r
        assert node.function_s3_bucket in r
