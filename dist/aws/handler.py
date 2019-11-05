"""
PODPAC AWS Handler

Attributes
----------
s3 : TYPE
    Description
settings_json : dict
    Description
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import subprocess
import sys
import urllib.parse as urllib
from six import string_types
from collections import OrderedDict

import _pickle as cPickle

import boto3
import botocore


def is_s3_trigger(event):
    """
    Helper method to determine if the given event was triggered by an S3 event
    
    Parameters
    ----------
    event : dict
        Event dict from AWS. See [TODO: add link reference]
    
    Returns
    -------
    Bool
        True if the event is an S3 trigger
    """
    return "Records" in event and event["Records"][0]["eventSource"] == "aws:s3"


def handler(event, context, get_deps=True, ret_pipeline=False):
    """Lambda function handler
    
    Parameters
    ----------
    event : TYPE
        Description
    context : TYPE
        Description
    get_deps : bool, optional
        Description
    ret_pipeline : bool, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """

    def _update_podpac_settings(old_settings_json, new_settings_json):
        """
        Helper method to merge two local settings json/dict objects, then update the `PodpacSettings`.

        The settings variable here is podpac.settings.

        Parameters
        ----------
        old_settings_json : dict
            old settings dict
        new_settings_json : dict
            new settings dict to merge in
        """
        updated_settings = {**old_settings_json, **new_settings_json}
        for key in updated_settings:
            settings[key] = updated_settings[key]

    def _check_for_cached_output(input_file_key, pipeline, settings_json, bucket):
        """
        Helper function to determine if the requested output is already computed (and force_compute is false.)


        Parameters
        ----------
        input_file_key : str
            Description
        pipeline : dict
            Description
        settings_json : dict
            Description
        bucket : str
            Description

        Returns
        -------
        Bool
            Returns true if the requested output is already computed
        """
        output_filename = input_file_key.replace(".json", "." + pipeline["output"]["format"])
        output_filename = output_filename.replace(
            settings_json["FUNCTION_S3_INPUT"], settings_json["FUNCTION_S3_OUTPUT"]
        )
        try:
            s3.head_object(Bucket=bucket, Key=output_filename)
            # Object exists, so we don't have to recompute
            if not pipeline.get("force_compute", False):
                return True, output_filename
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # It does not exist, so we should proceed
                return False, output_filename
            # Something else has gone wrong... not handling this case.
            return False, output_filename
        return False, output_filename

    print(event)

    # Add /tmp/ path to handle python path for dependencies
    sys.path.append("/tmp/")

    # TODO: move to s3fs for simpler access
    # not sure if this is possible?
    # continue to assume that lambda handler has access to all s3 resources
    s3 = boto3.client("s3")

    args = []
    kwargs = {}

    # This event was triggered by S3, so let's see if the requested output is already cached.
    if is_s3_trigger(event):

        # We always have to look to the bucket that triggered the event for the input
        bucket = event["Records"][0]["s3"]["bucket"]["name"]

        file_key = urllib.unquote_plus(event["Records"][0]["s3"]["object"]["key"])
        _json = ""

        # get the object
        obj = s3.get_object(Bucket=bucket, Key=file_key)

        # get lines
        lines = obj["Body"].read().split(b"\n")
        for r in lines:
            if len(_json) > 0:
                _json += "\n"
            _json += r.decode()
        pipeline = json.loads(_json, object_pairs_hook=OrderedDict)

        # We can return if there is a valid cached output to save compute time.
        settings_json = pipeline["settings"]
        cached, output_filename = _check_for_cached_output(file_key, pipeline, settings_json, bucket)
        if cached:
            return
    else:
        print("DSullivan: we have an API Gateway event. Will now get deps in order to proceed.")
        url = event["queryStringParameters"]
        if isinstance(url, string_types):
            url = urllib.parse_qs(urllib.urlparse(url).query)

        # Capitalize the keywords for consistency
        settings_json = {}
        for k in url:
            if k.upper() == "SETTINGS":
                settings_json = url[k]
        bucket = settings_json["S3_BUCKET_NAME"]

    # get dependencies path
    if "FUNCTION_DEPENDENCIES_KEY" in settings_json:
        dependencies = settings_json["FUNCTION_DEPENDENCIES_KEY"]
    else:
        # TODO: this could be a problem, since we can't import podpac settings yet
        # which means the input event might have to include the version or
        # "FUNCTION_DEPENDENCIES_KEY".
        dependencies = "podpac_deps_{}.zip".format(
            settings_json["PODPAC_VERSION"]
        )  # this should be equivalent to version.semver()
    # Download dependencies from specific bucket/object
    if get_deps:
        s3.download_file(bucket, dependencies, "/tmp/" + dependencies)
        subprocess.call(["unzip", "/tmp/" + dependencies, "-d", "/tmp"])
        sys.path.append("/tmp/")
        subprocess.call(["rm", "/tmp/" + dependencies])

    # Need to set matplotlib backend to 'Agg' before importing it elsewhere
    import matplotlib

    matplotlib.use("agg")
    from podpac import settings
    from podpac.core.node import Node
    from podpac.core.coordinates import Coordinates
    from podpac.core.utils import JSONEncoder, _get_query_params_from_url
    import podpac.datalib

    try:
        _update_podpac_settings(settings, settings_json)
    except Exception:
        print("The settings could not be updated.")

    if is_s3_trigger(event):
        node = Node.from_definition(pipeline["pipeline"])
        coords = Coordinates.from_json(json.dumps(pipeline["coordinates"], indent=4, cls=JSONEncoder))
        fmt = pipeline["output"]["format"]
        kwargs = pipeline["output"].copy()
        kwargs.pop("format")
    else:
        coords = Coordinates.from_url(event["queryStringParameters"])
        node = Node.from_url(event["queryStringParameters"])
        fmt = _get_query_params_from_url(event["queryStringParameters"])["FORMAT"].split("/")[-1]
        if fmt in ["png", "jpg", "jpeg"]:
            kwargs["return_base64"] = True

    output = node.eval(coords)
    if ret_pipeline:
        return node

    body = output.to_format(fmt, *args, **kwargs)

    # output_filename only exists if this was an S3 triggered event.
    if output_filename is not None:
        s3.put_object(Bucket=bucket, Key=output_filename, Body=body)
    else:
        try:
            json.dumps(body)
        except Exception as e:
            print("AWS: body is not serializable, attempting to decode.")
            body = body.decode()
    return {"statusCode": 200, "headers": {"Content-Type": "image/png"}, "isBase64Encoded": True, "body": body}


#############
# Test Script
#############
if __name__ == "__main__":
    from podpac import settings

    # Need to authorize our s3 client when running locally.
    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=settings["AWS_SECRET_ACCESS_KEY"],
        region_name=settings["AWS_REGION_NAME"],
    )
    event = {"Records": [{"s3": {"object": {"key": "json/SinCoords.json"}, "bucket": {"name": "podpac-mls-test"}}}]}

    example = handler(event, {}, get_deps=False)
    print(example)
