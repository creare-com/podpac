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

    print(event)

    # Add /tmp/ path to handle python path for dependencies
    sys.path.append("/tmp/")

    # get S3 client - note this will have the same access that the current "function role" does
    s3 = boto3.client("s3")

    args = []
    kwargs = {}

    # This event was triggered by S3, so let's see if the requested output is already cached.
    if is_s3_trigger(event):

        # We always have to look to the bucket that triggered the event for the input
        bucket = event["Records"][0]["s3"]["bucket"]["name"]

        file_key = urllib.unquote_plus(event["Records"][0]["s3"]["object"]["key"])
        _json = ""

        # get the pipeline object and read
        pipeline_obj = s3.get_object(Bucket=bucket, Key=file_key)
        pipeline = json.loads(pipeline_obj["Body"].read().decode("utf-8"))

        # We can return if there is a valid cached output to save compute time.
        settings_trigger = pipeline["settings"]

        # check for cached output
        cached = False
        output_filename = file_key.replace(".json", "." + pipeline["output"]["format"]).replace(
            settings_trigger["FUNCTION_S3_INPUT"], settings_trigger["FUNCTION_S3_OUTPUT"]
        )

        try:
            s3.head_object(Bucket=bucket, Key=output_filename)

            # Object exists, so we don't have to recompute
            # TODO: the "force_compute" parameter will never work as is written in aws.py
            if not pipeline.get("force_compute", False):
                cached = True

        except botocore.exceptions.ClientError:
            pass

        if cached:
            return

    # TODO: handle "invoke" and "APIGateway" triggers explicitly
    else:
        print("Not triggered by S3")

        url = event["queryStringParameters"]
        if isinstance(url, string_types):
            url = urllib.parse_qs(urllib.urlparse(url).query)

        # Capitalize the keywords for consistency
        settings_trigger = {}
        for k in url:
            if k.upper() == "SETTINGS":
                settings_trigger = url[k]
        bucket = settings_trigger["S3_BUCKET_NAME"]

    # get dependencies path
    if "FUNCTION_DEPENDENCIES_KEY" in settings_trigger:
        dependencies = settings_trigger["FUNCTION_DEPENDENCIES_KEY"]
    else:
        # TODO: this could be a problem, since we can't import podpac settings yet
        # which means the input event might have to include the version or
        # "FUNCTION_DEPENDENCIES_KEY".
        dependencies = "podpac_deps_{}.zip".format(
            settings_trigger["PODPAC_VERSION"]
        )  # this should be equivalent to version.semver()

    # Download dependencies from specific bucket/object
    if get_deps:
        s3.download_file(bucket, dependencies, "/tmp/" + dependencies)
        subprocess.call(["unzip", "/tmp/" + dependencies, "-d", "/tmp"])
        sys.path.append("/tmp/")
        subprocess.call(["rm", "/tmp/" + dependencies])

    # Load PODPAC

    # Need to set matplotlib backend to 'Agg' before importing it elsewhere
    import matplotlib

    matplotlib.use("agg")
    from podpac import settings
    from podpac.core.node import Node
    from podpac.core.coordinates import Coordinates
    from podpac.core.utils import JSONEncoder, _get_query_params_from_url
    import podpac.datalib

    # update podpac settings with inputs from the trigger
    for key in settings_trigger:
        settings[key] = settings_trigger[key]

    if is_s3_trigger(event):
        node = Node.from_definition(pipeline["pipeline"])
        coords = Coordinates.from_json(json.dumps(pipeline["coordinates"], indent=4, cls=JSONEncoder))
        fmt = pipeline["output"]["format"]
        kwargs = pipeline["output"].copy()
        kwargs.pop("format")

    # TODO: handle "invoke" and "APIGateway" triggers explicitly
    else:
        coords = Coordinates.from_url(event["queryStringParameters"])
        node = Node.from_url(event["queryStringParameters"])
        fmt = _get_query_params_from_url(event["queryStringParameters"])["FORMAT"].split("/")[-1]
        if fmt in ["png", "jpg", "jpeg"]:
            kwargs["return_base64"] = True

    output = node.eval(coords)

    # FOR DEBUGGING
    if ret_pipeline:
        return node

    body = output.to_format(fmt, *args, **kwargs)

    # output_filename only exists if this was an S3 triggered event.
    if is_s3_trigger(event):
        s3.put_object(Bucket=bucket, Key=output_filename, Body=body)

    # TODO: handle "invoke" and "APIGateway" triggers explicitly
    else:
        try:
            json.dumps(body)
        except Exception as e:
            print("Output body is not serializable, attempting to decode.")
            body = body.decode()

        return {"statusCode": 200, "headers": {"Content-Type": "image/png"}, "isBase64Encoded": True, "body": body}
