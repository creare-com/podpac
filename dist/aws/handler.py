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


def get_trigger(event):
    """
    Helper method to determine the trigger for the lambda invocation
    
    Parameters
    ----------
    event : dict
        Event dict from AWS. See [TODO: add link reference]
    
    Returns
    -------
    str
        One of "S3", "eval", or "APIGateway"
    """

    if "Records" in event and event["Records"][0]["eventSource"] == "aws:s3":
        return "S3"
    elif "queryStringParameters" in event:
        return "APIGateway"
    else:
        return "eval"


def parse_event(trigger, event):
    """Parse pipeline, settings, and output details from event depending on trigger
    
    Parameters
    ----------
    trigger : str
        One of "S3", "eval", or "APIGateway"
    event : dict
        Event dict from AWS. See [TODO: add link reference]
    """

    if trigger == "eval":
        print("Triggered by Invoke")

        # TODO: implement
        return None
    elif trigger == "S3":
        print("Triggered from S3")

        # get boto s3 client
        s3 = boto3.client("s3")

        # We always have to look to the bucket that triggered the event for the input
        triggered_bucket = event["Records"][0]["s3"]["bucket"]["name"]

        # get the pipeline object and read
        file_key = urllib.unquote_plus(event["Records"][0]["s3"]["object"]["key"])
        pipline_obj = s3.get_object(Bucket=triggered_bucket, Key=file_key)
        pipeline = json.loads(pipline_obj["Body"].read().decode("utf-8"))

        # create output filename
        pipeline["output_filename"] = file_key.replace(".json", "." + pipeline["output"]["format"]).replace(
            pipeline["settings"]["FUNCTION_S3_INPUT"], pipeline["settings"]["FUNCTION_S3_OUTPUT"]
        )

        if not pipeline["settings"]["force_compute"]:
            # get configured s3 bucket to check for cache
            bucket = pipeline["settings"]["S3_BUCKET_NAME"]

            # We can return if there is a valid cached output to save compute time.
            try:
                s3.head_object(Bucket=bucket, Key=pipeline["output_filename"])
                return None

            # throws ClientError if no file is found
            except botocore.exceptions.ClientError:
                pass

        # return pipeline definition
        return pipeline

    elif trigger == "APIGateway":
        print("Triggered from API Gateway")

        url = event["queryStringParameters"]
        if isinstance(url, string_types):
            url = urllib.parse_qs(urllib.urlparse(url).query)

        # Capitalize the keywords for consistency
        pipeline = {}
        for param in url:

            if param.upper() == "PIPELINE":
                pipeline["pipeline"] = url[param]

            # TODO: do we still need this? will overwrite pipeline above
            if param.upper() == "SETTINGS":
                pipeline["settings"] = url[param]

            if param.upper() == "OUTPUT":
                pipeline["output"] = url[param]

        return pipeline

    else:
        raise Exception("Unsupported trigger")


def handler(event, context, get_deps=True, ret_pipeline=False):
    """Lambda function handler
    
    Parameters
    ----------
    event : dict
        Description
    context : TYPE
        Description
    get_deps : bool, optional
        Description
    ret_pipeline : bool, optional
        Description
    """
    print(event)

    # Add /tmp/ path to handle python path for dependencies
    sys.path.append("/tmp/")

    # handle triggers
    trigger = get_trigger(event)

    # parse event
    pipeline = parse_event(trigger, event)

    # bail if we can't parse
    if pipeline is None:
        return

    # -----
    # TODO: remove when layers is configured
    # get configured bucket to download dependencies
    bucket = pipeline["settings"]["S3_BUCKET_NAME"]

    # get dependencies path
    if "FUNCTION_DEPENDENCIES_KEY" in pipeline["settings"]:
        dependencies = pipeline["settings"]["FUNCTION_DEPENDENCIES_KEY"]
    else:
        dependencies = "podpac_deps_{}.zip".format(
            pipeline["settings"]["PODPAC_VERSION"]
        )  # this should be equivalent to version.semver()

    # Download dependencies from specific bucket/object
    s3 = boto3.client("s3")
    s3.download_file(bucket, dependencies, "/tmp/" + dependencies)
    subprocess.call(["unzip", "/tmp/" + dependencies, "-d", "/tmp"])
    sys.path.append("/tmp/")
    subprocess.call(["rm", "/tmp/" + dependencies])
    # -----

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
    for key in pipeline["settings"]:
        settings[key] = pipeline["settings"][key]

    # TODO: load this into pipeline["output"]
    kwargs = {}
    if trigger == "eval":
        pass

    elif trigger == "S3":
        node = Node.from_definition(pipeline["pipeline"])
        coords = Coordinates.from_json(json.dumps(pipeline["coordinates"], indent=4, cls=JSONEncoder))
        fmt = pipeline["output"]["format"]
        kwargs = pipeline["output"].copy()
        kwargs.pop("format")  # get rid of format

    # TODO: handle API Gateway better - is this always going to be WCS?
    elif trigger == "APIGateway":
        # TODO: handle this in the parser above - not sure what the spec should be here
        node = Node.from_url(event["queryStringParameters"])
        coords = Coordinates.from_url(event["queryStringParameters"])
        fmt = _get_query_params_from_url(event["queryStringParameters"])["FORMAT"].split("/")[-1]
        if fmt in ["png", "jpg", "jpeg"]:
            kwargs["return_base64"] = True

    # run analysis and covert to output format
    output = node.eval(coords)
    body = output.to_format(fmt, **kwargs)

    # ########
    # Response
    # ########
    if trigger == "eval":
        pass
        # return {"statusCode": 200, "headers": {"Content-Type": "image/png"}, "isBase64Encoded": True, "body": body}
    elif trigger == "S3":
        s3.put_object(Bucket=settings["S3_BUCKET_NAME"], Key=pipeline["output_filename"], Body=body)

    elif trigger == "APIGateway":
        # TODO: can we handle the deserialization better?
        try:
            json.dumps(body)
        except Exception as e:
            print("Output body is not serializable, attempting to decode.")
            body = body.decode()

        return {"statusCode": 200, "headers": {"Content-Type": "image/png"}, "isBase64Encoded": True, "body": body}
