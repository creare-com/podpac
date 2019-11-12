"""
PODPAC AWS Handler
"""

import json
import subprocess
import sys
import urllib.parse as urllib
from collections import OrderedDict


import boto3
import botocore


def default_pipeline(pipeline=None):
    """Get default pipeline definiton, merging with input pipline if supplied
    
    Parameters
    ----------
    pipeline : dict, optional
        Input pipline. Will fill in any missing defaults.
    
    Returns
    -------
    dict
        pipeline dict
    """
    defaults = {
        "pipeline": {},
        "settings": {},
        "output": {"format": "netcdf", "filename": None, "format_kwargs": {}},
        # API Gateway
        "url": "",
        "params": {},
    }

    # merge defaults with input pipelines, if supplied
    if pipeline is not None:
        pipeline = {**defaults, **pipeline}
        pipeline["output"] = {**defaults["output"], **pipeline["output"]}
        pipeline["settings"] = {**defaults["settings"], **pipeline["settings"]}
    else:
        pipeline = defaults

    return pipeline


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

        # event is the pipeline, provide consistent pipeline defaults
        pipeline = default_pipeline(event)

        return pipeline

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

        # provide consistent pipeline defaults
        pipeline = default_pipeline(pipeline)

        # create output filename
        pipeline["output"]["filename"] = file_key.replace(".json", "." + pipeline["output"]["format"]).replace(
            pipeline["settings"]["FUNCTION_S3_INPUT"], pipeline["settings"]["FUNCTION_S3_OUTPUT"]
        )

        if not pipeline["settings"]["FUNCTION_FORCE_COMPUTE"]:

            # get configured s3 bucket to check for cache
            bucket = pipeline["settings"]["S3_BUCKET_NAME"]

            # We can return if there is a valid cached output to save compute time.
            try:
                s3.head_object(Bucket=bucket, Key=pipeline["output"]["filename"])
                return None

            # throws ClientError if no file is found
            except botocore.exceptions.ClientError:
                pass

        # return pipeline definition
        return pipeline

    elif trigger == "APIGateway":
        print("Triggered from API Gateway")

        pipeline = default_pipeline()
        pipeline["url"] = event["queryStringParameters"]
        pipeline["params"] = urllib.parse_qs(urllib.urlparse(pipeline["url"]).query)

        # make all params lowercase
        pipeline["params"] = [param.lower() for param in pipeline["params"]]

        # look for specific parameter definitions in query parameters
        for param in pipeline["params"]:
            if param == "settings":
                pipeline["settings"] = pipeline["params"][param]

            if param == "output":
                pipeline["output"] = pipeline["params"][param]

        # handle FORMAT in query parameters
        if "format" in pipeline["params"]:
            pipeline["output"]["format"] = pipeline["params"]["format"].split("/")[-1]

        # handle image returns
        if pipeline["output"]["format"] in ["png", "jpg", "jpeg"]:
            pipeline["output"]["format_kwargs"]["return_base64"] = True

        return pipeline

    else:
        raise Exception("Unsupported trigger")


def handler(event, context):
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
    settings = {**settings, **pipeline["settings"]}

    # build the Node and Coordinates
    if trigger in ("eval", "S3"):
        node = Node.from_definition(pipeline["pipeline"])
        coords = Coordinates.from_json(json.dumps(pipeline["coordinates"], indent=4, cls=JSONEncoder))

    # TODO: handle API Gateway better - is this always going to be WCS?
    elif trigger == "APIGateway":
        node = Node.from_url(pipeline["url"])
        coords = Coordinates.from_url(pipeline["url"])

    # run analysis
    output = node.eval(coords)

    # convert to output format
    body = output.to_format(pipeline["output"]["format"], **pipeline["output"]["format_kwargs"])

    # Response
    if trigger == "eval":
        return body

    elif trigger == "S3":
        s3.put_object(Bucket=settings["S3_BUCKET_NAME"], Key=pipeline["output"]["filename"], Body=body)

    elif trigger == "APIGateway":

        # TODO: can we handle the deserialization better?
        try:
            json.dumps(body)
        except Exception as e:
            print("Output body is not serializable, attempting to decode.")
            body = body.decode()

        return {"statusCode": 200, "headers": {"Content-Type": "image/png"}, "isBase64Encoded": True, "body": body}
