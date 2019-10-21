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
from collections import OrderedDict

import _pickle as cPickle

import boto3
import botocore

# Add /tmp/ path to handle python path for dependencies
sys.path.append("/tmp/")

# TODO: move to s3fs for simpler access
# not sure if this is possible?
# continue to assume that lambda handler has access to all s3 resources
s3 = boto3.client("s3")

# TODO: handle settings from within the API gateway call or dynamically from bucket
# this is currently bundled with lambda function
settings_json = {}  # default to empty
try:
    # currently loaded into lambda function
    with open("settings.json", "r") as f:
        settings_json = json.load(f)

    print (settings_json)  # TODO: remove
except:
    pass


def handler(event, context, get_deps=True, ret_pipeline=False):

    # TODO: remove
    print (event)

    # get bucket path
    # this should be defined by S3 trigger or API gateway trigger
    bucket = settings_json["S3_BUCKET_NAME"]

    # get dependencies path
    # dependencies are labelled using `git describe`
    dependencies = "podpac_deps_{}.zip".format(settings_json["PODPAC_VERSION"])

    if "Records" in event and event["Records"][0]["eventSource"] == "aws:s3":
        # <start S3 trigger specific>

        # TODO: get the name of the calling bucket from event for s3 triggers?
        # bucket = event['Records'][0]['s3']['bucket']['name']

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
    else:
        # elif ('pathParameters' in event and event['pathParameters'] is not None and 'proxy' in event['pathParameters']) or ('authorizationToken' in event and event['authorizationToken'] == "incoming-client-token"):
        # TODO: Need to get the pipeline from the event...
        print ("DSullivan: we have an API Gateway event")
        pipeline = None

    # Download dependencies from specific bucket/object
    if get_deps:
        s3.download_file(bucket, "podpac/" + dependencies, "/tmp/" + dependencies)
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

    # check if file exists
    if pipeline is not None:
        filename = file_key.replace(".json", "." + pipeline["output"]["format"])
        filename = filename.replace(settings["S3_JSON_FOLDER"], settings["S3_OUTPUT_FOLDER"])
        try:
            s3.head_object(Bucket=bucket, Key=filename)
            # Object exists, so we don't have to recompute
            if not pipeline.get("force_compute", False):
                return
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # It does not exist, so we should proceed
                pass
            else:
                # Something else has gone wrong... not handling this case.
                pass

    args = []
    kwargs = {}
    # if from S3 trigger
    if pipeline is not None:
        # if 'Records' in event and event['Records'][0]['eventSource'] == 'aws:s3':
        node = Node.from_definition(pipeline["pipeline"])
        coords = Coordinates.from_json(json.dumps(pipeline["coordinates"], indent=4, cls=JSONEncoder))
        fmt = pipeline["output"]["format"]
        kwargs = pipeline["output"].copy()
        kwargs.pop("format")

    # else from api gateway and it's a WMS/WCS request
    else:
        # elif ('pathParameters' in event and event['pathParameters'] is not None and 'proxy' in event['pathParameters']) or ('authorizationToken' in event and event['authorizationToken'] == "incoming-client-token"):
        print (_get_query_params_from_url(event["queryStringParameters"]))
        coords = Coordinates.from_url(event["queryStringParameters"])
        node = Node.from_url(event["queryStringParameters"])
        fmt = _get_query_params_from_url(event["queryStringParameters"])["FORMAT"].split("/")[-1]
        if fmt in ["png", "jpg", "jpeg"]:
            kwargs["return_base64"] = True

    output = node.eval(coords)
    if ret_pipeline:
        return node

    body = output.to_format(fmt, *args, **kwargs)
    if pipeline is not None:
        s3.put_object(Bucket=bucket, Key=filename, Body=body)
    else:
        try:
            json.dumps(body)
        except Exception as e:
            print ("AWS: body is not serializable, attempting to decode.")
            body = body.decode()
    return {"statusCode": 200, "headers": {"Content-Type": "image/png"}, "isBase64Encoded": True, "body": body}


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
    print (example)
