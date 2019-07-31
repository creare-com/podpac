from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import subprocess
import sys
import urllib.parse as urllib
from collections import OrderedDict

import _pickle as cPickle

import boto3
import botocore

# sys.path.insert(0, '/tmp/podpac/')
sys.path.append('/tmp/')
# sys.path.append(os.getcwd() + '/podpac/')

s3 = boto3.client('s3')
deps = 'podpac_deps_ESIP3.zip'


def handler(event, context, get_deps=True, ret_pipeline=False):
    print(event)
    bucket_name = 'podpac-s3'
    if get_deps:
        s3.download_file(bucket_name, 'podpac/' + deps, '/tmp/' + deps)
        subprocess.call(['unzip', '/tmp/' + deps, '-d', '/tmp'])
        sys.path.append('/tmp/')
        subprocess.call(['rm', '/tmp/' + deps])

    if 'Records' in event and event['Records'][0]['eventSource'] == 'aws:s3':
        # <start S3 trigger specific>
        file_key = urllib.unquote_plus(
            event['Records'][0]['s3']['object']['key'])
        _json = ''
        # get the object
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        # get lines
        lines = obj['Body'].read().split(b'\n')
        for r in lines:
            if len(_json) > 0:
                _json += '\n'
            _json += r.decode()
        _json = json.loads(
            _json, object_pairs_hook=OrderedDict)
        pipeline_json = _json['pipeline']
    else:
    # elif ('pathParameters' in event and event['pathParameters'] is not None and 'proxy' in event['pathParameters']) or ('authorizationToken' in event and event['authorizationToken'] == "incoming-client-token"):
        # TODO: Need to get the pipeline_json from the event...
        print("DSullivan: we have an API Gateway event")
        pipeline_json = None

    # Need to set matplotlib backend to 'Agg' before importing it elsewhere
    import matplotlib
    matplotlib.use('agg')
    from podpac import settings
    from podpac.core.node import Node
    from podpac.core.pipeline import Pipeline
    from podpac.core.coordinates import Coordinates
    from podpac.core.utils import JSONEncoder, _get_query_params_from_url
    import podpac.datalib

    # check if file exists
    if pipeline_json is not None:
        pipeline = Pipeline(definition=pipeline_json, do_write_output=False)
        filename = file_key.replace('.json', '.' + pipeline.output.format)
        filename = filename.replace(
            settings['S3_JSON_FOLDER'], settings['S3_OUTPUT_FOLDER'])
        try:
            s3.head_object(Bucket=bucket_name, Key=filename)
            # Object exists, so we don't have to recompute
            if not _json.get('force_compute', False):
                return
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                # It does not exist, so we should proceed
                pass
            else:
                # Something else has gone wrong... not handling this case.
                pass

    args = []
    kwargs = {}
    # if from S3 trigger
    if pipeline_json is not None:
    # if 'Records' in event and event['Records'][0]['eventSource'] == 'aws:s3':
        pipeline = Pipeline(definition=pipeline_json, do_write_output=False)
        coords = Coordinates.from_json(
            json.dumps(_json['coordinates'], indent=4, cls=JSONEncoder))
        format = pipeline.output.format

    # else from api gateway and it's a WMS/WCS request
    else:
    # elif ('pathParameters' in event and event['pathParameters'] is not None and 'proxy' in event['pathParameters']) or ('authorizationToken' in event and event['authorizationToken'] == "incoming-client-token"):
        print(_get_query_params_from_url(event['queryStringParameters']))
        coords = Coordinates.from_url(event['queryStringParameters'])
        pipeline = Node.from_url(event['queryStringParameters'])
        pipeline.do_write_output = False
        format = _get_query_params_from_url(event['queryStringParameters'])[
                                           'FORMAT'].split('/')[-1]
        if format in ['png', 'jpg', 'jpeg']:
            kwargs['return_base64'] = True

    output = pipeline.eval(coords)
    if ret_pipeline:
        return pipeline

    body = output.to_format(format, *args, **kwargs)
    if pipeline_json is not None:
        s3.put_object(Bucket=bucket_name,
                      Key=filename, Body=body)
    else:
        try:
            json.dumps(body)
        except Exception as e:
            print("AWS: body is not serializable, attempting to decode.")
            body = body.decode()
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "image/png"
        },
        "isBase64Encoded": True,
        "body": body
    }


if __name__ == '__main__':
    from podpac import settings
    # Need to authorize our s3 client when running locally.
    s3 = boto3.client('s3',
                      aws_access_key_id=settings['AWS_ACCESS_KEY_ID'],
                      aws_secret_access_key=settings['AWS_SECRET_ACCESS_KEY'],
                      region_name=settings['AWS_REGION_NAME']
                     )
    event = {
        "Records": [{
            "s3": {
                "object": {
                    "key": "json/SinCoords.json"
                },
                "bucket": {
                    "name": "podpac-s3",
                }
            }
        }]
    }

    example = handler(event, {}, get_deps=False)
    print(example)
