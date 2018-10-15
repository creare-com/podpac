from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import os
import sys
import traceback
import urllib.parse as urllib
from collections import OrderedDict

import boto3

sys.path.append('/tmp')
sys.path.append(os.getcwd() + '/podpac/')

api_root = 'https://.'
# s3_bucket = 'creare-podpac-lambda'
s3 = boto3.client('s3')
deps = 'podpac_deps.zip'


def return_exception(e, event, context, pipeline=None):
    traceback.print_tb(e.__traceback__)
    contexts = str(context)
    try:
        contexts = json.dumps(contexts, sort_keys=True,
                              indent=2, separators=(',', ': '))
        if pipeline:
            pipeline = json.dumps(pipeline)
        else:
            pipeline = ''
    except Exception as e:
        pass

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'text/html',
            'Access-Control-Allow-Origin': '*',
        },
        'body': '<h1>Event</h1><br><br><br>' + str(event)
                + '<h1>Context</h1><br><br><br>' + str(context)
                + '<h1>Pipeline</h1><br><br><br>' + str(pipeline)
                + '<h1>Exception</h1><br><br><br>' + str(e),
        'isBase64Encoded': False,
    }


def handler(event, context, ret_pipeline=False):
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = urllib.unquote_plus(
        event['Records'][0]['s3']['object']['key'])
    try:
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
    except Exception as e:
        return return_exception(e, event, context)

    import numpy as np
    # Need to set matplotlib backend to 'Agg' before importing it elsewhere
    sys.path.append(os.getcwd() + '/matplotlib/')
    import matplotlib
    matplotlib.use('agg')
    from podpac.core.pipeline import Pipeline
    from podpac.core.coordinates import Coordinates
    try:
        pipeline = Pipeline(definition=pipeline_json)

        time = _json['coordinates']['TIME']
        bbox = _json['coordinates']['BBOX'].split(',')
        width = int(_json['coordinates']['WIDTH'])
        height = int(_json['coordinates']['WIDTH'])

        w, s, e, n = np.array(bbox, float)
        dwe = (w - e) / width / 2.
        dns = (n - s) / height / 2.
        coords = Coordinates.grid(time=np.datetime64(time), lat=(n - dns, s + dns, height),
                                  lon=(w + dwe, e - dwe, width), order=['lat', 'lon', 'time'])
        pipeline.execute(coords)
        if ret_pipeline:
            return pipeline
        pipeline.pipeline_output.write()
        s3.put_object(Bucket=bucket_name,
                      Key='output/' + pipeline.pipeline_output.name + '.' + pipeline.pipeline_output.format, Body=pipeline.pipeline_output.image)
        return img_response(pipeline.pipeline_output.image)
    except Exception as e:
        return return_exception(e, event, context, pipeline)


def img_response(img):
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'image/png',
            'Access-Control-Allow-Origin': '*',
        },
        'body': 'data:image/png;base64,' + str(img),
        'isBase64Encoded': True,
    }


if __name__ == '__main__':
    # Need to authorize our s3 client when running locally.
    from podpac import settings
    s3 = boto3.client('s3',
                      aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                      region_name=settings.AWS_REGION_NAME
                     )
    event = {
        "Records": [{
            "s3": {
                "object": {
                    "key": "json/example.json"
                },
                "bucket": {
                    "name": "podpac-s3",
                }
            }
        }]
    }

    example = handler(event, {})
    print(example)
