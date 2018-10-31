from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import os
import sys
import urllib.parse as urllib
from collections import OrderedDict

import boto3

import _pickle as cPickle

sys.path.append('/tmp')
sys.path.append(os.getcwd() + '/podpac/')

s3 = boto3.client('s3')


def handler(event, context, ret_pipeline=False):
    bucket_name = event['Records'][0]['s3']['bucket']['name']
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

    # Need to set matplotlib backend to 'Agg' before importing it elsewhere
    sys.path.append(os.getcwd() + '/matplotlib/')
    import matplotlib
    matplotlib.use('agg')
    from podpac import settings
    from podpac.core.pipeline import Pipeline
    from podpac.core.coordinates import Coordinates
    pipeline = Pipeline(definition=pipeline_json, do_write_output=False)
    coords = Coordinates.from_json(
        json.dumps(_json['coordinates'], indent=4))
    pipeline.eval(coords)
    if ret_pipeline:
        return pipeline

    # NOTE: This could also just be file_key from above, replacing '.json' with '.pickle'
    filename = '%s%s_%s_%s.%s' % (
        settings.S3_OUTPUT_FOLDER,
        pipeline.pipeline_output.name,
        pipeline.hash,
        coords.hash,
        pipeline.pipeline_output.format)

    body = cPickle.dumps(pipeline._output)
    s3.put_object(Bucket=bucket_name,
                  Key=filename, Body=body)
    return


if __name__ == '__main__':
    from podpac import settings
    # Need to authorize our s3 client when running locally.
    s3 = boto3.client('s3',
                      aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                      region_name=settings.AWS_REGION_NAME
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

    example = handler(event, {})
    print(example)
