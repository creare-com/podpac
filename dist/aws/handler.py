from __future__ import division, unicode_literals, print_function, absolute_import

import base64
import json
import boto3
import subprocess
from io import BytesIO
from collections import OrderedDict
import sys, os
import urllib
sys.path.append('/tmp')

api_root = 'https://.'
s3_bucket = 'podpac-s3'
s3 = boto3.client('s3')
deps = 'podpac_deps.zip'

def return_exception(e, event, context, pipeline=None):
    events = str(event)
    contexts = str(context)
    try:
        events = json.dumps(event, sort_keys=True, indent=2, separators=(',', ': '))
        contexts = json.dumps(contexts, sort_keys=True, indent=2, separators=(',', ': '))
        if pipeline:
            pipeline = json.dumps(pipeline)
        else:
            pipeline = ''
    except:
        pass
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'text/html',
            'Access-Control-Allow-Origin': '*',
        },
        'body': '<h1>Event</h1><br><br><br>' + str(event)\
                + '<h1>Context</h1><br><br><br>' + str(context)
                + '<h1>Pipeline</h1><br><br><br>' + str(pipeline)
                + '<h1>Exception</h1><br><br><br>' + str(e),
        'isBase64Encoded': False,
    }    

def handler(event, context):

    # Get request arguments
    try:
        qs = event.get('queryStringParameters', event)
        service = urllib.unquote(qs['SERVICE'])
        version = urllib.unquote(qs['VERSION'])
        request = urllib.unquote(qs['REQUEST'])
        fmt = urllib.unquote(qs['FORMAT'])
        time = urllib.unquote(qs['TIME'])
        bbox = urllib.unquote(qs['BBOX']).split(',')
        crs = urllib.unquote(qs['CRS'])
        response_crs = urllib.unquote(qs['RESPONSE_CRS'])
        width = urllib.unquote(qs['WIDTH'])
        height = urllib.unquote(qs['HEIGHT'])
        params = urllib.unquote(qs['PARAMS'])
        pipeline = json.loads(params, object_pairs_hook=OrderedDict)['pipeline']
        try: 
            pipeline = json.loads(pipeline, object_pairs_hook=OrderedDict)
        except:
            pass
    except Exception as e:
        return return_exception(e, event, context)

    # Download additional dependencies ( we should do this in a thread )
    s3.download_file(s3_bucket, 'podpac/' + deps, '/tmp/' + deps)

    subprocess.call(['unzip', '/tmp/' + deps, '-d', '/tmp'])
    subprocess.call(['rm', '/tmp/' + deps])

    import numpy as np
    # Need to set matplotlib backend to 'Agg' before importing it elsewhere
    import matplotlib 
    matplotlib.use('agg')
    from podpac import Coordinate
    from podpac.core.pipeline import Pipeline
    try:
        pipeline = Pipeline(source=pipeline)
        
        w, s, e, n = np.array(bbox, float)     
        coord = Coordinate(lat=(n, s, height), lon=(w, e, width),
                           time=np.datetime64(time), 
                           order=['time', 'lat', 'lon'])
        pipeline.execute(coord)
        
        for output in pipeline.outputs:
            if output.format == 'png':
                break
        return img_response(output.image)
    except Exception as e:
        return return_exception(e, event, context, pipeline)        

def img_response(img):
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'image/png',
            'Access-Control-Allow-Origin': '*',
        },
        'body': 'data:image/png;base64,' + img,
        'isBase64Encoded': True,
    }

