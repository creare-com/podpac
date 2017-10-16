from __future__ import division, unicode_literals, print_function, absolute_import

import base64
import json
import boto3
import subprocess
from io import BytesIO
import sys, os
import urllib
sys.path.append('/tmp')

api_root = 'https://.'
s3_bucket = 'podpac-s3'
s3 = boto3.client('s3')
deps = 'podpac_deps.zip'

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
        pipeline = json.loads(params)['pipeline']
    except Exception as e:
        events = str(event)
        contexts = str(context)
        try:
            events = json.dumps(event, sort_keys=True, indent=2, separators=(',', ': '))
            contexts = json.dumps(contexts, sort_keys=True, indent=2, separators=(',', ': '))
        except:
            pass

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'text/html'
            },
            'body': '<h1>Event</h1><br><br><br>' + str(event)\
                    + '<h1>Context</h1><br><br><br>' str(context),
            'isBase64Encoded': False,
        }

    # Download additional dependencies ( we should do this in a thread )
    s3.download_file(s3_bucket, 'podpac/' + deps, '/tmp/' + deps)

    subprocess.call(['unzip', '/tmp/' + deps, '-d', '/tmp'])
    subprocess.call(['rm', '/tmp/' + deps])

    import requests
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import cm
    from matplotlib.image import imsave
    import numpy as np
    import podpac
    from podpac.core.pipeline import pipeline

    pipeline = Pipeline(source=pipeline)
    
    w, s, e, n = np.array(bbox, float)     
    coord = podpac.Coordinate(lat=(n, s, height), lon=(w, e, width),
                              time=np.datetime64(time), 
                              order=['time', 'lat', 'lon'])
    pipeline.execute(coord)

    if 'png' in fmt:
        o = pipeline.outputs[0].node.output
        c = (o.data - np.min(o.data)) / (np.max(o.data) - np.min(o.data) + 1e-16)
        i = cm.viridis(c, bytes=True)
        im_data = BytesIO()
        imsave(im_data, i, format='png')
        im_data.seek(0)
        img = base64.b64encode(im_data.getvalue())
    return o 

def img_response(img):
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'image/png',
            'Access-Control-Allow-Origin': '*',
        },
        'body': img,
        'isBase64Encoded': True,
    }

