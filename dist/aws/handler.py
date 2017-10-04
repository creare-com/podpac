from __future__ import division, unicode_literals, print_function, absolute_import

import base64
import json
import boto3
import subprocess
from io import BytesIO
import sys, os
sys.path.append('/tmp')

api_root = 'https://.'
s3_bucket = 'podpac-s3'
s3 = boto3.client('s3')
deps = 'podpac_deps.zip'

def handler(event, context):


    #return {
    #    'statusCode': 200,
    #    'headers': {
    #        'Content-Type': 'text/html'
    #    },
    #    'body': str(event) + "AAAAAAAAAAAAA" + str(context),
    #    'isBase64Encoded': False,
    #}

    # Download additional dependencies
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
    from podpac.core.algorithm import algorithm

    # Get request arguments
    try:
        eqn = event.get('queryStringParameters', event).get('eqn', 'A + B')
    except:
        eqn = 'A' 
    #coords = body['coords']
    node = algorithm.Arithmetic(A=algorithm.SinCoords(), B=algorithm.SinCoords())
    #coord = podpac.Coordinate(coords)
    coord = podpac.Coordinate(lat=(90, -90, 1.), lon=(-180, 180, 1.),
                              order=['lat', 'lon'])
    o = node.execute(coord, params={'eqn': eqn})
    c = (o.data - np.min(o.data)) / (np.max(o.data) - np.min(o.data) + 1e-16)
    i = cm.viridis(c, bytes=True)
    im_data = BytesIO()
    imsave(im_data, i, format='png')
    im_data.seek(0)
    img = base64.b64encode(im_data.getvalue())
    return img_response(img)

def img_response(img, eqn):
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'image/png'
        },
        'body': img,
        'isBase64Encoded': True,
    }

