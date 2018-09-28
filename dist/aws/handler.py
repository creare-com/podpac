from __future__ import division, unicode_literals, print_function, absolute_import

import base64
import json
import boto3
import subprocess
from io import BytesIO
from collections import OrderedDict
import sys, os
if sys.version_info.major == 2:
    import urllib
else:
    import urllib.parse as urllib
sys.path.append('/tmp')
sys.path.append(os.getcwd() + '/podpac/')

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

def handler(event, context, get_deps=False, ret_pipeline=False):

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
        width = int(urllib.unquote(qs['WIDTH']))
        height = int(urllib.unquote(qs['HEIGHT']))
        params = urllib.unquote(qs['PARAMS'])
        pipeline = json.loads(params, object_pairs_hook=OrderedDict)['pipeline']
        try: 
            pipeline = json.loads(pipeline, object_pairs_hook=OrderedDict)
        except:
            pass
    except Exception as e:
        return return_exception(e, event, context)

    # Download additional dependencies ( we should do this in a thread )
    if get_deps:
        s3.download_file(s3_bucket, 'podpac/' + deps, '/tmp/' + deps)

        subprocess.call(['unzip', '/tmp/' + deps, '-d', '/tmp'])
        subprocess.call(['rm', '/tmp/' + deps])

    import numpy as np
    # Need to set matplotlib backend to 'Agg' before importing it elsewhere
    #import matplotlib 
    #matplotlib.use('agg')
    #from podpac import coordinate
    import podpac
    from podpac.core.pipeline import Pipeline
    try:
        pipeline = Pipeline(definition=pipeline)
        
        w, s, e, n = np.array(bbox, float)     
        dwe = (w-e)/width/2.
        dns = (n-s)/height/2.
        coord = Coordinate(lat=(n-dns, s+dns, height),
                           lon=(w+dwe, e-dwe, width),
                           time=np.datetime64(time), 
                           order=['time', 'lat', 'lon'])
        pipeline.execute(coord)
        if ret_pipeline:
            return pipeline
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
        'body': 'data:image/png;base64,' + img,
        'isBase64Encoded': True,
    }

if __name__ == '__main__' and len(sys.argv) >= 2 and sys.argv[1] == 'test':
    event = {
        "SERVICE": "WMS",                                                                 
        "VERSION": "1.0.0",                                                               
        "REQUEST": "GetCoverage",                                                         
        "FORMAT": "image/png",                                                            
        "COVERAGE": "PIPELINE",                                                           
        "TIME": "2017-08-08T12:00:00",
        "BBOX": "-77.1,39.0,-76.8,39.3",
        "CRS": "EPSG:4326",
        "RESPONSE_CRS": "EPSG:4326",
        "WIDTH": "256",
        "HEIGHT": "256",
        "PARAMS": "%7B%22pipeline%22%3A%22%7B%5Cn%20%20%5C%22nodes%5C%22%3A%20%7B%5Cn%20%20%20%20%20%20%5C%22sm%5C%22%3A%20%7B%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22node%5C%22%3A%20%5C%22datalib.smap.SMAP%5C%22%2C%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22attrs%5C%22%3A%20%7B%5Cn%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5C%22product%5C%22%3A%20%5C%22SPL4SMAU.003%5C%22%2C%5Cn%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5C%22interpolation%5C%22%3A%20%5C%22nearest%5C%22%5Cn%20%20%20%20%20%20%20%20%20%20%7D%5Cn%20%20%20%20%20%20%7D%5Cn%20%20%7D%2C%5Cn%20%20%5C%22outputs%5C%22%3A%20%5B%5Cn%20%20%20%20%20%20%7B%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22mode%5C%22%3A%20%5C%22image%5C%22%2C%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22format%5C%22%3A%20%5C%22png%5C%22%2C%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22nodes%5C%22%3A%20%5B%5C%22sm%5C%22%5D%5Cn%20%20%20%20%20%20%7D%20%20%20%5Cn%20%20%5D%5Cn%7D%5Cn%22%7D" 
        }
    # %7B%22pipeline%22%3A%22%7B%5Cn%20%20%5C%22nodes%5C%22%3A%20%7B%5Cn%20%20%20%20%20%20%5C%22sm%5C%22%3A%20%7B%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22node%5C%22%3A%20%5C%22datalib.smap.SMAP%5C%22%2C%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22attrs%5C%22%3A%20%7B%5Cn%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5C%22product%5C%22%3A%20%5C%22SPL4SMAU.003%5C%22%2C%5Cn%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5C%22interpolation%5C%22%3A%20%5C%22nearest%5C%22%5Cn%20%20%20%20%20%20%20%20%20%20%7D%5Cn%20%20%20%20%20%20%7D%5Cn%20%20%7D%2C%5Cn%20%20%5C%22outputs%5C%22%3A%20%5B%5Cn%20%20%20%20%20%20%7B%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22mode%5C%22%3A%20%5C%22image%5C%22%2C%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22format%5C%22%3A%20%5C%22png%5C%22%2C%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22nodes%5C%22%3A%20%5B%5C%22sm%5C%22%5D%5Cn%20%20%20%20%20%20%7D%20%20%20%5Cn%20%20%5D%5Cn%7D%5Cn%22%7D
    # -77.1,39.0,-76.8,39.3
    # https://22d3a0pwlf.execute-api.us-east-1.amazonaws.com/prod/?SERVICE=WMS&VERSION=1.0.0&REQUEST=GetCoverage&FORMAT=image%2Fpng&COVERAGE=PIPELINE&TIME=2017-08-08T12%3A00%3A00&BBOX=-77.1%2C39.0%2C-76.8%2C39.3&CRS=EPSG:4326&RESPONSE_CRS=EPSG:4326&WIDTH=256&HEIGHT=256&PARAMS=%7B%22pipeline%22%3A%22%7B%5Cn%20%20%5C%22nodes%5C%22%3A%20%7B%5Cn%20%20%20%20%20%20%5C%22sm%5C%22%3A%20%7B%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22node%5C%22%3A%20%5C%22datalib.smap.SMAP%5C%22%2C%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22attrs%5C%22%3A%20%7B%5Cn%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5C%22product%5C%22%3A%20%5C%22SPL4SMAU.003%5C%22%2C%5Cn%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5C%22interpolation%5C%22%3A%20%5C%22bilinear%5C%22%5Cn%20%20%20%20%20%20%20%20%20%20%7D%5Cn%20%20%20%20%20%20%7D%5Cn%20%20%7D%2C%5Cn%20%20%5C%22outputs%5C%22%3A%20%5B%5Cn%20%20%20%20%20%20%7B%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22mode%5C%22%3A%20%5C%22image%5C%22%2C%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22format%5C%22%3A%20%5C%22png%5C%22%2C%5Cn%20%20%20%20%20%20%20%20%20%20%5C%22nodes%5C%22%3A%20%5B%5C%22sm%5C%22%5D%5Cn%20%20%20%20%20%20%7D%20%20%20%5Cn%20%20%5D%5Cn%7D%5Cn%22%7D
    pipeline = handler(event, {}, get_deps=False, ret_pipeline=True)


