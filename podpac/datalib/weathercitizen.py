"""
Weather Citizen

Crowd sourced environmental observations from mobile devices (https://weathercitizen.org)

- Documentation: https://weathercitizen.org/docs
- API: https://api.weathercitizen.org

Requires

- requests: `pip install requests`
- pandas: `pip install pandas`

Optionally:

- read_protobuf: `pip install read-protobuf` - decodes sensor burst media files
"""

import json
from datetime import datetime
from datetime import timedelta
import logging

import traitlets as tl
import pandas as pd
from pandas.io.json import json_normalize
import requests

from podpac.data import DataSource
from podpac.core.data.datasource import COMMON_DATA_DOC
from podpac.core.utils import common_doc, trait_is_defined
from podpac.core.coordinates import Coordinates, UniformCoordinates1d, ArrayCoordinates1d, StackedCoordinates


URL = "https://api.weathercitizen.org/"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"  # always UTC  (ISO 8601 / RFC 3339 format)

# create log for module
_logger = logging.getLogger(__name__)


class WeatherCitizen(DataSource):
    """DataSource to handle WeatherCitizen data
       
    Attributes
    ----------
    source : str
        URL of the WeatherCitizen server.
        Defaults to "https://api.WeatherCitizen.org"
    data_key : str, int
        data key of interest, default `pressure`
    uuid : str, list(str), options
        String or list of strings to filter data by uuid
    device : str, list(str), ObjectId, list(ObjectId), optional
        String or list of strings to filter data by device object id
    version : string, list(str), optional
        String or list of strings to filter data to filter data by WeatherCitizen version
    query : dict, optional
        Arbitrary pymongo query to apply to data.
        Note that certain fields in this query may be overriden if other keyword arguments are specified
    quiet : bool, optional
        Don't display log messages or progress
    """

    source = tl.Unicode(allow_none=True, default_value=URL).tag(attr=True)
    data_key = tl.Unicode(allow_none=True, default_value="pressure").tag(attr=True)
    uuid = tl.Unicode(allow_none=True, default_value=None).tag(attr=True)
    device = tl.Unicode(allow_none=True, default_value=None).tag(attr=True)
    version = tl.Unicode(allow_none=True, default_value=None).tag(attr=True)
    query = tl.Unicode(allow_none=True, default_value=None).tag(attr=True)
    quiet = tl.Bool(allow_none=True, default_value=False).tag(attr=True)

    @common_doc(COMMON_DATA_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        """

        # return Coordinates([stacked], **coords.properties)
        pass

    @common_doc(COMMON_DATA_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """

        # return self.create_output_array(coordinates, data=data)
        pass


##############
# Standalone functions
##############
def get(
    collection="geosensors",
    start_time=None,
    end_time=None,
    box=None,
    near=None,
    uuid=None,
    device=None,
    version=None,
    query=None,
    quiet=False,
    dry_run=False,
):
    """Get documents from the server for devices in a timerange
    
    Parameters
    ----------
    collection : str, list(str)
        Collection(s) to query
    start_time : str, datetime, optional
        String or datetime for start of timerange (>=).
        Defaults to 1 hour ago.
        This input must be compatible with pandas `pd.to_datetime(start_time, utc=True)`
        Input assumes UTC by default, but will recognize timezone string EDT, UTC, etc. For example "2019-09-01 08:00 EDT"
    end_time : str, datetime, optional
        Same as `start_time` but specifies end of time range (<).
        Defaults to now.
    box : list(list(float)), optional
        Geo bounding box described as 2-d array of bottom-left and top-right corners.
        If specified, `near` will be ignored.
        Contents: [[ <bottom left coordinates> ], [ <upper right coordinates> ]]
        For example: [[-83, 36], [-81, 34]]
    near : tuple([float, float], int), optional
        Geo bounding box described as 2-d near with a center point and a radius (km) from center point.
        This input will be ignored if box is defined.
        Contents: ([<lon>, <lat>], <radius in km>)
        For example: ([-72.544655, 40.932559], 16000)
    uuid : str, list(str), options
        String or list of strings to filter data by uuid
    device : str, list(str), ObjectId, list(ObjectId), optional
        String or list of strings to filter data by device object id
    version : string, list(str), optional
        String or list of strings to filter data to filter data by WeatherCitizen version
    query : dict, optional
        Arbitrary pymongo query to apply to data.
        Note that certain fields in this query may be overriden if other keyword arguments are specified
    quiet : bool, optional
        Don't display log messages or progress
    dry_run : bool, optional
        Return urls of queries instead of the actual query.
        Returns a list of str with urls for each collections.
        Defaults to False.
    
    Returns
    -------
    list
        List of items from server matching query.
        If `dry_run` is True, returns a list or url strings for query.
    """

    # always make collection a list
    if isinstance(collection, str):
        collection = [collection]

    # get query string for each collection in list
    query_strs = [
        _build_query(
            collection=coll,
            start_time=start_time,
            end_time=end_time,
            box=box,
            near=near,
            uuid=uuid,
            device=device,
            version=version,
            query=query,
        )
        for coll in collection
    ]

    # dry run
    if dry_run:
        return query_strs

    if not quiet:
        print("Querying WeatherCitizen API")

    # start query at page 0 with no items
    # iterate through collections aggregating items
    items = []
    for query_str in query_strs:
        items += _get(query_str, quiet=quiet)

    if not quiet:
        print("\r")
        print("Downloaded {} records".format(len(items)))

    return items


def get_record(collection, obj_id, url=URL):
    """Get a single record from a collection by obj_id
    
    Parameters
    ----------
    collection : str
        Collection name
    obj_id : str
        Object id
    """

    # check url
    if url[-1] != "/":
        url = "{}/".format(url)

    # query the server
    r = requests.get(url + collection + "/" + obj_id)

    if r.status_code != 200:
        raise ValueError("Failed to query the server with status {}.\n\nResponse:\n {}".format(r.status_code, r.text))

    return r.json()


def get_file(media, save=False, output_path=None, quiet=False):
    """Get media file
    
    Parameters
    ----------
    media : str, dict
        Media record or media record object id in the media or geomedia collections.
    save : bool, optional
        Save to file
    output_path : None, optional
        If save is True, output the file to different file path
    quiet : bool, optional
        Don't display log messages or progress
    
    Returns
    -------
    bytes
        If output_path is None, returns raw file content as bytes
    
    Raises
    ------
    ValueError
        Description
    """

    if isinstance(media, str):
        media_id = media
    elif isinstance(media, dict):
        media_id = media["_id"]

    try:
        record = get_record("media", media_id)
    except ValueError:
        try:
            record = get_record("geomedia", media_id)

        except ValueError:
            raise ValueError("Media id {} not found in the database".format(media_id))

    # get file
    r = requests.get(record["file"]["url"])

    if r.status_code != 200:
        raise ValueError(
            "Failed to download binary data with status code {}.\n\nResponse:\n {}".format(r.status_code, r.text)
        )

    # save to file if output_path is not None
    if save:
        if output_path is None:
            output_path = record["properties"]["filename"]
        with open(output_path, "wb") as f:
            f.write(r.content)
    else:
        return r.content


def read_sensorburst(media):
    """Download and read sensorburst records.
    
    Requires: 
    - read-protobuf: `pip install read-protobuf`
    - sensorburst_pb2: Download from https://api.weathercitizen.org/static/sensorburst_pb2.py
        - Once downloaded, put this file in the directory as your analysis
    
    Parameters
    ----------
    media : str, dict, list of str, list of dict
        Media record(s) or media record object id(s) in the media or geomedia collections.
    
    Returns
    -------
    pd.DataFrame
        Returns pandas dataframe of records
    """

    from read_protobuf import read_protobuf
    import sensorburst_pb2

    if isinstance(media, (str, dict)):
        media = [media]

    # get pb content
    pbs = [get_file(m) for m in media]

    # initialize protobuf object
    Burst = sensorburst_pb2.Burst()

    # get the first dataframe
    df = read_protobuf(pbs[0], Burst)

    # append later dataframes
    if len(pbs) > 1:
        for pb in pbs[1:]:
            df = df.append(read_protobuf(pb, Burst), sort=False)

    return df


def to_dataframe(items):
    """Create normalized dataframe from records
    
    Parameters
    ----------
    items : list of dict
        Record items returned from `get()`
    """
    df = json_normalize(items)

    return df


def to_csv(items, filename="weathercitizen-data.csv"):
    """Convert items to CSV output
    
    Parameters
    ----------
    items : list of dict
        Record items returned from `get()`
    """

    df = to_dataframe(items)

    df.to_csv(filename)


def update_progress(current, total):
    """
    Parameters
    ----------
    current : int, float
        current number
    total : int, floar
        total number
    """

    if total == 0:
        return

    progress = float(current / total)
    bar_length = 20
    block = int(round(bar_length * progress))
    text = "Progress: |{0}| [{1} / {2}]".format("#" * block + " " * (bar_length - block), current, total)

    print("\r", text, end="")


def _build_query(
    collection="geosensors",
    start_time=None,
    end_time=None,
    box=None,
    near=None,
    uuid=None,
    device=None,
    version=None,
    query=None,
):
    """Build a query string for a single collection.
    See :func:`get` for type definitions of each input
    
    Returns
    -------
    string
        query string
    """

    if query is None:
        query = {}

    # filter by time
    # default to 1 hour ago
    one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).strftime(DATE_FORMAT)
    if start_time is not None:
        start_time = pd.to_datetime(start_time, utc=True, infer_datetime_format=True).strftime(DATE_FORMAT)
        query["properties.time"] = {"$gte": start_time}
    else:
        query["properties.time"] = {"$gte": one_hour_ago}

    # default to now
    if end_time is not None:
        end_time = pd.to_datetime(end_time, utc=True, infer_datetime_format=True).strftime(DATE_FORMAT)
        query["properties.time"]["$lte"] = end_time

    # geo bounding box
    if box is not None:
        if len(box) != 2:
            raise ValueError("box parameter must be a list of length 2")

        query["geometry"] = {"$geoWithin": {"$box": box}}

    # geo bounding circle
    if near is not None:
        if len(near) != 2 or not isinstance(near, tuple):
            raise ValueError("near parameter must be a tuple of length 2")

        query["geometry"] = {"$near": {"$geometry": {"type": "Point", "coordinates": near[0]}, "$maxDistance": near[1]}}

    # specify uuid
    if uuid is not None:
        if isinstance(uuid, str):
            query["properties.uuid"] = uuid
        elif isinstance(uuid, list):
            query["properties.uuid"] = {"$in": uuid}

    # specify device
    if device is not None:
        if isinstance(device, str):
            query["properties.device"] = device
        elif isinstance(device, list):
            query["properties.device"] = {"$in": device}

    # specify version
    if version is not None:
        if isinstance(version, str):
            query["version"] = version
        elif isinstance(version, list):
            query["version"] = {"$in": version}

    # add collection to query string
    query_str = "{}?where={}".format(collection, json.dumps(query))

    return query_str


def _get(query, items=None, url=URL, quiet=False):
    """Internal method to query API.
    See `get` for interface.
    
    Parameters
    ----------
    query : dict, str
        query dict or string
        if dict, it will be converted into a string with json.dumps()
    items : list, optional
        aggregated items as this method is recursively called. Defaults to [].
    url : str, optional
        API url. Defaults to module URL.
    quiet : bool, optional
        Don't display log messages or progress
    
    Returns
    -------
    list
    
    Raises
    ------
    ValueError
        Description
    """

    # if items are none, set to []
    if items is None:
        items = []

    # check url
    if url[-1] != "/":
        url = "{}/".format(url)

    # query the server
    r = requests.get(url + query)

    if r.status_code != 200:
        raise ValueError("Failed to query the server with status {}.\n\nResponse:\n {}".format(r.status_code, r.text))

    # get json out of response
    resp = r.json()
    if len(resp["_items"]):

        # show progress
        if not quiet:
            current_page = resp["_meta"]["page"]
            total_pages = round(resp["_meta"]["total"] / resp["_meta"]["max_results"])
            update_progress(current_page, total_pages)

        # append items
        items += resp["_items"]

        # get next set, if in links
        if "_links" in resp and "next" in resp["_links"]:
            return _get(resp["_links"]["next"]["href"], items=items)
        else:
            return items
    else:
        return items
