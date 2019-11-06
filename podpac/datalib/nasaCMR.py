"""
Search using NASA CMR
"""

from __future__ import division, unicode_literals, print_function, absolute_import
import warnings
from copy import deepcopy
from collections import OrderedDict
from six import string_types

import requests
import os
import json
import logging

_logger = logging.getLogger(__name__)


import numpy as np
import traitlets as tl

CMR_URL = r"https://cmr.earthdata.nasa.gov/search/"


def get_collection_id(auth_session=None, short_name=None, keyword=None, **kwargs):
    """ Users NASA CMR to retrieve metadata about a data collection
    
    Parameters
    -----------
    auth_session: podpac.core.authentication.Session, optional
        An authenticated Earthdata login session
        
    short_name: str, optional
        The short name of the dataset
        
    keyword: str, optional
        Any keyword search parameters
        
    **kwargs: str, optional
        Any additional query parameters
        
    Returns
    ---------
    list
        A list of collection id's (ideally only one)
        
    Examples: 
    -----------
    >>> # This make the following request https://cmr.earthdata.nasa.gov/search/collections.json?short_name=SPL2SMAP_S
    >>> get_collection_id(short_name='SPL2SMAP_S')
    ['C1522341104-NSIDC_ECS']
    """

    base_url = CMR_URL + "collections.json?"
    if short_name is not None:
        kwargs["short_name"] = short_name
    if keyword is not None:
        kwargs["keyword"] = keyword

    query_string = "&".join([k + "=" + v for k, v in kwargs.items()])

    if auth_session is None:
        auth_session = requests

    json_ = _get_from_url(auth_session, base_url + query_string)

    pydict = json.loads(json_)

    entries = pydict["feed"]["entry"]
    if len(entries) > 1:
        _logger.warning("Found more than 1 entry for collection_id search")

    collection_id = [e["id"] for e in entries]

    return collection_id


def search_granule_json(auth_session=None, entry_map=None, **kwargs):
    """ Search for specific files from NASA CMR for a particular collection
    
    Parameters
    -----------
    auth_session: podpac.core.authentication.Session, optional
        An authenticated Earthdata login session
    
    entry_map: function
        A function applied to each individual entry. Could be used to filter out certain data in an entry
    
    **kwargs: dict
        Additional query string parameters. 
        At minimum the provider, provider_id, concept_id, collection_concept_id, short_name, version, or entry_title 
        need to be provided for a granule search. 
        
    Returns
    ---------
    list
        Entries for each granule in the collection based on the search terms
    """
    base_url = CMR_URL + "granules.json?"

    if "collection_id" not in kwargs and "short_name" not in kwargs:
        raise ValueError(
            "Need to provide either"
            " provider, provider_id, concept_id, collection_concept_id, short_name, version or entry_title"
            " for granule search."
        )

    if "page_size" not in kwargs:
        kwargs["page_size"] = "2000"

    if entry_map is None:
        entry_map = lambda x: x

    query_string = "&".join([k + "=" + str(v) for k, v in kwargs.items()])

    if auth_session is None:
        auth_session = requests

    url = base_url + query_string
    if "page_num" not in kwargs:
        entries = _get_all_granule_pages(auth_session, url, entry_map)
    else:
        granules = _get_from_url(auth_session, url)
        pydict = json.loads(granules)
        entries = list(map(entry_map, pydict["feed"]["entry"]))

    return entries


def _get_all_granule_pages(auth_session, url, entry_map, max_paging_depth=1000000):
    """ Helper function for searching through all pages for a collection.
    
    Parameters
    -----------
    auth_session: podpac.core.authentication.EarthDataSession
        Authenticated EDS session
    url: str
        URL to website
    entry_map: function
        Function for mapping the entries to a desired format
    max_paging_depth 
    """
    page_size = int([q for q in url.split("?")[1].split("&") if "page_size" in q][0].split("=")[1])
    max_pages = int(max_paging_depth / page_size)

    pydict = json.loads(_get_from_url(auth_session, url))
    entries = list(map(entry_map, pydict["feed"]["entry"]))

    for i in range(1, max_pages):
        page_url = url + "&page_num=%d" % (i + 1)
        page_entries = json.loads(_get_from_url(auth_session, page_url))["feed"]["entry"]
        if not page_entries:
            break
        entries.extend(list(map(entry_map, page_entries)))
    return entries


def _get_from_url(auth_session, url):
    """Helper function to get data from an url with error checking. 
    
    Parameters
    -----------
    auth_session: podpac.core.authentication.EarthDataSession
        Authenticated EDS session
    url: str
        URL to website
    """
    try:
        r = auth_session.get(url)
        if r.status_code != 200:
            _logger.warning(
                "Could not connect to {}, status code {}. \n *** Return Text *** \n {} \n *** End Return Text ***".format(
                    url, r.status_code, r.text
                )
            )

    except requests.ConnectionError as e:
        _logger.warning("Cannot connect to {}:".format(url) + str(e))
        r = None
    except RuntimeError as e:
        _logger.warning("Cannot authenticate to {}. Check credentials. Error was as follows:".format(url) + str(e))
    return r.text
