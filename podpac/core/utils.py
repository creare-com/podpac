"""
Utils Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import sys
import json
import datetime
import functools
import importlib
import logging
from collections import OrderedDict
from copy import deepcopy

try:
    import urllib.parse as urllib
except:  # Python 2.7
    import urlparse as urllib

from six import string_types
import lazy_import
import traitlets as tl
import numpy as np
import xarray as xr
import pandas as pd  # Core dependency of xarray

# Optional Imports
requests = lazy_import.lazy_module("requests")

# create log for module
_log = logging.getLogger(__name__)

import podpac
from . import settings


def common_doc(doc_dict):
    """ Decorator: replaces commond fields in a function docstring

    Parameters
    -----------
    doc_dict : dict
        Dictionary of parameters that will be used to format a doctring. e.g. func.__doc__.format(**doc_dict)
    """

    def _decorator(func):
        if func.__doc__ is None:
            return func

        func.__doc__ = func.__doc__.format(**doc_dict)
        return func

    return _decorator


def trait_is_defined(obj, trait_name):
    """Utility method to determine if trait is defined on object without
    call to default (@tl.default)

    Parameters
    ----------
    object : object
        Class with traits
    trait_name : str
        Class property to investigate

    Returns
    -------
    bool
        True if the trait exists on the object and is defined
        False if the trait does not exist on the object or the trait is not defined
    """
    return obj.has_trait(trait_name) and trait_name in obj._trait_values


def create_logfile(
    filename=settings.settings["LOG_FILE_PATH"],
    level=logging.INFO,
    format="[%(asctime)s] %(name)s.%(funcName)s[%(lineno)d] - %(levelname)s - %(message)s",
):
    """Convience method to create a log file that only logs
    podpac related messages

    Parameters
    ----------
    filename : str, optional
        Filename of the log file. Defaults to ``podpac.log``
    level : int, optional
        Log level to use (0 - 50). Defaults to ``logging.INFO`` (20)
        See https://docs.python.org/3/library/logging.html#levels
    format : str, optional
        String format for log messages.
        See https://docs.python.org/3/library/logging.html#logrecord-attributes
        for creating format. Default is:
        format='[%(asctime)s] %(name)s.%(funcName)s[%(lineno)d] - %(levelname)s - %(message)s'

    Returns
    -------
    logging.Logger, logging.Handler, logging.Formatter
        Returns the constructed logger, handler, and formatter for the log file
    """
    # get logger for podpac module only
    log = logging.getLogger("podpac")
    log.setLevel(level)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # create a file handler
    handler = logging.FileHandler(filename, "a")

    # create a logging format
    # see https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    log.addHandler(handler)

    # insert log from utils into logfile
    _log.info("Logging to file {}".format(filename))

    return log, handler, formatter


if sys.version < "3.6":
    # for Python 2 and Python < 3.6 compatibility
    class OrderedDictTrait(tl.Dict):
        """ OrderedDict trait """

        default_value = OrderedDict()

        def validate(self, obj, value):
            if value == {}:
                value = OrderedDict()
            elif not isinstance(value, OrderedDict):
                raise tl.TraitError(
                    "The '%s' trait of an %s instance must be an OrderedDict, but a value of %s %s was specified"
                    % (self.name, obj.__class__.__name__, value, type(value))
                )
            super(OrderedDictTrait, self).validate(obj, value)
            return value


else:
    OrderedDictTrait = tl.Dict


class ArrayTrait(tl.TraitType):
    """ A coercing numpy array trait. """

    def __init__(self, ndim=None, shape=None, dtype=None, dtypes=None, *args, **kwargs):
        if ndim is not None and shape is not None and len(shape) != ndim:
            raise ValueError("Incompatible ndim and shape (ndim=%d, shape=%s)" % (ndim, shape))
        if dtype is not None and not isinstance(dtype, type):
            if dtype not in np.typeDict:
                raise ValueError("Unknown dtype '%s'" % dtype)
            dtype = np.typeDict[dtype]
        self.ndim = ndim
        self.shape = shape
        self.dtype = dtype
        super(ArrayTrait, self).__init__(*args, **kwargs)

    def validate(self, obj, value):
        # coerce type
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        # ndim
        if self.ndim is not None and self.ndim != value.ndim:
            raise tl.TraitError(
                "The '%s' trait of an %s instance must have ndim %d, but a value with ndim %d was specified"
                % (self.name, obj.__class__.__name__, self.ndim, value.ndim)
            )

        # shape
        if self.shape is not None and self.shape != value.shape:
            raise tl.TraitError(
                "The '%s' trait of an %s instance must have shape %s, but a value %s with shape %s was specified"
                % (self.name, obj.__class__.__name__, self.shape, value, value.shape)
            )

        # dtype
        if self.dtype is not None:
            try:
                value = value.astype(self.dtype)
            except:
                raise tl.TraitError(
                    "The '%s' trait of an %s instance must have dtype %s, but a value with dtype %s was specified"
                    % (self.name, obj.__class__.__name__, self.dtype, value.dtype)
                )

        return value


class TupleTrait(tl.List):
    """ An instance of a Python tuple that accepts the 'trait' argument (like Set, List, and Dict). """

    def validate(self, obj, value):
        value = super(TupleTrait, self).validate(obj, value)
        return tuple(value)


class NodeTrait(tl.Instance):
    def __init__(self, *args, **kwargs):
        from podpac import Node as _Node

        super(NodeTrait, self).__init__(_Node, *args, **kwargs)

    def validate(self, obj, value):
        super(NodeTrait, self).validate(obj, value)
        if podpac.core.settings.settings["DEBUG"]:
            value = deepcopy(value)
        return value


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # podpac objects with definitions
        if isinstance(obj, (podpac.Coordinates, podpac.Node, podpac.data.Interpolation, podpac.core.style.Style)):
            return obj.definition

        # podpac Interpolator type
        if isinstance(obj, type) and obj in podpac.data.INTERPOLATORS:
            return obj().definition

        # pint Units
        if isinstance(obj, podpac.core.units.ureg.Unit):
            return str(obj)

        # datetime64
        if isinstance(obj, np.datetime64):
            return obj.astype(str)

        # timedelta64
        if isinstance(obj, np.timedelta64):
            return podpac.core.coordinates.utils.make_timedelta_string(obj)

        # datetime
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()

        # dataframe
        if isinstance(obj, pd.DataFrame):
            return obj.to_json()

        # numpy array
        if isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.datetime64):
                return obj.astype(str).tolist()
            if np.issubdtype(obj.dtype, np.timedelta64):
                return [podpac.core.coordinates.utils.make_timedelta_string(e) for e in obj]
            if np.issubdtype(obj.dtype, np.number):
                return obj.tolist()
            else:
                try:
                    # completely serialize the individual elements using the custom encoder
                    return json.loads(json.dumps([e for e in obj], cls=JSONEncoder))
                except TypeError as e:
                    raise TypeError("Cannot serialize numpy array\n%s" % e)

        # raise the TypeError
        return json.JSONEncoder.default(self, obj)


def is_json_serializable(obj, cls=json.JSONEncoder):
    try:
        json.dumps(obj, cls=cls)
    except:
        return False
    else:
        return True


def _get_param(params, key):
    if isinstance(params[key], list):
        return params[key][0]
    return params[key]


def _get_query_params_from_url(url):
    if isinstance(url, string_types):
        url = urllib.parse_qs(urllib.urlparse(url).query)

    # Capitalize the keywords for consistency
    params = {}
    for k in url:
        params[k.upper()] = url[k]

    return params


def _get_from_url(url, session=None):
    """Helper function to get data from an url with error checking.
    
    Parameters
    ----------
    url : str
        URL to website
    session : :class:`requests.Session`, optional
        Requests session to use when making the GET request to `url`
    
    Returns
    -------
    str
        Text response from request.
        See https://2.python-requests.org/en/master/api/#requests.Response.text
    """
    try:
        if session is None:
            r = requests.get(url)
        else:
            r = session.get(url)

        if r.status_code != 200:
            _log.warning(
                "Could not connect to {}, status code {}. \n *** Return Text *** \n {} \n *** End Return Text ***".format(
                    url, r.status_code, r.text
                )
            )

    except requests.ConnectionError as e:
        _log.warning("Cannot connect to {}:".format(url) + str(e))
        r = None
    except RuntimeError as e:
        _log.warning("Cannot authenticate to {}. Check credentials. Error was as follows:".format(url) + str(e))

    return r


def cached_property(*args, **kwargs):
    """
    Decorator that creates a property that is cached.

    Keyword Arguments
    -----------------
    use_cache_ctrl : bool
        If True, the property is cached using the Node cache_ctrl. If False, the property is only cached as a private
        attribute. Default False.

    Notes
    -----
    Podpac caching using the cache_ctrl will be unreliable if the property depends on any non-tagged traits.
    The property should only use node attrs (traits tagged with ``attr=True``).

    Examples
    --------

    >>> class MyNode(Node):
        # property that is recomputed every time
        @property
        def my_property(self):
            return 0

        # property is computed once for each object
        @cached_property
        def my_cached_property(self):
            return 1

        # property that is computed once and can be reused by other Nodes or sessions, depending on the cache_ctrl
        @cached_property(use_cache_ctrl=True)
        def my_persistent_cached_property(self):
            return 2
    """

    use_cache_ctrl = kwargs.pop("use_cache_ctrl", False)

    if args and (len(args) != 1 or not callable(args[0])):
        raise TypeError("cached_property decorator does not accept any positional arguments")

    if kwargs:
        raise TypeError("cached_property decorator does not accept keyword argument '%s'" % list(kwargs.keys())[0])

    def d(fn):
        key = "_podpac_cached_property_%s" % fn.__name__

        @property
        def wrapper(self):
            if hasattr(self, key):
                value = getattr(self, key)
            elif use_cache_ctrl and self.has_cache(key):
                value = self.get_cache(key)
                setattr(self, key, value)
            else:
                value = fn(self)
                setattr(self, key, value)
                if use_cache_ctrl:
                    self.put_cache(value, key)
            return value

        return wrapper

    if args:
        return d(args[0])
    else:
        return d


def ind2slice(Is):
    """ Convert boolean and integer index arrays to slices.

    Integer and boolean arrays are converted to slices that span the selected elements, but may include additional
    elements. If possible, the slices are stepped.

    Arguments
    ---------
    Is : tuple
        tuple of indices (slice, integer array, boolean array, or single integer)

    Returns
    -------
    Js : tuple
        tuple of slices
    """

    if isinstance(Is, tuple):
        return tuple(_ind2slice(I) for I in Is)
    else:
        return _ind2slice(Is)


def _ind2slice(I):
    # already a slice
    if isinstance(I, slice):
        return I

    # convert to numpy array
    I = np.atleast_1d(I)

    # convert boolean array to index array
    if I.dtype == bool:
        (I,) = np.where(I)

    # empty slice
    if I.size == 0:
        return slice(0, 0)

    # singleton
    if I.size == 1:
        return I[0]

    # stepped slice
    diff = np.diff(I)
    if diff.size and np.all(diff == diff[0]) and diff[0] != 0:
        return slice(I.min(), I.max() + diff[0], diff[0])

    # non-stepped slice
    return slice(I.min(), I.max() + 1)
