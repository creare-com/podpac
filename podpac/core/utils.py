"""
Utils Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import sys
import json
import functools
import importlib
from collections import OrderedDict
import logging

import traitlets as tl
import numpy as np

# create log for module
_log = logging.getLogger(__name__)

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

def cached_property(func):
    """Summary

    Parameters
    ----------
    func : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """

    @property
    @functools.wraps(func)
    def f(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        cache_name = '_cached_' + func.__name__
        if hasattr(self, cache_name):
            cache_val = getattr(self, cache_name)
        else:
            cache_val = None
        if cache_val is not None:
            return cache_val
        cache_val = func(self)
        setattr(self, cache_name, cache_val)
        return cache_val
    return f


def clear_cache(self, change, attrs):
    """Summary

    Parameters
    ----------
    change : TYPE
        Description
    attrs : TYPE
        Description
    """
    if (change['old'] is None and change['new'] is not None) or \
               np.any(np.array(change['old']) != np.array(change['new'])):
        for attr in attrs:
            setattr(self, '_cached_' + attr, None)


def get_settings_file(path=None):
    """Summary

    Parameters
    ----------
    path : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    if path is None:
        path = os.path.expanduser("~")
    file = os.path.join(path, '.podpac', 'settings.json')
    return file


def save_setting(key, value, path=None):
    """Summary

    Parameters
    ----------
    key : TYPE
        Description
    value : TYPE
        Description
    path : None, optional
        Description
    """
    file = get_settings_file(path)
    if not os.path.exists(file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        config = {}
    else:
        with open(file) as fid:
            try:
                config = json.load(fid)
            except:
                config = {}
    config[key] = value

    with open(file, 'w') as fid:
        json.dump(config, fid)


def load_setting(key, path=None):
    """Summary

    Parameters
    ----------
    key : TYPE
        Description
    path : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    file = get_settings_file(path)
    if not os.path.exists(file):
        return None

    with open(file) as fid:
        try:
            config = json.load(fid)
        except:
            return {}
    return config.get(key, None)


def trait_is_defined(obj, trait):
    """Utility method to determine if trait is defined on object without
    call to default (@tl.default)

    Parameters
    ----------
    object : object
        Class with traits
    trait : str
        Class property to investigate

    Returns
    -------
    bool
        True if the trait exists on the object and is defined
        False if the trait does not exist on the object or the trait is not defined
    """
    return obj.has_trait(trait) and trait in obj._trait_values

def optional_import(module_name, package=None, return_root=False):
    '''
    Import optional packages if present.

    Parameters
    -----------
    module_name: str
        The name of the module to import
    package: str, optional
        Default is None. The root package, in case module_name is relative
    return_root: bool
        Default if False. If True, will return the root package instead of the module

    Examples
    ----------
    >>> bar = optional_import('foo.bar')  # Returns bar
    >>> foo = optional_import('foo.bar', return_root=True)  # Returns foo

    Returns
    --------
    module
        The imported module if available. None otherwise.
    '''

    try:
        if return_root:
            module = importlib.__import__(module_name)
        else:
            module = importlib.import_module(module_name)
    except ImportError:
        module = None
    except AttributeError:
        try: # Python 2.7
            module = __import__(module_name)
        except ImportError:
            module = None
    return module

def create_logfile(filename='podpac.log',
                   level=logging.INFO,
                   format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'):
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
        for creating format
    
    Returns
    -------
    logging.Logger, logging.Handler, logging.Formatter
        Returns the constructed logger, handler, and formatter for the log file
    """
    # get logger for podpac module only
    log = logging.getLogger('podpac')
    log.setLevel(level)

    # create a file handler
    handler = logging.FileHandler(filename, 'a')

    # create a logging format
    # see https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    log.addHandler(handler)

    # insert log from utils into logfile
    _log.info('created logfile')

    return log, handler, formatter


if sys.version < '3.6':
    # for Python 2 and Python < 3.6 compatibility
    class OrderedDictTrait(tl.Dict):
        """ OrderedDict trait """

        default_value = OrderedDict()
        
        def validate(self, obj, value):
            if not isinstance(value, OrderedDict):
                raise tl.TraitError('...')
            super(OrderedDictTrait, self).validate(obj, value)
            return value

else:
    OrderedDictTrait = tl.Dict
