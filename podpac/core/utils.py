"""
Utils Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import json

import traitlets as tl
import numpy as np

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
        else: cache_val = None
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
        os.makedirs(os.path.dirname(file))
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
