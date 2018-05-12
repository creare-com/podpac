"""
Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import json

import traitlets as tl
import numpy as np


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

if __name__ == "__main__":

    class Dum(tl.HasTraits):
        @cached_property
        def test(self):
            print("Calculating Test")
            return 'test_prints' + str(self.lala)

        lala = tl.Int(0)

        @tl.observe('lala')
        def lalaobs(self, change):
            clear_cache(self, change, ['test'])

    d = Dum()
    print(d.test, d.test)
    d.lala = 10
    print(d.test, d.test)
