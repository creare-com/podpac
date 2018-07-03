from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np
import traitlets as tl
import os

import podpac.core.utils as ut

class Foo(tl.HasTraits):
    @ut.cached_property
    def test(self):
        print("Calculating Test")
        return 'test_' + str(self.bar)

    bar = tl.Int(0)

    @tl.observe('bar')
    def barobs(self, change):
        ut.clear_cache(self, change, ['test'])

class TestCachedProperty(object):
    def test_changing_observerd_variable_in_foo_clears_cache(self):
        foo = Foo()
        assert foo.test == 'test_0' # uses default value of `bar`
        assert foo.test == 'test_0' # doesn't change with multiple calls
        foo.bar = 10
        assert foo.test == 'test_10' # uses new value of `bar`
        assert foo.test == 'test_10' # doesn't change with multiple calls

class TestSettingsFile(object):
    def test_settings_file_defaults_to_home_dir(self):
        file_path = ut.get_settings_file()
        path = os.path.expanduser("~")
        assert file_path == os.path.join(path, '.podpac', 'settings.json')

    def test_saved_setting_persists(self):
        basedir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(basedir,'.__tmp__')
        os.mkdir(path) # intentionally fails if this folder already exists as it will be deleted

        key = "key"
        value = "value"
        ut.save_setting(key, value, path)
        saved_value = ut.load_setting(key, path)
        assert value == saved_value

        os.remove(os.path.join(path, '.podpac', 'settings.json'))
        os.rmdir(os.path.join(path, '.podpac')) # intentionally fails if anything else is in this folder
        os.rmdir(path) # intentionally fails if anything else is in this folder

