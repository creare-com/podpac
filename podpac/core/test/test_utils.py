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
    def tmp_dir_path(self):
        basedir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(basedir,'.__tmp__')
        return path

    def make_settings_tmp_dir(self):
        path = self.tmp_dir_path()
        os.mkdir(path) # intentionally fails if this folder already exists as it will be deleted
        return path

    def tear_down_tmp_settings(self):
        path = self.tmp_dir_path()
        os.remove(os.path.join(path, '.podpac', 'settings.json'))
        os.rmdir(os.path.join(path, '.podpac')) # intentionally fails if anything else is in this folder
        os.rmdir(path) # intentionally fails if anything else is in this folder

    def test_settings_file_defaults_to_home_dir(self):
        file_path = ut.get_settings_file()
        path = os.path.expanduser("~")
        assert file_path == os.path.join(path, '.podpac', 'settings.json')

    def test_single_saved_setting_persists(self):
        path = self.make_settings_tmp_dir()

        key = "key"
        value = "value"
        ut.save_setting(key, value, path)
        saved_value = ut.load_setting(key, path)
        assert value == saved_value

        self.tear_down_tmp_settings()

    def test_multiple_saved_settings_persist(self):
        path = self.make_settings_tmp_dir()

        key1 = "key1"
        value1 = "value1"
        ut.save_setting(key1, value1, path)

        key2 = "key2"
        value2 = "value2"
        ut.save_setting(key2, value2, path)

        saved_value1 = ut.load_setting(key1, path)
        assert value1 == saved_value1

        saved_value2 = ut.load_setting(key2, path)
        assert value2 == saved_value2

        self.tear_down_tmp_settings()

    def test_loading_saved_setting_if_file_not_exists_returns_none(self):
        path = self.tmp_dir_path()

        key = "key1"
        saved_value = ut.load_setting(key, path)

        assert saved_value is None

    def test_misconfigured_settings_file_will_return_empty_dictionary(self):
        path = self.make_settings_tmp_dir()

        key = "key"
        value = "value"
        ut.save_setting(key, value, path)

        with open(ut.get_settings_file(path), 'w') as f:
            f.write("not proper json")

        saved_value = ut.load_setting(key, path)
        assert isinstance(saved_value,dict) and len(saved_value) == 0

        self.tear_down_tmp_settings()

    def test_misconfigured_settings_gets_replaced_on_save_setting(self):
        path = self.make_settings_tmp_dir()

        key = "key"
        value = "value"
        ut.save_setting(key, value, path)

        with open(ut.get_settings_file(path), 'w') as f:
            f.write("not proper json")

        ut.save_setting(key, value, path)

        saved_value = ut.load_setting(key, path)
        assert saved_value == value

        self.tear_down_tmp_settings()

class TestCommonDocs(object):
    def test_common_docs_does_not_affect_anonymous_functions(self):
        f = lambda x: x
        f2 = ut.common_doc({"key": "value"})(f)
        assert f(42) == f2(42)
        assert f.__doc__ is None


# TODO: add log testing
class TestLog(object):
    def test_create_log(self):
        pass
