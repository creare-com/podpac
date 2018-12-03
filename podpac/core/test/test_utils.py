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
