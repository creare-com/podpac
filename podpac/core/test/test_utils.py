from __future__ import division, unicode_literals, print_function, absolute_import

import os
from collections import OrderedDict

import pytest
import numpy as np
import traitlets as tl
import sys
import podpac.core.utils as ut

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

class TestOrderedDictTrait(object):
    def test(self):
        class MyClass(tl.HasTraits):
            d = ut.OrderedDictTrait()

        m = MyClass(d=OrderedDict([('a', 1)]))

        with pytest.raises(tl.TraitError):
            MyClass(d=[])

    @pytest.mark.skipif(sys.version < '3.6', reason="python < 3.6")
    def test_dict_python36(self):
        class MyClass(tl.HasTraits):
            d = ut.OrderedDictTrait()

        m = MyClass(d={'a': 1})

    @pytest.mark.skipif(sys.version >= '3.6', reason='python >= 3.6')
    def test_dict_python2(self):
        class MyClass(tl.HasTraits):
            d = ut.OrderedDictTrait()

        with pytest.raises(tl.TraitError):
            m = MyClass(d={'a': 1})
        
        # empty is okay, will be converted
        m = MyClass(d={})

class TestArrayTrait(object):
    def test(self):
        class MyClass(tl.HasTraits):
            a = ut.ArrayTrait()

        # basic usage
        o = MyClass(a=np.array([0, 4]))
        assert isinstance(o.a, np.ndarray)
        np.testing.assert_equal(o.a, [0, 4])
        
        # coerce
        o = MyClass(a=[0, 4])
        assert isinstance(o.a, np.ndarray)
        np.testing.assert_equal(o.a, [0, 4])

        # invalid
        # As of numpy 0.16, no longer raises an error
        #with pytest.raises(tl.TraitError):
            #MyClass(a=[0, [4, 5]])

    def test_ndim(self):
        class MyClass(tl.HasTraits):
            a = ut.ArrayTrait(ndim=2)

        MyClass(a=np.array([[0, 4]]))
        MyClass(a=[[0, 4]])
        
        # invalid
        with pytest.raises(tl.TraitError):
            MyClass(a=[4, 5])

    def test_shape(self):
        class MyClass(tl.HasTraits):
            a = ut.ArrayTrait(shape=(2, 2))

        MyClass(a=np.array([[0, 1], [2, 3]]))
        MyClass(a=[[0, 1], [2, 3]])
        
        # invalid
        with pytest.raises(tl.TraitError):
            MyClass(a=np.array([0, 1, 2, 3]))

    def test_dtype(self):
        class MyClass(tl.HasTraits):
            a = ut.ArrayTrait(dtype=float)

        m = MyClass(a=np.array([0.0, 1.0]))
        assert m.a.dtype == float
        
        m = MyClass(a=[0.0, 1.0])
        assert m.a.dtype == float
        
        # astype
        m = MyClass(a=[0, 1])
        assert m.a.dtype == float

        # invalid
        with pytest.raises(tl.TraitError):
            MyClass(a=np.array(['a', 'b']))

    def test_args(self):
        # shape and ndim must match
        t = ut.ArrayTrait(ndim=2, shape=(2, 2))
        
        with pytest.raises(ValueError):
            ut.ArrayTrait(ndim=1, shape=(2, 2))

        # dtype lookup
        t = ut.ArrayTrait(dtype='datetime64')
        assert t.dtype == np.datetime64

        # invalid dtype
        with pytest.raises(ValueError):
            ut.ArrayTrait(dtype='notatype')