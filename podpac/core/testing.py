
import numpy as np

SUBDTYPE_TEMPLATE = '''
Array dtype incorrect

 Array: %r

 Actual dtype: %s
 Desired dtype: %s
'''

def assert_equal_dtype(a, b, dtype=None):
    np.testing.assert_equal(a, b)
    
    if dtype is None:
        dtype = b.dtype

    a = np.array(a)
    assert np.issubdtype(a.dtype, dtype), SUBDTYPE_TEMPLATE % (
        a, a.dtype, np.dtype(dtype).name)