# Lazy-import the core dependencies for faster import of PODPAC
import lazy_import
requests = lazy_import.lazy_module('requests')
pint = lazy_import.lazy_module('pint')
plt = lazy_import.lazy_module('matplotlib.pyplot')
np = lazy_import.lazy_module('numpy')
sp = lazy_import.lazy_module('scipy')
tl = lazy_import.lazy_module('traitlets')
xr = lazy_import.lazy_module('xarray')

from .managers import aws_lambda 

