import sys

# Lazy-import the core dependencies for faster import of PODPAC
import lazy_import

# Monkey-patch lazy_import for Python2 compatibility
if sys.version_info.major == 2:
    # Need to save a reference to the real function
    lazy_import._old_lazy_module = lazy_import.lazy_module

    def lazy_module(modname, *args, **kwargs):
        # Python 2 complains about unicode strings, so we turn the modname into a str
        return lazy_import._old_lazy_module(str(modname), *args, **kwargs)

    # Patch
    lazy_import.lazy_module = lazy_module
del sys

requests = lazy_import.lazy_module("requests")
pint = lazy_import.lazy_module("pint")
matplotlib = lazy_import.lazy_module("matplotlib")
plt = lazy_import.lazy_module("matplotlib.pyplot")
np = lazy_import.lazy_module("numpy")
tl = lazy_import.lazy_module("traitlets")
# xr = lazy_import.lazy_module("xarray")
