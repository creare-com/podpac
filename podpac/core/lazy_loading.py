import importlib
import sys
import types
import lazy_loader as lazy


class _LazyModule(types.ModuleType):
    """Placeholder for a module which imports the module on first attribute access.

    This is slower than ``lazy_loader.load`` as it never registers the placeholder as the module itself.
    It instead just imports the module and proxies the access. It should be used only when ``lazy_loader.load``
    cannot be used, such as for submodule imports.
    """

    def __getattr__(self, attr):
        return getattr(importlib.import_module(self.__name__), attr)


def lazy_module(name):
    """Return a module that is imported on first attribute access.

    Top-level packages are loaded with ``lazy_loader.load``. Submodules use a placeholder that defers
    both the parent package and the submodule, because ``lazy_loader.load`` imports the parent immediately.

    Parameters
    ----------
    name : str
        Fully qualified module name, e.g. ``"rasterio"`` or ``"osgeo.gdal"``.

    Returns
    -------
    module : types.ModuleType
        The module if it is already imported, otherwise a placeholder that imports it on first attribute access.
        A missing package raises ``ModuleNotFoundError`` on that first access, not here.
    """
    if "." in name:
        return sys.modules.get(name) or _LazyModule(name)
    return lazy.load(name)
