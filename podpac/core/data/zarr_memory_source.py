import traitlets as tl

from lazy_import import lazy_module, lazy_class


zarrGroup = lazy_class("zarr.Group")


from podpac.core.data.zarr_source import Zarr


class ZarrMemory(Zarr):
    dataset = tl.Any(zarrGroup)
    source = tl.Unicode(default_value="ram://")
    
    
