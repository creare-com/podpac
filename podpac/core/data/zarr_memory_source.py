import traitlets as tl

from podpac.core.data.zarr_source import Zarr


class ZarrMemory(Zarr):
    dataset = tl.Any()
    source = tl.Unicode(default_value="ram://")
