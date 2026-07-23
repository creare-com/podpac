"""
Compatibility helpers bridging zarr 2.x and zarr 3.x APIs.

zarr 3 removed the `zarr.convenience` and `zarr.hierarchy` submodules (use the
top-level `zarr.open`, `zarr.open_consolidated`, `zarr.group`, `zarr.Group` instead,
which are stable across both major versions), removed `Group.create_dataset` in favor
of `Group.create_array`, changed the exception raised when consolidated metadata is
missing, and no longer accepts fsspec `MutableMapping` stores (like `s3fs.S3Map`)
directly.
"""
from lazy_import import lazy_module

zarr = lazy_module("zarr")


def _zarr_v3():
    return int(zarr.__version__.split(".")[0]) >= 3


def zarr_open(store, mode="r", **kwargs):
    return zarr.open(store, mode=mode, **kwargs)


def zarr_open_consolidated(store, mode="r", **kwargs):
    try:
        return zarr.open_consolidated(store, mode=mode, **kwargs)
    except (KeyError, ValueError):
        return None  # No consolidated metadata available


def zarr_group():
    return zarr.group()


def create_zarr_array(group, name, chunks=None, **kwargs):
    if hasattr(group, "create_array"):
        if chunks is True:
            chunks = "auto"
        return group.create_array(name, chunks=chunks, **kwargs)
    return group.create_dataset(name, chunks=chunks, **kwargs)


def get_s3_store(s3, root):
    if _zarr_v3():
        return zarr.storage.FsspecStore(s3, path=root)
    s3fs = lazy_module("s3fs")
    return s3fs.S3Map(root=root, s3=s3, check=False)
