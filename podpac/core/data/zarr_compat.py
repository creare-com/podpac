"""
Compatibility helpers bridging zarr 2.x and zarr 3.x APIs.

zarr 3 removed `Group.create_dataset` in favor of `Group.create_array`, and no
longer accepts fsspec `MutableMapping` stores (like `s3fs.S3Map`) directly.
"""

from lazy_import import lazy_module

zarr = lazy_module("zarr")


def _zarr_v3():
    return int(zarr.__version__.split(".")[0]) >= 3


def create_zarr_array(group, name, chunks=None, **kwargs):
    if hasattr(group, "create_array"):
        if chunks is True:
            chunks = "auto"
        if chunks is not None:
            kwargs["chunks"] = chunks
        # v3's create_array, unlike v2's create_dataset, has no implicit dtype fallback
        # when `data` isn't provided directly.
        kwargs.setdefault("dtype", "float64")
        return group.create_array(name, **kwargs)
    if chunks is not None:
        kwargs["chunks"] = chunks
    return group.create_dataset(name, **kwargs)


def get_s3_store(s3, root):
    if _zarr_v3():
        return zarr.storage.FsspecStore(s3, path=root)
    s3fs = lazy_module("s3fs")
    return s3fs.S3Map(root=root, s3=s3, check=False)
