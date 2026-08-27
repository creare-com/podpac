"""
Compatibility helpers bridging zarr 2.x and zarr 3.x APIs.

zarr 3 removed `Group.create_dataset` in favor of `Group.create_array`, and no
longer accepts fsspec `MutableMapping` stores (like `s3fs.S3Map`) directly.
"""

from typing import Any

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


def _ensure_async_fs(fs: Any) -> Any:
    """Coerce an fsspec filesystem instance into one zarr 3's FsspecStore accepts.

    `S3Mixin` builds a synchronous `s3fs` instance, but zarr 3's `FsspecStore`
    requires an async-native filesystem: it raises `TypeError` if the
    filesystem's class never implemented the async protocol (e.g. s3fs
    versions predating `fsspec.asyn.AsyncFileSystem`, which podpac's own
    "s3fs>=0.4" floor still technically allows), and warns even when the class
    is async-capable but this particular instance wasn't created with
    `asynchronous=True`. This mirrors zarr's own internal `_make_async` helper
    (used by `FsspecStore.from_url`/`from_mapper`).

    Parameters
    ----------
    fs : fsspec.spec.AbstractFileSystem
        The filesystem instance to coerce, e.g. an `s3fs.S3FileSystem`. May
        already be async, async-capable but synchronous, or not async-capable
        at all.

    Returns
    -------
    fsspec.spec.AbstractFileSystem
        `fs` itself if it is already asynchronous; otherwise a new instance
        of `type(fs)` constructed with `asynchronous=True` if the class
        supports it; otherwise `fs` wrapped in an `AsyncFileSystemWrapper`.
    """
    if getattr(fs, "asynchronous", False):
        return fs
    if getattr(fs, "async_impl", False):
        return type(fs)(*fs.storage_args, **{**fs.storage_options, "asynchronous": True})
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    return AsyncFileSystemWrapper(fs, asynchronous=True)


def get_s3_store(s3, root):
    if _zarr_v3():
        return zarr.storage.FsspecStore(_ensure_async_fs(s3), path=root)
    s3fs = lazy_module("s3fs")
    return s3fs.S3Map(root=root, s3=s3, check=False)
