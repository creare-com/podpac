from __future__ import division, print_function, absolute_import

import fnmatch
from lazy_import import lazy_module

boto3 = lazy_module("boto3")

import podpac
from podpac.core.settings import settings
from podpac.core.cache.utils import CacheException, CacheWildCard
from podpac.core.cache.file_cache_store import FileCacheStore


class S3CacheStore(FileCacheStore):  # pragma: no cover

    cache_mode = "s3"
    cache_modes = set(["s3", "all"])
    _limit_setting = "S3_CACHE_MAX_BYTES"
    _delim = "/"

    def __init__(self, s3_bucket=None, aws_region_name=None, aws_access_key_id=None, aws_secret_access_key=None):
        """Initialize a cache that uses a folder on a local disk file system.

        Parameters
        ----------
        max_size : None, optional
            Maximum allowed size of the cache store in bytes. Defaults to podpac 'S3_CACHE_MAX_BYTES' setting, or no limit if this setting does not exist.
        use_settings_limit : bool, optional
            Use podpac settings to determine cache limits if True, this will also cause subsequent runtime changes to podpac settings module to effect the limit on this cache. Default is True.
        s3_bucket : str, optional
            bucket name, overides settings
        aws_region_name : str, optional
            e.g. 'us-west-1', 'us-west-2','us-east-1'
        aws_access_key_id : str, optional
            overides podpac settings if both `aws_access_key_id` and `aws_secret_access_key` are specified
        aws_secret_access_key : str, optional
            overides podpac settings if both `aws_access_key_id` and `aws_secret_access_key` are specified
        """

        if not settings["S3_CACHE_ENABLED"]:
            raise CacheException("S3 cache is disabled in the podpac settings.")

        self._root_dir_path = settings["S3_CACHE_DIR"]

        if s3_bucket is None:
            s3_bucket = settings["S3_BUCKET_NAME"]
        if aws_access_key_id is None or aws_secret_access_key is None:
            aws_access_key_id = settings["AWS_ACCESS_KEY_ID"]
            aws_secret_access_key = settings["AWS_SECRET_ACCESS_KEY"]
        if aws_region_name is None:
            aws_region_name = settings["AWS_REGION_NAME"]
        aws_session = boto3.session.Session(region_name=aws_region_name)
        self._s3_client = aws_session.client(
            "s3",
            # config= boto3.session.Config(signature_version='s3v4'),
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self._s3_bucket = s3_bucket

        try:
            self._s3_client.head_bucket(Bucket=self._s3_bucket)
        except Exception as e:
            raise e

    # -----------------------------------------------------------------------------------------------------------------
    # main cache API
    # -----------------------------------------------------------------------------------------------------------------

    @property
    def size(self):
        paginator = self._s3_client.get_paginator("list_objects")
        operation_parameters = {"Bucket": self._s3_bucket, "Prefix": self._root_dir_path}
        page_iterator = paginator.paginate(**operation_parameters)
        total_size = 0
        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    total_size += obj["Size"]
        return total_size

    def cleanup(self):
        """
        Remove expired entries.
        """

        pass  # TODO metadata

    # -----------------------------------------------------------------------------------------------------------------
    # helper methods
    # -----------------------------------------------------------------------------------------------------------------

    def search(self, node, item=CacheWildCard(), coordinates=CacheWildCard()):
        """Fileglob to match files that could be storing cached data for specified node,key,coordinates

        Parameters
        ----------
        node : podpac.core.node.Node
        item : str, CacheWildCard
            CacheWildCard indicates to match any key
        coordinates : podpac.core.coordinates.coordinates.Coordinates, CacheWildCard, None
            CacheWildCard indicates to macth any coordinates

        Returns
        -------
        TYPE : str
            Fileglob of existing paths that match the request
        """

        delim = self._delim
        prefix = self._get_node_dir(node)
        prefix = prefix if prefix.endswith(delim) else prefix + delim
        response = self._s3_client.list_objects_v2(Bucket=self._s3_bucket, Prefix=prefix, Delimiter=delim)

        if response["KeyCount"] > 0:
            obj_names = [o["Key"].replace(prefix, "") for o in response["Contents"]]
        else:
            obj_names = []

        node_dir = self._get_node_dir(node)
        obj_names = fnmatch.filter(obj_names, self._get_filename_pattern(node, item, coordinates))
        paths = [self._path_join(node_dir, filename) for filename in obj_names]
        return paths

    # -----------------------------------------------------------------------------------------------------------------
    # file storage abstraction
    # -----------------------------------------------------------------------------------------------------------------

    def _save(self, path, s, metadata=None):
        # note s needs to be b'bytes' or file below
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.put_object
        self._s3_client.put_object(Bucket=self._s3_bucket, Body=s, Key=path)
        # TODO metadata

    def _load(self, path):
        response = self._s3_client.get_object(Bucket=self._s3_bucket, Key=path)
        return response["Body"].read()

    def _path_join(self, *paths):
        return self._delim.join(paths)

    def _basename(self, path):
        if self._delim in path:
            dirname, basename = path.rsplit(self._delim, 1)
        else:
            basename = path
        return basename

    def _remove(self, path):
        self._s3_client.delete_object(Bucket=self._s3_bucket, Key=path)

    def _exists(self, path):
        response = self._s3_client.list_objects_v2(Bucket=self._s3_bucket, Prefix=path)
        obj_count = response["KeyCount"]
        return obj_count == 1 and response["Contents"][0]["Key"] == path

    def _make_dir(self, path):
        # Does not need to do anything for S3 as the prefix is just part of the object name.
        # note: I believe AWS uses prefixes to decide how to partition objects in a bucket which could affect performance.
        pass

    def _rmtree(self, path):
        paginator = self._s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self._s3_bucket, Prefix=path)

        to_delete = dict(Objects=[])
        for item in pages.search("Contents"):
            if item:
                to_delete["Objects"].append(dict(Key=item["Key"]))
            if len(to_delete["Objects"]) >= 1000:
                self._s3_client.delete_objects(Bucket=self._s3_bucket, Delete=to_delete)
                to_delete = dict(Objects=[])

        if len(to_delete["Objects"]):
            self._s3_client.delete_objects(Bucket=self._s3_bucket, Delete=to_delete)

    def _is_empty(self, directory):
        if not directory.endswith(self._delim):
            directory += self._delim
        response = self._s3_client.list_objects_v2(Bucket=self._s3_bucket, Prefix=directory, MaxKeys=2)
        # TODO throw an error if key count is zero as this indicates `directory` is not an existing directory.
        return response["KeyCount"] == 1

    def _rmdir(self, directory):
        # s3 can have "empty" directories
        # should check if directory is empty and the delete
        # delete_objects could be used if recursive=True is specified to this function.
        # NOTE: This can remove the object representing the prefix without deleting other objects with the prefix.
        #       This is because s3 is really a key/value store. This function should maybe be changed to throw an
        #       error if the prefix is not "empty" or should delete all objects with the prefix. The former would
        #       be more consistent with the os function used in the DiskCacheStore.
        # ToDo: 1) examine boto3 response, 2) handle object versioning (as it stands this will apply to the "null version")
        #      https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.delete_object
        if not directory.endswith(self._delim):
            directory += self._delim
        self._s3_client.delete_object(Bucket=self._s3_bucket, Key=directory)

    def _dirname(self, path):
        dirname, basename = path.rsplit(self._delim, 1)
        return dirname

    def _get_metadata(self, path, key):
        return None  # TODO metadata

    def _set_metadata(self, path, key, value):
        pass  # TODO metadata
