# Settings

PODPAC settings are accessed through the `podpac.settings` module.
The settings are stored in a dictionary format:

```python
from podpac import settings

print(settings)

>>> {
        'DEBUG': False,
        'CACHE_DIR': None,
        'CACHE_TO_S3': False,
        'ROOT_PATH': '/Users/user/.podpac',
        'AWS_ACCESS_KEY_ID': None,
        'AWS_SECRET_ACCESS_KEY': None,
        'AWS_REGION_NAME': None,
        'S3_BUCKET_NAME': None,
        'S3_JSON_FOLDER': None,
        'S3_OUTPUT_FOLDER': None,
        'AUTOSAVE_SETTINGS': False
    }
```

These settings can be pre-configured by creating a custom `settings.json` in the current working directory,
the podpac root directory, or a directory specified by the user at runtime.

## Load Settings from Default Paths

You can override default podpac settings by creating a `settings.json` file in one of two places:

* the podpac `ROOT_PATH`. By default this is a `.podpac` directory in the users home directory (i.e. `~/.podpac/settings.json`).
* the current working directory (i.e. `./settings.json`)

If `settings.json` files exist in multiple places, podpac will load settings in the following order,
overwriting previously loaded settings in the process:

* podpac default settings
* home directory settings (`~/.podpac/settings.json`)
* current working directory settings (`./settings.json`)

## Load Settings from a Custom Path

You can also load a `settings.json` file from outside of the podpac `ROOT_PATH` or current working directory using the `settings.load()` method:

```python
from podpac import settings

settings.load(path='custom/path/', filename='settings.json')
```

## Active Settings File

The attribute `settings.settings_path` shows the path of the last loaded settings file (e.g. the active settings file).

```python
from podpac import settings

print(settings.settings_path)
```

## Save Settings

The active settings file (`settings.settings_path`) can be saved by using the `settings.save()` method:

```python
from podpac import settings

# writes out current settings dictionary to json file at settings.settings_path
settings.save()
```

To keep the active settings file updated as changes are made to the settings dictionary at runtime,
set the property `settings['AUTOSAVE_SETTINGS']` field to `True`.

## Default Settings

The default settings can be accessed on the `settings.defaults` attribute.

```python
from podpac import settings

print(settings.defaults)
```

## Details on Specific Settings 

For information on specific settings for your version of podpac see the settings module doc.

```python
>>> from podpac import settings
>>> print(settings.__doc__)
"""
    Persistently stored podpac settings

    Podpac settings are persistently stored in a ``settings.json`` file created at runtime.
    By default, podpac will create a settings json file in the users
    home directory (``~/.podpac/settings.json``) when first run.

    Default settings can be overridden or extended by:
      * editing the ``settings.json`` file in the home directory (i.e. ``~/.podpac/settings.json``)
      * creating a ``settings.json`` in the current working directory (i.e. ``./settings.json``)

    If ``settings.json`` files exist in multiple places, podpac will load settings in the following order,
    overwriting previously loaded settings in the process:
      * podpac settings defaults
      * home directory settings (``~/.podpac/settings.json``)
      * current working directory settings (``./settings.json``)

    :attr:`settings.settings_path` shows the path of the last loaded settings file (e.g. the active settings file).
    To persistenyl update the active settings file as changes are made at runtime,
    set the ``settings['AUTOSAVE_SETTINGS']`` field to ``True``. The active setting file can be persistently
    saved at any time using :meth:`settings.save`.

    The default settings are shown below:

    Attributes
    ----------
    DEFAULT_CRS : str
        Default coordinate reference system for spatial coordinates. Defaults to 'EPSG:4326'.
    AWS_ACCESS_KEY_ID : str
        The access key for your AWS account.
        See the `boto3 documentation
        <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#environment-variable-configuration>`_
        for more details.
    AWS_SECRET_ACCESS_KEY : str
        The secret key for your AWS account.
        See the `boto3 documentation
        <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#environment-variable-configuration>`_
        for more details.
    AWS_REGION_NAME : str
        Name of the AWS region, e.g. us-west-1, us-west-2, etc.
    DEFAULT_CACHE : list
        Defines a default list of cache stores in priority order. Defaults to `['ram']`.
    CACHE_OUTPUT_DEFAULT : bool
        Default value for node ``cache_output`` trait.
    RAM_CACHE_MAX_BYTES : int
        Maximum RAM cache size in bytes.
        Note, for RAM cache only, the limit is applied to the total amount of RAM used by the python process;
        not just the contents of the RAM cache. The python process will not be restrited by this limit,
        but once the limit is reached, additions to the cache will be subject to it.
        Defaults to ``1e9`` (~1G).
        Set to `None` explicitly for no limit.
    DISK_CACHE_MAX_BYTES : int
        Maximum disk space for use by the disk cache in bytes.
        Defaults to ``10e9`` (~10G).
        Set to `None` explicitly for no limit.
    S3_CACHE_MAX_BYTES : int
        Maximum storage space for use by the s3 cache in bytes.
        Defaults to ``10e9`` (~10G).
        Set to `None` explicitly for no limit.
    DISK_CACHE_DIR : str
        Subdirectory to use for the disk cache. Defaults to ``'cache'`` in the podpac root directory.
    S3_CACHE_DIR : str
        Subdirectory to use for S3 cache (within the specified S3 bucket). Defaults to ``'cache'``.
    RAM_CACHE_ENABLED: bool
        Enable caching to RAM. Note that if disabled, some nodes may fail. Defaults to ``True``.
    DISK_CACHE_ENABLED: bool
        Enable caching to disk. Note that if disabled, some nodes may fail. Defaults to ``True``.
    S3_CACHE_ENABLED: bool
        Enable caching to RAM. Note that if disabled, some nodes may fail. Defaults to ``True``.
    ROOT_PATH : str
        Path to primary podpac working directory. Defaults to the ``.podpac`` directory in the users home directory.
    S3_BUCKET_NAME : str
        The AWS S3 Bucket to use for cloud based processing.
    S3_JSON_FOLDER : str
        Folder within :attr:`S3_BUCKET_NAME` to use for cloud based processing.
    S3_OUTPUT_FOLDER : str
        Folder within :attr:`S3_BUCKET_NAME` to use for outputs.
    AUTOSAVE_SETTINGS: bool
        Save settings automatically as they are changed during runtime. Defaults to ``False``.
    MULTITHREADING: bool
        Uses multithreaded evaluation, when applicable. Defaults to ``False``.
    N_THREADS: int
        Number of threads to use (only if MULTITHREADING is True). Defaults to ``10``.
    CHUNK_SIZE: int, 'auto', None
        Chunk size for iterative evaluation, when applicable (e.g. Reduce Nodes). Use None for no iterative evaluation,
        and 'auto' to automatically calculate a chunk size based on the system. Defaults to ``None``.
"""
>>>
```
