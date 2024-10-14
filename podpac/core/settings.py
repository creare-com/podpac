"""
Podpac Settings
"""

import os
import json
from copy import deepcopy
import errno
import uuid
import logging

from podpac import version

# Python 2/3 handling for JSONDecodeError
try:
    from json import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

_logger = logging.getLogger(__name__)

# Settings Defaults
DEFAULT_SETTINGS = {
    # podpac core settings
    "DEBUG": False,  # This flag currently sets self._output on nodes
    "ROOT_PATH": os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~")), ".config", "podpac"),
    "AUTOSAVE_SETTINGS": False,
    "LOG_TO_FILE": False,
    "LOG_FILE_PATH": os.path.join(
        os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~")), "podpac", "logs", "podpac.log"
    ),
    "MULTITHREADING": False,
    "N_THREADS": 8,
    "CHUNK_SIZE": None,  # Size of chunks for parallel processing or large arrays that do not fit in memory
    "ENABLE_UNITS": True,
    "PODPAC_VERSION": version.semver(),
    "UNSAFE_EVAL_HASH": uuid.uuid4().hex,  # unique id for running unsafe evaluations
    "DEFAULT_CRS": "+proj=longlat +datum=WGS84 +no_defs +vunits=m",  # EPSG:4326 with vertical units as meters
    "DEFAULT_INTERPOLATION": "nearest",
    # cache
    "DEFAULT_CACHE": ["ram"],
    "CACHE_DATASOURCE_OUTPUT_DEFAULT": True,
    "CACHE_NODE_OUTPUT_DEFAULT": False,
    "RAM_CACHE_MAX_BYTES": 1e9,  # ~1GB
    "DISK_CACHE_MAX_BYTES": 10e9,  # ~10GB
    "S3_CACHE_MAX_BYTES": 10e9,  # ~10GB
    "DISK_CACHE_DIR": "cache",
    "S3_CACHE_DIR": "cache",
    "RAM_CACHE_ENABLED": True,
    "DISK_CACHE_ENABLED": True,
    "S3_CACHE_ENABLED": True,
    # AWS
    "AWS_ACCESS_KEY_ID": None,
    "AWS_SECRET_ACCESS_KEY": None,
    "AWS_REGION_NAME": None,
    "AWS_TAGS": None,
    "AWS_BUDGET_AMOUNT": None,
    "AWS_BUDGET_EMAIL": None,
    "AWS_REQUESTER_PAYS": False,
    "S3_BUCKET_NAME": None,
    "FUNCTION_NAME": None,
    "FUNCTION_ROLE_NAME": None,
    "FUNCTION_DEPENDENCIES_KEY": None,
    "FUNCTION_S3_INPUT": None,
    "FUNCTION_S3_OUTPUT": None,
    "FUNCTION_FORCE_COMPUTE": False,
    "ALGORITHM_XARRAY_FLOATING_POINT_CORRECTION":False
}


class PodpacSettings(dict):
    """
    Persistently stored podpac settings

    Podpac settings are persistently stored in a ``settings.json`` file created at runtime.
    By default, podpac will create a settings json file in the users
    home directory (``~/.config/podpac/settings.json``), or $XDG_CONFIG_HOME/podpac/settings.json when first run.

    Default settings can be overridden or extended by:
      * editing the ``settings.json`` file in the settings directory (i.e. ``~/.podpac/settings.json`` or
      ``$XDG_CONFIG_HOME/podpac/settings.json``)
      * creating a ``settings.json`` in the current working directory (i.e. ``./settings.json``)

    If ``settings.json`` files exist in multiple places, podpac will load settings in the following order,
    overwriting previously loaded settings in the process (i.e. highest numbered settings file prefered):
      1. podpac settings defaults
      2. settings directory  (``~/.podpac/settings.json``  or ``$XDG_CONFIG_HOME/podpac/settings.json``)
      3. current working directory settings (``./settings.json``)

    :attr:`settings.settings_path` shows the path of the last loaded settings file (e.g. the active settings file).
    To persistently update the active settings file as changes are made at runtime,
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
    AWS_BUDGET_AMOUNT : float
        Budget amount for AWS resources
    AWS_BUDGET_EMAIL : str
        Notification email for when AWS usage reaches 80% of the `AWS_BUDGET_AMOUNT`
    AWS_REQUESTER_PAYS : bool
        Allow access to "requester pays" S3 buckets
    DEFAULT_CACHE : list
        Defines a default list of cache stores in priority order. Defaults to `['ram']`.
    CACHE_NODE_OUTPUT_DEFAULT : bool
        Default value for node ``cache_output`` trait. If True, the outputs of nodes (eval) will be automatically cached.
    CACHE_DATASOURCE_OUTPUT_DEFAULT : bool
        Default value for DataSource nodes ``cache_output`` trait. If True, the outputs of nodes (eval) will be automatically cached.
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
        Use settings.cache_path to access this settings (this property looks for the environmental variable
        `XDG_CACHE_HOME` to adjust the location of the cache directory)
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
    FUNCTION_S3_INPUT : str
        Folder within :attr:`S3_BUCKET_NAME` to use for triggering node evaluation.
    FUNCTION_S3_OUTPUT : str
        Folder within :attr:`S3_BUCKET_NAME` to use for outputs.
    FUNCTION_FORCE_COMPUTE : bool
        Force the lambda function to compute pipeline, even if result is already cached.
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

    def __init__(self):
        self._loaded = False

        # call dict init
        super(PodpacSettings, self).__init__()

        # load settings from default locations
        self.load()

        # set loaded flag
        self._loaded = True

    def __setitem__(self, key, value):

        # get old value if it exists
        try:
            old_val = deepcopy(self[key])
        except KeyError:
            old_val = None

        super(PodpacSettings, self).__setitem__(key, value)

        # save settings file if value has changed
        if self._loaded and self["AUTOSAVE_SETTINGS"] and old_val != value:
            self.save()

    def __getitem__(self, key):

        # return none if the parameter does not exist
        try:
            return super(PodpacSettings, self).__getitem__(key)
        except KeyError:
            return None

    def _load_defaults(self):
        """Load default settings"""

        for key in DEFAULT_SETTINGS:
            self[key] = DEFAULT_SETTINGS[key]

    def _load_user_settings(self, path=None, filename="settings.json"):
        """Load user settings from settings.json file

        Parameters
        ----------
        path : str
            Full path to containing directory of settings file
        filename : str
            Filename of custom settings file
        """

        # custom file path - only used if path is not None
        filepath = os.path.join(path, filename) if path is not None else None

        # home path location is in the ROOT_PATH
        root_filepath = os.path.join(self["ROOT_PATH"], filename)

        # cwd path
        cwd_filepath = os.path.join(os.getcwd(), filename)

        # set settings path to default to start
        self._settings_filepath = root_filepath

        # if input path is specifed, create the input path if it doesn't exist
        if path is not None:

            # make empty settings path
            if not os.path.exists(path):
                raise ValueError("Input podpac settings path does not exist: {}".format(path))

        # order of paths to import settings - the later settings will overwrite earlier ones
        filepath_choices = [root_filepath, cwd_filepath, filepath]

        # try path choices in order, overwriting earlier ones with later ones
        for p in filepath_choices:
            # reset json settings
            json_settings = None

            # see if the path exists
            if p is not None and os.path.exists(p):

                try:
                    with open(p, "r") as f:
                        json_settings = json.load(f)
                except JSONDecodeError:

                    # if the root_filepath settings file is broken, raise
                    if p == root_filepath:
                        raise

                # if path exists and settings loaded then load those settings into the dict
                if json_settings is not None:
                    for key in json_settings:
                        self[key] = json_settings[key]

                    # save this path as the active
                    self._settings_filepath = p

    @property
    def settings_path(self):
        """Path to the last loaded ``settings.json`` file

        Returns
        -------
        str
            Path to the last loaded ``settings.json`` file
        """
        return self._settings_filepath

    @property
    def cache_path(self):
        """Path to the cache

        Returns
        -------
        str
            Path to where the cache is stored
        """
        if os.path.isabs(settings["DISK_CACHE_DIR"]):
            path = settings["DISK_CACHE_DIR"]
        else:
            path = os.path.join(os.environ.get("XDG_CACHE_HOME", settings["ROOT_PATH"]), settings["DISK_CACHE_DIR"])

        return path

    @property
    def defaults(self):
        """
        Show the podpac default settings
        """
        return DEFAULT_SETTINGS

    def save(self, filepath=None):
        """
        Save current settings to active settings file

        :attr:`settings.settings_path` shows the path to the currently active settings file

        Parameters
        ----------
        filepath : str, optional
            Path to settings file to save. Defaults to :attr:`self.settings_filepath`
        """

        # custom filepath
        if filepath is not None:
            self._settings_filepath = filepath

        # if no settings path is found, create
        if not os.path.exists(self._settings_filepath):
            os.makedirs(os.path.dirname(self._settings_filepath), exist_ok=True)

        with open(self._settings_filepath, "w") as f:
            json.dump(self, f, indent=4)

    def reset(self):
        """
        Reset settings to defaults.

        This method will ignore the value in :attr:`AUTOSAVE_SETTINGS`.
        To persistenly reset settings to defaults, call :meth:`settings.save()` after this method.
        """

        # ignore autosave
        self["AUTOSAVE_SETTINGS"] = False

        # clear all values
        self.clear()

        # load default settings
        self._load_defaults()

    def load(self, path=None, filename="settings.json"):
        """
        Load a new settings file to be active

        :attr:`settings.settings_path` shows the path to the currently active settings file

        Parameters
        ----------
        path : str, optional
            Path to directory which contains the settings file. Defaults to :attr:`DEFAULT_SETTINGS['ROOT_PATH']`
        filename : str, optional
            Filename of the settings file. Defaults to 'settings.json'
        """
        # load default settings
        self._load_defaults()

        # load user settings
        self._load_user_settings(path, filename)

        # it breaks things to set these paths to None, set back to default if set to None
        if self["ROOT_PATH"] is None:
            self["ROOT_PATH"] = DEFAULT_SETTINGS["ROOT_PATH"]

        if self["DISK_CACHE_DIR"] is None:
            self["DISK_CACHE_DIR"] = DEFAULT_SETTINGS["DISK_CACHE_DIR"]

        if self["S3_CACHE_DIR"] is None:
            self["S3_CACHE_DIR"] = DEFAULT_SETTINGS["S3_CACHE_DIR"]

    @property
    def allow_unsafe_eval(self):
        return "PODPAC_UNSAFE_EVAL" in os.environ and os.environ["PODPAC_UNSAFE_EVAL"] == self["UNSAFE_EVAL_HASH"]

    def set_unsafe_eval(self, allow=False):
        _logger.warning(
            "DEPRECATION WARNING: The `set_unsafe_eval` method has been deprecated and will be removed in future versions of PODPAC. Use `allow_unrestricted_code_execution` instead. "
        )
        self.allow_unrestricted_code_execution(allow)

    def allow_unrestricted_code_execution(self, allow=False):
        """Allow unsafe evaluation for this podpac environment

        Parameters
        ----------
        allow : bool, optional
            Enable unsafe evaluation. Defaults to False.
        """
        if allow:
            os.environ["PODPAC_UNSAFE_EVAL"] = self["UNSAFE_EVAL_HASH"]
            _logger.warning(
                "Setting unrestricted code execution can results in vulnerabilities on publically accessible servers. Use with caution."
            )
        else:
            if "PODPAC_UNSAFE_EVAL" in os.environ:
                os.environ.pop("PODPAC_UNSAFE_EVAL")

    def __enter__(self):
        # save original settings
        self._original = {k: v for k, v in self.items()}

    def __exit__(self, type, value, traceback):
        # restore original settings
        for k, v in self._original.items():
            self[k] = v


# load settings dict when module is loaded
settings = PodpacSettings()
