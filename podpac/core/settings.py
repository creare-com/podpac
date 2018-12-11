"""
Podpac Settings
"""


import os
import json
from copy import deepcopy
import errno

# Python 2/3 handling for JSONDecodeError
try:
    from json import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

# Settings Defaults
DEFAULT_SETTINGS = {
    'DEBUG': False,
    'CACHE_DIR': 'cache',
    'CACHE_TO_S3': False,
    'ROOT_PATH': os.path.join(os.path.expanduser('~'), '.podpac'),
    'AWS_ACCESS_KEY_ID': None,
    'AWS_SECRET_ACCESS_KEY': None,
    'AWS_REGION_NAME': None,
    'S3_BUCKET_NAME': None,
    'S3_JSON_FOLDER': None,
    'S3_OUTPUT_FOLDER': None,
    'AUTOSAVE_SETTINGS': False
}


class PodpacSettings(dict):
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
    CACHE_DIR : str
        Directory to use as a cache locally or on S3. Defaults to ``'cache'``.
    CACHE_TO_S3 : bool
        Cache results to the AWS S3 bucket.
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

    """
    
    def __init__(self, path=None):
        self._loaded = False

        # call dict init
        super(PodpacSettings, self).__init__()

        # load default settings
        self._load_defaults()

        # load user settings
        self._load_user_settings(path)

        # it breaks things to set the root path to None, set back to default if set to None
        if self['ROOT_PATH'] is None:
            self['ROOT_PATH'] = DEFAULT_SETTINGS['ROOT_PATH']

        # TODO: handle this in the cache module
        if self['S3_BUCKET_NAME'] and self['CACHE_TO_S3']:
            self['CACHE_DIR'] = 'cache'
        else:
            self['CACHE_DIR'] = os.path.abspath(os.path.join(self['ROOT_PATH'], self['CACHE_DIR']))
            self._mkdir(self['CACHE_DIR'])

        # set loaded flag
        self._loaded = True

        # if no settings file exists, this will create one in the user's home directory
        self.save()

    def __setitem__(self, key, value):

        # get old value if it exists
        try:
            old_val = deepcopy(self[key])
        except KeyError:
            old_val = None

        super(PodpacSettings, self).__setitem__(key, value)

        # save settings file if value has changed
        if self._loaded and self['AUTOSAVE_SETTINGS'] and old_val != value:
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

    def _load_user_settings(self, path=None, filename='settings.json'):
        """Load user settings from settings.json file
        
        Parameters
        ----------
        path : str, optional
            Full path to custom settings file
        filename : str, optional
            Filename of custom settings file
        """

        filepath = os.path.join(path, filename) if path is not None else None
        default_path = self['ROOT_PATH']
        default_filepath = os.path.join(default_path, filename)
        
        # set settings path to default to start
        self._settings_filepath = default_filepath

        # if input path is specifed, create the input path if it doesn't exist
        if path is not None:

            # make empty settings path
            if not os.path.exists(path):
                self._mkdir(path)

            # write an empty settings file
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    json.dump({}, f)

        # order of paths to check for settings
        filepath_choices = [
            default_filepath,                       # default path
            os.path.join(os.getcwd(), filename),    # current working directory
            filepath                                # input directory
        ]

        # try path choices in order, overwriting earlier ones with later ones
        for p in filepath_choices:
            # reset json settings
            json_settings = None

            # see if the path exists
            if p is not None and os.path.exists(p):

                try:
                    with open(p, 'r') as f:
                        json_settings = json.load(f)
                except JSONDecodeError:

                    # if the default_filepath settings file is broken, raise
                    if p == default_filepath:
                        raise

                # if path exists and settings loaded then load those settings into the dict
                if json_settings is not None:
                    for key in json_settings:
                        self[key] = json_settings[key]

                    # save this path as the active
                    self._settings_filepath = p

    def _mkdir(self, path):
        """Wrapper for os.mkdirs(exist_ok=True)
        
        Parameters
        ----------
        path : str
            path to directory
        """
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

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
    def defaults(self):
        """
        Show the podpac default settings
        """
        return DEFAULT_SETTINGS
    
    def save(self):
        """
        Save current settings to active settings file

        :attr:`settings.settings_path` shows the path to the currently active settings file
        """

        # if no settings path is found, create
        if not os.path.exists(self._settings_filepath):
            self._mkdir(os.path.dirname(self._settings_filepath))

        with open(self._settings_filepath, 'w') as f:
            json.dump(self, f)





# load settings dict when module is loaded
settings = PodpacSettings()
