"""
Podpac Settings
"""


import os
import json
from copy import deepcopy

# Python 2/3 handling for JSONDecodeError
try:
    from json import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

# Settings Defaults
DEFAULT_SETTINGS = {
    'CACHE_DIR': 'cache',
    'CACHE_TO_S3': False,
    'ROOT_PATH': None,
    'AWS_ACCESS_KEY_ID': None,
    'AWS_SECRET_ACCESS_KEY': None,
    'AWS_REGION_NAME': None,
    'S3_BUCKET_NAME': None,
    'S3_JSON_FOLDER': None,
    'S3_OUTPUT_FOLDER': None,
    'SAVE_SETTINGS': True
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
    
    If settings are changed at runtime, the source ``settings.json`` will be updated.
    If you would prefer the settings file not be changed, set the ``settings['SAVE_SETTINGS']`` field to ``False``.

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
        Path to primary podpac working directory.
    S3_BUCKET_NAME : str
        The AWS S3 Bucket to use for cloud based processing.
    S3_JSON_FOLDER : str
        Folder within :attr:`S3_BUCKET_NAME` to use for cloud based processing.
    S3_OUTPUT_FOLDER : str
        Folder within :attr:`S3_BUCKET_NAME` to use for outputs.
    SAVE_SETTINGS: bool
        Save settings as they are changed during runtime. Defaults to ``True``.

    """
    
    def __init__(self, path=None):
        self._loaded = False

        # call dict init
        super(PodpacSettings, self).__init__()

        # load default settings
        self._load_defaults()

        # load user settings
        self._load_user_settings(path)

        # set loaded flag
        self._loaded = True

        # write out settings
        self._save_settings()

    def __setitem__(self, key, value):

        # get old value if it exists
        try:
            old_val = deepcopy(self[key])
        except KeyError:
            old_val = None

        super(PodpacSettings, self).__setitem__(key, value)

        # save settings file if value has changed
        if self._loaded and self['SAVE_SETTINGS'] and old_val != value:
            self._save_settings()

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
        default_path = os.path.join(os.path.expanduser('~'), '.podpac')
        default_filepath = os.path.join(default_path, filename)
        self._settings_path = None

        # reset user settings
        user_settings_json = None

        # create the custom path if it doesn't exist
        if path is not None:

            # make empty settings path
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            # write an empty settings file
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    json.dump({}, f)

        # order of paths to check for settings
        path_choices = [
            filepath,
            os.path.join(os.getcwd(), filename),  # current working directory
            default_filepath                              # default path
        ]

        # try path choices in order, break when one works
        for p in path_choices:

            # see if the path exists
            if p is not None and os.path.exists(p):

                try:
                    with open(p, 'r') as f:
                        user_settings_json = json.load(f)
                except JSONDecodeError as e:
                    if p == default_filepath:
                        raise e

            # if path exists and settings loaded, then save path and break
            if user_settings_json is not None:
                self._settings_path = p
                break

        # if no user settings found, create a persistent file to store data
        if user_settings_json is None:
            os.makedirs(default_path, exist_ok=True)
            user_settings_json = {}
            self._settings_path = default_filepath

        # load user settings into dict
        for key in user_settings_json:
            self[key] = user_settings_json[key]


    def _save_settings(self):
        """Save current settings to active settings file"""

        with open(self._settings_path, 'w') as f:
            json.dump(self, f)



# load settings dict when module is loaded
settings = PodpacSettings()
