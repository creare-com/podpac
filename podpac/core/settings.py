"""
Podpac Settings
"""


import os
import json
from copy import deepcopy

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
    PODPAC Settings

    Settings can be overridden or extended using a ``settings.json`` file located in one of two places:
      * the current working directory (i.e. ``./settings.json``)
      * the *.podpac* directory in the users home directory (i.e. ``~/.podpac/settings.json``)
    
    If not ``settings.json`` file is found, podpac will create one in the users
    home directory (``~/.podpac/settings.json``).
    If settings are changed during runtime, ``settings.json`` will be updated to reflect the changes.
    If you would prefer the settings file not be overwritten, set the ``SAVE_SETTINGS`` field to ``False``.

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

    def _load_user_settings(self, path=None):
        """Load user settings from settings.json file
        
        Parameters
        ----------
        path : str, optional
            Load settings from custom path
        """
        default_filename = 'settings.json'
        default_path = os.path.join(os.path.expanduser('~'), '.podpac', default_filename)
        self._settings_path = None

        # order of paths to check for settings
        path_choices = [
            path,                                         # user specified
            os.path.join(os.getcwd(), default_filename),  # current working directory
            default_path                                  # default path
        ]

        # reset user settings
        user_settings_json = None

        # try path choices in order, break when one works
        for p in path_choices:

            # see if the path exists
            if p is not None and os.path.exists(p):

                # don't catch this so it throws a JSONDecodeError
                with open(p, 'r') as f:
                    user_settings_json = json.load(f)

            # if path exists and settings loaded, then save path and break
            if user_settings_json is not None:
                self._settings_path = p
                break

        # if no user settings found, create a persistent file to store data
        if user_settings_json is None:
            os.makedirs(default_path)
            user_settings_json = {}
            self._settings_path = default_path

        # load user settings into dict
        for key in user_settings_json:
            self[key] = user_settings_json[key]


    def _save_settings(self):
        """Save current settings to active settings file"""

        with open(self._settings_path, 'w') as f:
            json.dump(self, f)



# load settings dict when module is loaded
settings = PodpacSettings()
