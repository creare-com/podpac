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