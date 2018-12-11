# Settings

Podpac settings are persistently stored in a `settings.json` file created at runtime.
By default, podpac will create a settings json file in the users
home directory (`~/.podpac/settings.json`) when first run.
This file can be overridden by creating a custom `settings.json` in the current working directory
or the podpac root directory.

## Default Settings

See the attribute `settings.DEFAULT_SETTINGS` for the default podpac settings.

```python
import podpac

print(podpac.settings.defaults)
```

## Override Settings

Default settings can be overridden or extended by:

* editing the `settings.json` file in the home directory (i.e. `~/.podpac/settings.json`)
* creating a `settings.json` in the current working directory (i.e. `./settings.json`)

If `settings.json` files exist in multiple places, podpac will load settings in the following order,
overwriting previously loaded settings in the process:

* podpac settings defaults
* home directory settings (`~/.podpac/settings.json`)
* current working directory settings (`./settings.json`)

The attribute `settings.settings_path` shows the path of the last loaded settings file (e.g. the active settings file).
To keep the active settings file updated as changes are made to the settings dictionary at runtime,
set the property `settings['AUTOSAVE_SETTINGS']` field to `True`.
The active setting file can be persistently saved at any time using the method `settings.save()`.

## Settings.json

PODPAC settings are configured in a [json](https://json.org) file.

The example output from a default settings file is shown below:

```json
{
    "DEBUG": false,
    "CACHE_DIR": "/Users/user/.podpac/cache",
    "CACHE_TO_S3": false,
    "ROOT_PATH": "/Users/user/.podpac",
    "AWS_ACCESS_KEY_ID": null,
    "AWS_SECRET_ACCESS_KEY": null,
    "AWS_REGION_NAME": null,
    "S3_BUCKET_NAME": null,
    "S3_JSON_FOLDER": null,
    "S3_OUTPUT_FOLDER": null,
    "AUTOSAVE_SETTINGS": false
}
```