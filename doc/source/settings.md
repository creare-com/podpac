# Settings

This tutorial describes methods for viewing and editing PODPAC settings used to configure the features of the library.

To follow along, open a Python interpreter or Jupyter notebook in the Python environment where PODPAC is installed.

```
# activate the PODPAC environment, using anaconda
$ conda activate podpac

# start a ipython interpreter
$ ipython
```

## View Settings

PODPAC settings are accessed through the `podpac.settings` module.
Import the settings module:

```ipython
In [1]: from podpac import settings
```

The settings are stored in a dictionary format, accessible in the interpreter:

```ipython
In [2]: settings
Out[2]:
{'DEBUG': False,
 'ROOT_PATH': 'C:\\Users\\user\\.podpac',
 'AUTOSAVE_SETTINGS': False,
 ...
}
```

To view the default settings, view `settings.defaults`:

```ipython
In [3]: settings.defaults
Out[3]:
{'DEBUG': False,
 'ROOT_PATH': 'C:\\Users\\user\\.podpac',
 'AUTOSAVE_SETTINGS': False,
 ...
}
```

For documentation on individual PODPAC settings, view the `podpac.settings` module docstring:

```ipython
In [1]: from podpac import settings
In [2]:print(settings.__doc__)

    Persistently stored podpac settings

    Podpac settings are persistently stored in...

```

## Edit Settings

PODPAC settings can be edited multiple ways:

- [As a dictionary](#as-a-dictionary): For interactive and dynamic changes 
- [As a JSON file](#as-a-json-file): For consistent configurations across installations

### As a Dictionary

Since the `podpac.settings` module is a dictionary, the simplest way to edit settings is to directly set values:
 
```ipython
In [1]: from podpac import settings

In [2]: settings["DEBUG"] = True
In [3]: settings["CACHE_OUTPUT_DEFAULT"] = False
```

These changed settings will only be active for current interpreter session (or script).
Close and reopen a new interpreter session and you will see the settings values are back to default:

```ipython
In [1]: from podpac import settings

In [2]: settings["DEBUG"]
Out[2]: False
```

To persistently save changes to `settings` values, run `settings.save()` after making changes:

```ipython
In [1]: from podpac import settings

In [2]: settings["DEBUG"] = True
In [3]: settings.save()
```

To auto-save settings on *any* changes, set the `settings["AUTOSAVE_SETTINGS"]` value to `True`:

```ipython
In [1]: from podpac import settings

In [2]: settings["AUTOSAVE_SETTINGS"] = True
```

To reset settings to defaults, call the `settings.reset()` method.
To persistently reset defaults, call the `settings.save()` method after calling `settings.reset()`:

```ipython
In [1]: from podpac import settings

In [2]: settings.reset()

In [3]: settings.save()   # to persistently reset defaults
```

### As a JSON file

PODPAC settings can be pre-configured by creating a custom `settings.json` file.

Create a `settings.json` file in the current working directory:

```json
{
    "DEBUG": true,
    "S3_BUCKET_NAME": "podpac-bucket"
}
```

Open a new interpreter and load the `podpac.settings` module to see the overwritten values:

```ipython
In [1]: from podpac import settings

In [2]: settings["DEBUG"]
Out[2]: True
```

This file can also be placed in the the PODPAC root directory.
To see the PODPAC root directory, view `settings["ROOT_PATH"]`:

```ipython
In [1]: from podpac import settings

In [2]: settings["ROOT_PATH"]
Out[5]: 'C:\\Users\\user\\.podpac'
```

Edit the `settings.json` file in the `"ROOT_PATH"` location, then open a new interpreter and load the `podpac.settings` module to see the overwritten values:

```json
{
    "DISK_CACHE_MAX_BYTES ": 1e9,
}
```

```ipython
In [1]: from podpac import settings

In [2]: settings["DISK_CACHE_MAX_BYTES"]
Out[2]:  1000000000.0
```

If a `settings.json` files exist in multiple places, PODPAC will load settings in the following order,
overwriting previously loaded settings in the process:

* podpac default settings
* home directory settings (`~/.podpac/settings.json`)
* current working directory settings (`./settings.json`)

The attribute `settings.settings_path` shows the path of the last loaded settings file (e.g. the active settings file).

```python
In [1]: from podpac import settings

In [2]: settings.settings_path
Out[2]: 'C:\\Users\\user\\.podpac'
```

A `settings.json` file can be loaded from outside the search path using the `settings.load()` method:

```python
In [1]: from podpac import settings

In [2]: settings.load(path='custom/path/', filename='settings.json')
```
