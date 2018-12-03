# Contributing

To get a sense of where the project is going, have a look at our [Roadmap](../roadmap)

There are a number of ways to contribute:

* Create new issues for feature requests or to report bugs
* Adding / correcting documentation
* Adding a new unit test
* Contributing a new node that accesses a specific datasource
* Contributing a new node that implements a domain-specific algorithm
* Commenting on issues to help out other users

To contribute:

* Fork the PODPAC repository on github
* Create a new feature branch from the `develop` branch

```bash
git checkout develop  # Assuming you've already checked out and tracked the develop branch
git branch feature/my_new_feature
```

* Make your changes / additions
* Add / modify the docstrings and other documentation
* Write any additional unit tests
* Create a new pull request

At this point we will review your changes, request modifications, and ultimately accept or reject your modifications. 

## Coding style

* Generally try to follow PEP8, but we're not strict about it. 
* Code should be compatible with both Python 2 and 3

### Docstrings

All classes and methods should be properly documented with docstrings.
Docstrings will be used to create the package documentation.

Many IDE's have auto docstring generators to make this process easier. See the [AutoDocstring](https://github.com/KristoforMaynard/SublimeAutoDocstring) sublime text plugin for one example.

#### Format

Podpac adheres to the [numpy format for docstrings](https://numpydoc.readthedocs.io/en/latest/format.html):

Examples:

- https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
- https://docs.scipy.org/doc/numpy/docs/howto_document.html#example-source

Note that class attributes can be documented multiple ways.
 - Key public attributes and properties should be documented in the main class docstring under `Parameters`.
 - All public attributes (traits) should be documented in the subsequent line by setting its `__doc__` property.
 - All public properties created with the `@property` decorator should be documented in the getter method.

```python
class ExampleClass(tl.HasTraits):
    """The summary line for a class docstring should fit on one line.

    Additional details should be documented here.

    Parameters
    ----------
    attr1 : str
        Description of attr1.
    attr2 : dict, optional
        Description of attr2.
    """

    attr1 = tl.Str()
    attr.__doc__ = ":str: Description of attr1"

    attr2 = tl.Dict(allow_none=True)
    attr2.__doc__ = ":dict: Description of attr2"

    attr3 : tl.Int()
    attr2.__doc__ = ":int: Description of secondary attr3"

    @property
    def attr4(self):
        """:bool: Description of attr4."""

        return True
```

#### References

All references to podpac classes (`:class:`), methods (`:meth:`), and attributes (`:attr:`) should use the *public* path to the reference. If the class does not have a public reference, fall back on the *full* path reference to the class. For example:

```python
def method(coordinates, output=None):
    """Class Method.  
    See method :meth:`podpac.data.DataSource.eval`.
    See attribute :attr:`podpac.core.data.interpolate.INTERPOLATION_METHODS`.

    Parameters
    ----------
    coordinates : :class:`podpac.Coordinates`
      Coordinate input
    output : :class:`podpac.core.units.UnitsDataArray`, optional
      Container for output

    Returns
    --------
    :class:`podpac.core.units.UnitsDataArray`
      Returns a UnitsDataArray
    """
```

### Lint

To help adhere to PEP8, we use the `pylint` module. This provides the most benefit if you [configure your text editor or IDE](https://pylint.readthedocs.io/en/latest/user_guide/ide-integration.html)  to run pylint as you develop. To use `pylint` from the command line:

```bash
$ pylint podpac                 # lint the whole module
$ pylint podpac/settings.py     # lint single file
```

Configuration options are specified in `.pylintrc`.

## Import Conventions / API Conventions

### Public API

The client facing public API should be available on the root `podpac` module. 
These imports are defined in the root level `podpac/__init__.py` file.

The public API will contain a top level of primary imports (i.e. `Node`, `Coordinate`) and a second level of imports that wrap more advanced public functionality. For example, `podpac.algorithm` will contain "advanced user" public imports from `podpac.core.algorithm`.  The goal here is to keep the public namespace of `podpac` lean while providing organized access to higher level functionality.  The most advanced users can always access the full functionality of the package via the `podpac.core` module ([Developer API](#developer-api)). All of this configuration and organization should be contained in `podpac/__init__.py`, if possible.

For example:

```python
import podpac

dir(podpac)
[
 # Public Classes, Functions exposed here for users
 'Algorithm',
 'Node',
 'Coorindate',
 ...

 # organized submodules
 'algorithm,
 'data'
 'compositor'
 'pipeline'
 'alglib'
 'datalib'

 # the settings module
 'settings',

 # developer API goes here. i.e. any non-public functions, or rarely used utility functions etc.
 'core'
 ]
```

### Developer API

The Developer API follows the hierarchical structure of the `core` directory. 
All source code written into the `core` podpac module should reference other modules using the full path to the module to maintain consistency.

For example:

```python
import podpac

dir(podac.core)
[
 'algorithm',
 'compositor',
 'coordinate',
 'data',
 'node',
 'pipeline',
 'units',
 'utils'
 ...
 ]
```

In source code `/podpac/core/node.py`:

```python
"""
Podpac Module
"""

...

from podpac import settings
from podpac.core.units import Units, UnitsDataArray
from podpac.core.coordinates.coordinates import Coordinates
from podpac.core.utils import common_doc
```


**Note**: The modules `podpac.settings` and `podpac.units.ureg` MUST be imported without using the `from` syntax. For example:

```python
import podpac.settings                 # yes
from podpac.settings import CACHE_CIR  # no
```

## Testing

We use `pytest` to run unit tests. To run tests, run from the root of the repository:

```
$ pytest
$ pytest -k "TestClass"    # run only the TestClass
```

Configuration options are specified in `setup.cfg`.


### Integration testing

We use `pytest` to write integration tests. 
Generally these tests should be written in seperate files from unit tests.
To specify that a test is an integration test, use the custom pytest marker `@pytest.mark.integration`:

```python
import pytest

@pytest.mark.integration
def test_function():
    pass

@pytest.mark.integration
class TestClass(object):

    def test_method(self):
        pass
```

Integration tests do not run by default.
To run integration tests from the command line:

```bash
$ pytest -m integration
```

See [working with custom markers](https://docs.pytest.org/en/latest/example/markers.html) for more details on how to use markers in pytest.

### Code Coverage

We use [`pytest-cov`](https://github.com/pytest-dev/pytest-cov) (which uses [`coverage`](https://coverage.readthedocs.io/en/coverage-4.5.1/) underneath) to monitor code coverage of unit tests. To record coverage while running tests, run:

```bash
$ pytest --cov=podpac --cov-report html:./artifacts/coverage podpac   # outputs html coverage to directory artifacts/coverage
```

We use [`coveralls`](https://github.com/coveralls-clients/coveralls-python) to provide [coverage status and visualization](https://coveralls.io/github/creare-com/podpac). Commits will be marked as failing if coverage drops below 90% or drops by more than 0.5%.

## Governance

* We encourage and welcome contributions from the wider community
* Presently, a small group of core developers decide which contributions will be incorporated
    * This is a complex software library
    * Until the library is mature, the interfaces and features need tight control
    * Missing functionality for your project can be implemented as a 3rd party plugin
    * For now, we are trying to be disciplined to avoid feature creep. 

