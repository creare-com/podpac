"""
Test Setup
"""

import copy
import pytest
from podpac.core.settings import settings


def pytest_addoption(parser):
    """Add command line option to pytest
    Note you MUST invoke test as `pytest podpac --ci` to use these options.
    Using only `pytest --ci` will result in an error

    Parameters
    ----------
    parser : TYPE

    """
    # config option for when we're running tests on ci
    parser.addoption("--ci", action="store_true", default=False)


def pytest_runtest_setup(item):
    markers = [marker.name for marker in item.iter_markers()]
    if item.config.getoption("--ci") and "aws" in markers:
        pytest.skip("Skip aws tests during CI")


def pytest_configure(config):
    """Configuration before all tests are run

    Parameters
    ----------
    config : TYPE

    """

    config.addinivalue_line("markers", "aws: mark test as an aws test")
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_unconfigure(config):
    """Configuration after all tests are run

    Parameters
    ----------
    config : TYPE

    """
    pass


original_settings = {}


def pytest_sessionstart(session):
    # save settings
    global original_settings
    original_settings = copy.copy(settings)

    # set the default cache
    settings["DEFAULT_CACHE"] = []


def pytest_sessionfinish(session, exitstatus):
    # restore settings
    keys = list(settings.keys())
    for key in keys:
        if key in original_settings:
            settings[key] = original_settings[key]
        else:
            del settings[key]
