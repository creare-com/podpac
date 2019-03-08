
from podpac.core.settings import settings

original_default_cache = settings['DEFAULT_CACHE']

def pytest_sessionstart(session):
    settings['DEFAULT_CACHE'] = []

def pytest_sessionfinish(session, exitstatus):
    settings['DEFAULT_CACHE'] = original_default_cache