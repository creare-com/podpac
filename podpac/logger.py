"""
PODPAC Logging

Logging settings are configured in :attr:`podpac.settings.LOG`
See https://docs.python.org/3/library/logging.config.html#logging.config.dictConfig
and https://docs.python.org/3/howto/logging-cookbook.html#an-example-dictionary-based-configuration
"""

import logging
import logging.config

from podpac import settings


CONFIG = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '[%(levelname)s] %(message)s'
        },
        'verbose': {
            'class': 'logging.Formatter',
            'format': '[%(asctime)s] %(levelname)-8s %(message)s'
        }
    },
    'handlers':{
        # output to console
        'console': {
            'class': 'logging.StreamHandler',
            'level': logging.WARNING,
            'formatter': 'simple',
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': logging.DEBUG,
            'filename': 'podpac.log',
            'mode': 'w',
            'formatter': 'verbose',
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'file']
    }
}

# override CONFIG from settings
if settings.LOG is not None:
    CONFIG = settings.LOG

# set initial config
logging.config.dictConfig(CONFIG)

# load default logger
log = logging.getLogger()


def set_level(level, handler=None):
    """Set the logging level for the console, file, or both
    
    Parameters
    ----------
    level : int
        Logging level from 0 (DEBUG) to 50 (CRITICAL).
        See https://docs.python.org/3/library/logging.html#logging-levels
    handler : str, optional
        Log handler to which level applies, either 'console' or 'file'.
        If None, the level will be set to both handlers.
    """

    if not isinstance(level, int):
        raise ValueError('level input must be an int')

    if handler is not None:
        if handler in ['console', 'file']:
            CONFIG['handlers'][handler]['level'] = level
        else:
            log.warning('log handler input must be one of "console" or "file"')
    else:
        CONFIG['handlers']['console']['level'] = level
        CONFIG['handlers']['file']['level'] = level

    logging.config.dictConfig(CONFIG)
    log = logging.getLogger()


def set_filename(filename):
    """Set log filename
    
    Parameters
    ----------
    filename : str
        log filename
    """

    CONFIG['handlers']['file']['filename'] = filename

    logging.config.dictConfig(CONFIG)
    log = logging.getLogger()


def disable(handler):
    """Disable logging to certain handler
    
    Parameters
    ----------
    handler : str
        Log handler to disable -'console' or 'file'.
    """

    if handler not in ['console', 'file']:
        log.warning('log handler input must be one of "console" or "file"')
    elif handler in CONFIG['root']['handlers']:
        CONFIG['root']['handlers'] = list(set(CONFIG['root']['handlers']) - set([handler]))
        logging.config.dictConfig(CONFIG)
        log = logging.getLogger()
