import datetime
from six import string_types
import numpy as np


class CacheException(Exception):
    pass


class CacheWildCard(object):
    """Represent wildcard matches for inputs to remove operations (`rem`)
    that can match multiple items in the cache.
    """

    def __eq__(self, other):
        return True


def expiration_timestamp(value):
    """
    Parse and calculate an expiration timestamp.

    Arguments
    ---------
    value : float, datetime, timedelta, str
            User-friendly expiration value.
             * string values are parsed as datetime or timedelta.
             * timedeltas are added to the current time.
             * floats are interpreted as timestamps

    Returns
    -------
    expiration : float
            expiration timestamp
    """

    if value is None:
        return None

    if isinstance(value, float):
        return value

    expires = value

    # parse string datetime or timedelta
    if isinstance(expires, string_types):
        try:
            expires = np.datetime64(expires).item()
        except:
            pass

    if isinstance(expires, string_types) and "," in expires:
        try:
            expires = np.timedelta64(*expires.split(",")).item()
        except:
            pass

    # extract datetime or timedelta from numpy types
    if isinstance(expires, (np.datetime64, np.timedelta64)):
        expires = expires.item()

    # calculate and return expiration date
    if isinstance(expires, datetime.datetime):
        return expires.timestamp()
    elif isinstance(expires, datetime.date):
        return datetime.datetime.combine(expires, datetime.datetime.min.time()).timestamp()
    elif isinstance(expires, datetime.timedelta):
        return (datetime.datetime.now() + expires).timestamp()
    elif isinstance(value, string_types):
        raise ValueError("Invalid expiration date or delta '%s'" % value)
    else:
        raise TypeError("Invalid expiration date or delta '%s' of type %s" % (value, type(value)))
