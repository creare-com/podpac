class CacheException(Exception):
    pass


class CacheWildCard(object):
    """Represent wildcard matches for inputs to remove operations (`rem`)
    that can match multiple items in the cache.
    """

    def __eq__(self, other):
        return True
