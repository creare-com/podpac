"""
Module for dealing with multi-threaded execution. 

This is used to ensure that the total number of threads specified in the settings is not exceeded. 

"""

from __future__ import division, unicode_literals, print_function, absolute_import

from multiprocessing import Lock

from podpac.core.settings import settings

DEFAULT_N_THREADS = 10


class ThreadManager(object):
    """ This is a singleton class that keeps track of the total number of threads used in an application. 
    """

    _lock = Lock()
    _n_threads_used = 0
    __instance = None

    def __new__(cls):
        if ThreadManager.__instance is None:
            ThreadManager.__instance = object.__new__(cls)
        return ThreadManager.__instance

    def request_n_threads(self, n):
        """ Returns the number of threads allowed for a pool taking into account all other threads application, as 
        specified by podpac.settings["N_THREADS"].
        
        Parameters
        -----------
        n : int
            Number of threads requested by operation
            
        Returns
        --------
        int 
            Number of threads a pool may use. Note, this may be less than or equal to n, and may be 0. 
        """
        self._lock.acquire()
        available = max(0, settings.get("N_THREADS", DEFAULT_N_THREADS) - self._n_threads_used)
        claimed = min(available, n)
        self._n_threads_used += claimed
        self._lock.release()
        return claimed

    def release_n_threads(self, n):
        """ This releases the number of threads specified. 
        
        Parameters
        ------------
        n : int
            Number of threads to be released
            
        Returns
        --------
        int
            Number of threads available after releases 'n' threads
        """
        self._lock.acquire()
        self._n_threads_used = max(0, self._n_threads_used - n)
        available = max(0, settings.get("N_THREADS", DEFAULT_N_THREADS) - self._n_threads_used)
        self._lock.release()
        return available


thread_manager = ThreadManager()
