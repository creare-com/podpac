import os
import sys
import time
from threading import Thread

import pytest

from podpac import settings
from podpac.core.managers.multi_threading import FakeLock, thread_manager


class TestFakeLock(object):
    def test_enter_exist_single_thread(self):
        lock = FakeLock()
        assert lock._locked == False
        with lock:
            assert lock._locked
        assert lock._locked == False

    def test_fake_lock_multithreaded(self):
        lock = FakeLock()

        def f(s):
            print("In", s)
            with lock:
                print("Locked", s)
                assert lock._locked
                time.sleep(0.05)
            print("Unlocked", s)
            assert lock._locked == False

        if sys.version_info.major == 2:
            t1 = Thread(target=lambda: f("thread"))
            t2 = Thread(target=lambda: f("thread"))
            t1.daemon = True
            t2.daemon = True
        else:
            t1 = Thread(target=lambda: f("thread"), daemon=True)
            t2 = Thread(target=lambda: f("thread"), daemon=True)
        print("In Main Thread")
        f("main1")
        print("Starting Thread")
        t1.run()
        t2.run()
        f("main2")


class TestThreadManager(object):
    def test_request_release_threads_single_threaded(self):
        with settings:
            settings["N_THREADS"] = 5
            # Requests
            n = thread_manager.request_n_threads(3)
            assert n == 3
            n = thread_manager.request_n_threads(3)
            assert n == 2
            n = thread_manager.request_n_threads(3)
            assert n == 0

            # releases
            assert thread_manager._n_threads_used == 5
            n = thread_manager.release_n_threads(3)
            assert n == 3
            n = thread_manager.release_n_threads(2)
            assert n == 5
            n = thread_manager.release_n_threads(50)
            assert n == 5

    def test_request_release_threads_multi_threaded(self):
        def f(s):
            print("In", s)
            n1 = thread_manager.release_n_threads(s)
            time.sleep(0.05)
            n2 = thread_manager.release_n_threads(s)
            print("Released", s)
            assert n2 >= n1

        with settings:
            settings["N_THREADS"] = 7

            if sys.version_info.major == 2:
                t1 = Thread(target=lambda: f(5))
                t2 = Thread(target=lambda: f(6))
                t1.daemon = True
                t2.daemon = True
            else:
                t1 = Thread(target=lambda: f(5), daemon=True)
                t2 = Thread(target=lambda: f(6), daemon=True)
            f(1)
            t1.run()
            t2.run()
            f(7)
