import pytest
import os

import time

from podpac.core.managers.multi_threading import FakeLock
from threading import Thread


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

        t1 = Thread(target=lambda: f("thread"), daemon=True)
        t2 = Thread(target=lambda: f("thread"), daemon=True)
        print("In Main Thread")
        f("main1")
        print("Starting Thread")
        t1.run()
        t2.run()
        f("main2")
