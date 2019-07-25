import pytest
import os

from podpac.core.settings import PodpacSettings


class TestSettingsFile(object):
    def tmp_dir_path(self):
        basedir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(basedir, ".__tmp__")
        return path

    def make_settings_tmp_dir(self):
        path = self.tmp_dir_path()
        os.mkdir(path)  # intentionally fails if this folder already exists as it will be deleted
        return path

    def teardown_method(self):
        path = self.tmp_dir_path()
        try:
            os.remove(os.path.join(path, "settings.json"))
        except OSError:  # FileNotFoundError in py 3
            pass

        os.rmdir(path)  # intentionally fails if anything else is in this folder

    def test_settings_file_defaults_to_home_dir(self):
        self.make_settings_tmp_dir()  # so teardown method has something ot tear down
        settings = PodpacSettings()
        path = os.path.expanduser("~")
        assert settings.settings_path == os.path.join(path, ".podpac", "settings.json")

    def test_single_saved_setting_persists(self):
        path = self.make_settings_tmp_dir()

        key = "key"
        value = "value"
        settings = PodpacSettings()
        settings.load(path=path)
        settings["AUTOSAVE_SETTINGS"] = True
        settings[key] = value

        new_settings = PodpacSettings()
        new_settings.load(path=path)
        assert new_settings[key] == value

    def test_multiple_saved_settings_persist(self):
        path = self.make_settings_tmp_dir()

        key1 = "key1"
        value1 = "value1"
        settings = PodpacSettings()
        settings.load(path=path)
        settings["AUTOSAVE_SETTINGS"] = True
        settings[key1] = value1

        key2 = "key2"
        value2 = "value2"
        settings[key2] = value2

        new_settings = PodpacSettings()
        new_settings.load(path=path)
        assert new_settings[key1] == value1
        assert new_settings[key2] == value2

    def test_misconfigured_settings_file_fall_back_on_default(self):
        path = self.make_settings_tmp_dir()

        with open(os.path.join(path, "settings.json"), "w") as f:
            f.write("not proper json")

        settings = PodpacSettings()
        settings.load(path=path)
        assert isinstance(settings, dict)

        path = os.path.expanduser("~")
        assert settings.settings_path == os.path.join(path, ".podpac", "settings.json")
