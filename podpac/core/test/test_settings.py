import os
import pytest

from podpac.core.settings import PodpacSettings

_SETTINGS_FILENAME = "settings.json"


class TestSettingsFile(object):
    @pytest.fixture
    def settings_tmp_dir(self, tmp_path):
        path = os.path.join(tmp_path, ".__tmp__")
        os.mkdir(path)  # intentionally fails if this folder already exists as it will be deleted
        yield path
        try:
            os.remove(os.path.join(path, _SETTINGS_FILENAME))
        except OSError:  # FileNotFoundError in py 3
            pass
        os.rmdir(path)  # intentionally fails if anything else is in this folder

    def test_settings_file_defaults_to_home_dir(self):
        settings = PodpacSettings()
        path = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~"))
        assert settings.settings_path == os.path.join(path, ".config", "podpac", _SETTINGS_FILENAME)

    def test_single_saved_setting_persists(self, settings_tmp_dir):
        path = settings_tmp_dir

        key = "key"
        value = "value"
        settings = PodpacSettings()
        settings.load(path=path)
        settings["AUTOSAVE_SETTINGS"] = True
        settings[key] = value

        new_settings = PodpacSettings()
        new_settings.load(path=path)
        assert new_settings[key] == value

    def test_multiple_saved_settings_persist(self, settings_tmp_dir):
        path = settings_tmp_dir

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

    def test_misconfigured_settings_file_fall_back_on_default(self, settings_tmp_dir):
        path = settings_tmp_dir

        with open(os.path.join(path, _SETTINGS_FILENAME), "w") as f:
            f.write("not proper json")

        settings = PodpacSettings()
        settings.load(path=path)
        assert isinstance(settings, dict)

        path = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~"))
        assert settings.settings_path == os.path.join(path, ".config", "podpac", _SETTINGS_FILENAME)
