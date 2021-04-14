import warnings
import pytest
import podpac


@pytest.mark.integration
class TestWeatherCitizen(object):
    data_key = "pressure"
    uuid = "re5wm615"

    def test_eval_source_coordinates(self):
        node = podpac.datalib.weathercitizen.WeatherCitizen(data_key=self.data_key, uuid=self.uuid)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="parsing timezone aware datetimes is deprecated")
            o = node.eval(node.coordinates[:3])

    def test_eval_interpolated(self):
        node = podpac.datalib.weathercitizen.WeatherCitizen(data_key=self.data_key, uuid=self.uuid)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="parsing timezone aware datetimes is deprecated")
            o = node.eval(podpac.Coordinates([0, 0], dims=["lat", "lon"]))
