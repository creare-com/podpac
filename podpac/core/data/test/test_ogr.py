import pytest

import podpac
from podpac.core.data.ogr import OGR


@pytest.mark.skip("No test file available yet")
class TestOGR(object):
    source = "TODO"
    driver = "ESRI Shapefile"
    layer = "TODO"
    attribute = "TODO"

    def test_extents(self):
        node = OGR(source=self.source, driver=self.driver, layer=self.layer, attribute=self.attribute)
        node.extents

    def test_eval_uniform(self):
        node = OGR(source=self.source, driver=self.driver, layer=self.layer, attribute=self.attribute)
        coords = podpac.Coordinates([podpac.clinspace(43, 44, 10), podpac.clinspace(-73, -72, 10)], dims=["lat", "lon"])
        output = node.eval(coords)

    def test_eval_point(self):
        node = OGR(source=self.source, driver=self.driver, layer=self.layer, attribute=self.attribute)
        coords = podpac.Coordinates([43.7, -72.3], dims=["lat", "lon"])
        output = node.eval(coords)

    def test_eval_stacked(self):
        node = OGR(source=self.source, driver=self.driver, layer=self.layer, attribute=self.attribute)
        coords = podpac.Coordinates([[[43, 43.5, 43.7], [-72.0, -72.5, -72.7]]], dims=["lat_lon"])
        output = node.eval(coords)

    @pytest.mark.skip(reason="not yet implemented")
    def test_eval_nonuniform(self):
        node = OGR(source=self.source, driver=self.driver, layer=self.layer, attribute=self.attribute)
        coords = podpac.Coordinates([[43, 43.5, 43.7], [-72.0, -72.5, -72.7]], dims=["lat", "lon"])
        output = node.eval(coords)

    def test_eval_extra_dims(self):
        node = OGR(source=self.source, driver=self.driver, layer=self.layer, attribute=self.attribute)
        coords = podpac.Coordinates(
            [podpac.clinspace(43, 44, 10), podpac.clinspace(-73, -72, 10), "2018-01-01"], dims=["lat", "lon", "time"]
        )
        output = node.eval(coords)

    def test_eval_missing_dims(self):
        node = OGR(source=self.source, driver=self.driver, layer=self.layer, attribute=self.attribute)
        coords = podpac.Coordinates(["2018-01-01"], dims=["time"])
        with pytest.raises(RuntimeError, match="OGR source requires lat and lon dims"):
            output = node.eval(coords)
