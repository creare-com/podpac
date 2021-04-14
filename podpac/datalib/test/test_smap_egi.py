import pytest
import podpac


@pytest.mark.integration
class TestSMAP_EGI(object):
    def test_eval_level_3(self):
        # level 3 access
        c = podpac.Coordinates(
            [
                podpac.clinspace(-82, -81, 10),
                podpac.clinspace(38, 39, 10),
                podpac.clinspace("2015-07-06", "2015-07-08", 10),
            ],
            dims=["lon", "lat", "time"],
        )

        node = podpac.datalib.smap_egi.SMAP(product="SPL3SMP_AM")
        output = node.eval(c)
        print(output)
