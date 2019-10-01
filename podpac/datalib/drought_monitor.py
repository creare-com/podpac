from podpac.core.node import Node
from podpac.core.style import Style
from podpac.core.utils import NodeTrait
from podpac.core.algorithm.algorithm import Algorithm
from podpac.core.data.types import Zarr
from podpac.core.coordinates import ArrayCoordinates1d


class DroughtMonitorCategory(Zarr):
    dims = ["lat", "lon", "time"]
    cf_time = True
    cf_units = "days since 2018-01-01 00:00:00"
    cf_calendar = "proleptic_gregorian"


class DroughtCategory(Algorithm):
    soil_moisture = NodeTrait()
    d0 = NodeTrait()
    d1 = NodeTrait()
    d2 = NodeTrait()
    d3 = NodeTrait()
    d4 = NodeTrait()
    style = Style(
        clim=[0, 6],
        enumeration_colors=[
            [0.45098039, 0.0, 0.0, 1.0],
            [0.90196078, 0.0, 0.0, 1.0],
            [1.0, 0.66666667, 0.0, 1.0],
            [0.98823529, 0.82745098, 0.49803922, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
    )

    def algorithm(self, inputs):
        sm = inputs["soil_moisture"]
        d0 = inputs["d0"]
        d1 = inputs["d1"]
        d2 = inputs["d2"]
        d3 = inputs["d3"]
        d4 = inputs["d4"]

        return (
            (sm > 0) * (sm < d4) * ((sm - 0) / (d4 - 0) + 0)
            + (sm > d4) * (sm < d3) * ((sm - d4) / (d3 - d4) + 1)
            + (sm > d3) * (sm < d2) * ((sm - d3) / (d2 - d3) + 2)
            + (sm > d2) * (sm < d1) * ((sm - d2) / (d1 - d2) + 3)
            + (sm > d1) * (sm < d0) * ((sm - d1) / (d0 - d1) + 4)
            + (sm > d0) * (sm < 0.75) * ((sm - d0) / (0.75 - d1) + 5)
            + (sm > d0) * 6
        )


if __name__ == "__main__":
    import podpac

    c = podpac.Coordinates([46.6, -123.5, "2018-06-01"], dims=["lat", "lon", "time"])

    # local
    path = "droughtmonitor/beta_parameters.zarr"
    d0 = DroughtMonitorCategory(source=path, datakey="d0")
    print (d0.native_coordinates)
    print (d0.eval(c))

    # s3
    bucket = "podpac-internal-test"
    store = "drought_parameters.zarr"
    path = "s3://%s/%s" % (bucket, store)
    d0 = DroughtMonitorCategory(source=path, datakey="d0")
    print (d0.native_coordinates)
    print (d0.eval(c))

    # the Zarr node uses the podpac AWS settings by default, but credentials can be explicitly provided, too
    d0 = DroughtMonitorCategory(
        source=path,
        datakey="d0",
        access_key_id=podpac.settings["AWS_ACCESS_KEY_ID"],
        secret_access_key=podpac.settings["AWS_SECRET_ACCESS_KEY"],
        region_name=podpac.settings["AWS_REGION_NAME"],
    )
