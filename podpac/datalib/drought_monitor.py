from podpac.algorithm import Algorithm
from podpac.data import Zarr
from podpac.style import Style
from podpac.utils import NodeTrait


class DroughtMonitorCategory(Zarr):
    style = Style(clim=[0, 0.6], colormap="gist_earth_r")


class DroughtCategory(Algorithm):
    # soil_moisture = NodeTrait().tag(attr=True, required=True)
    # d0 = NodeTrait().tag(attr=True, required=True)
    # d1 = NodeTrait().tag(attr=True, required=True)
    # d2 = NodeTrait().tag(attr=True, required=True)
    # d3 = NodeTrait().tag(attr=True, required=True)
    # d4 = NodeTrait().tag(attr=True, required=True)
    soil_moisture = NodeTrait().tag(attr=True)
    d0 = NodeTrait().tag(attr=True)
    d1 = NodeTrait().tag(attr=True)
    d2 = NodeTrait().tag(attr=True)
    d3 = NodeTrait().tag(attr=True)
    d4 = NodeTrait().tag(attr=True)

    style = Style(
        clim=[0, 6],
        enumeration_colors={
            0: (0.45098039, 0.0, 0.0, 1.0),
            1: (0.90196078, 0.0, 0.0, 1.0),
            2: (1.0, 0.66666667, 0.0, 1.0),
            3: (0.98823529, 0.82745098, 0.49803922, 1.0),
            4: (1.0, 1.0, 0.0, 1.0),
            5: (1.0, 1.0, 1.0, 0.0),
        },
    )

    def algorithm(self, inputs, coordinates):
        sm = inputs["soil_moisture"]
        d0 = inputs["d0"]
        d1 = inputs["d1"]
        d2 = inputs["d2"]
        d3 = inputs["d3"]
        d4 = inputs["d4"]

        return (
            (sm >= 0) * (sm < d4) * ((sm - 0) / (d4 - 0) + 0)
            + (sm >= d4) * (sm < d3) * ((sm - d4) / (d3 - d4) + 1)
            + (sm >= d3) * (sm < d2) * ((sm - d3) / (d2 - d3) + 2)
            + (sm >= d2) * (sm < d1) * ((sm - d2) / (d1 - d2) + 3)
            + (sm >= d1) * (sm < d0) * ((sm - d1) / (d0 - d1) + 4)
            + (sm >= d0) * (sm < 0.75) * ((sm - d0) / (0.75 - d1) + 5)
            + (sm >= 0.75) * 6
        )


if __name__ == "__main__":
    import os
    import numpy as np
    import podpac

    c = podpac.Coordinates([46.6, -123.5, "2018-06-01"], dims=["lat", "lon", "time"])

    # local
    path = "droughtmonitor/beta_parameters.zarr"
    if not os.path.exists(path):
        print("No local drought monitor data found at '%s'" % path)
    else:
        # drought monitor parameters
        d0 = DroughtMonitorCategory(source=path, data_key="d0")
        print(d0.coordinates)
        print(d0.eval(c))

        # drought category
        mock_sm = podpac.data.Array(data=np.random.random(d0.coordinates.shape), coordinates=d0.coordinates)

        category = DroughtCategory(
            soil_moisture=mock_sm,
            d0=DroughtMonitorCategory(source=path, data_key="d0"),
            d1=DroughtMonitorCategory(source=path, data_key="d1"),
            d2=DroughtMonitorCategory(source=path, data_key="d2"),
            d3=DroughtMonitorCategory(source=path, data_key="d3"),
            d4=DroughtMonitorCategory(source=path, data_key="d4"),
        )
        print(category.eval(c))

    # s3
    bucket = "podpac-internal-test"
    store = "drought_parameters.zarr"
    path = "s3://%s/%s" % (bucket, store)
    d0 = DroughtMonitorCategory(source=path, data_key="d0")
    if not d0.s3.exists(path):
        print("No drought monitor data found at '%s'. Check your AWS credentials." % path)
    else:
        print(d0.coordinates)
        print(d0.eval(c))

        # drought category algorithm
        mock_sm = podpac.data.Array(source=np.random.random(d0.coordinates.shape), coordinates=d0.coordinates)

        category = DroughtCategory(
            soil_moisture=mock_sm,
            d0=DroughtMonitorCategory(source=path, data_key="d0"),
            d1=DroughtMonitorCategory(source=path, data_key="d1"),
            d2=DroughtMonitorCategory(source=path, data_key="d2"),
            d3=DroughtMonitorCategory(source=path, data_key="d3"),
            d4=DroughtMonitorCategory(source=path, data_key="d4"),
        )
        print(category.eval(c))
