from podpac.core.data.types import Zarr
from podpac.core.coordinates import ArrayCoordinates1d


class DroughtMonitorCategory(Zarr):
    dims = ["lat", "lon", "time"]
    cf_time = True
    cf_units = "days since 2018-01-01 00:00:00"
    cf_calendar = "proleptic_gregorian"


if __name__ == "__main__":
    import podpac

    c = podpac.Coordinates([46.6, -123.5, "2018-06-01"], dims=["lat", "lon", "time"])

    # local
    path = "droughtmonitor/beta_parameters.zarr"
    d0 = DroughtMonitorCategory(source=path, datakey="d0")
    print(d0.native_coordinates)
    print(d0.eval(c))

    # s3
    bucket = "podpac-internal-test"
    store = "drought_parameters.zarr"
    path = "s3://%s/%s" % (bucket, store)
    d0 = DroughtMonitorCategory(source=path, datakey="d0")
    print(d0.native_coordinates)
    print(d0.eval(c))

    # the Zarr node uses the podpac AWS settings by default, but credentials can be explicitly provided, too
    d0 = DroughtMonitorCategory(
        source=path,
        datakey="d0",
        access_key_id=podpac.settings["AWS_ACCESS_KEY_ID"],
        secret_access_key=podpac.settings["AWS_SECRET_ACCESS_KEY"],
        region_name=podpac.settings["AWS_REGION_NAME"],
    )
