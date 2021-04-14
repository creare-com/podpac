import numpy as np

import podpac
import podpac.datalib

# Create some nodes to help get realistic coordinates
cosmos = podpac.datalib.cosmos_stations.COSMOSStations()
soilscape = podpac.datalib.soilscape.SoilSCAPE20min(site="Canton_OK")

# Now do the coordinates
time_points = podpac.crange("2016-01-01", "2016-02-01", "1,D", "time")
# Soilscape coordinates
soilscape_points = soilscape.make_coordinates(time="2016-01-01")
soilscape_region = podpac.Coordinates(
    [
        podpac.clinspace(soilscape_points["lat"].bounds[1], soilscape_points["lat"].bounds[0], 64),
        podpac.clinspace(soilscape_points["lon"].bounds[0], soilscape_points["lon"].bounds[1], 64),
        "2016-01-01",
        4.0,
    ],
    dims=["lat", "lon", "time", "alt"],
)
soilscape_timeseries = podpac.coordinates.merge_dims(
    [soilscape_points[:2].drop("time"), podpac.Coordinates([time_points], crs=soilscape_points.crs)]
)

# COSMOS coordinates
cosmos_points = cosmos.source_coordinates.select({"lat": [36, 37], "lon": [-98, -97]})
cosmos_region = podpac.Coordinates(
    [
        podpac.clinspace(cosmos_points["lat"].bounds[1], cosmos_points["lat"].bounds[0], 64),
        podpac.clinspace(cosmos_points["lon"].bounds[0], cosmos_points["lon"].bounds[1], 64),
        "2016-01-01",
    ],
    dims=["lat", "lon", "time"],
)
cosmos_timeseries = podpac.coordinates.merge_dims(
    [cosmos_points, podpac.Coordinates([time_points], crs=cosmos_points.crs)]
)

COORDINATES = {
    "soilscape_points": soilscape_points,
    "soilscape_region": soilscape_region,
    "soilscape_timeseries": soilscape_timeseries,
    "cosmos_region": cosmos_region,
    "cosmos_timeseries": cosmos_timeseries,
}
