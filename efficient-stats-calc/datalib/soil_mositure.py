import numpy as np

import soilmap
import podpac
import datetime

def calculate_cdf(data):
    sorted_data = np.sort(data)
    probabilities = np.linspace(0, 1, len(sorted_data))
    return probabilities

def calculate_pdf(data, days):
    # Calculate the kernel density estimate
    kde = stats.gaussian_kde(data)

    x_values = np.linspace(min(data), max(data), days)
    
    # Evaluate the kde on the data
    pdf = kde.evaluate(x_values)
    
    return pdf


def get_soil_moisture_pdfs(years, month, lat_clinspace, lon_clinspace, days):

    sm = soilmap.datalib.geowatch.SoilMoisture(cache_ctrl=['ram']).solmst_0_10
    pdfs = []

    for year in years:
        # Create a list of dates for the specified month up to and including the 28th day
        dates = [f"{year}-{month:02d}-{day:02d}" for day in range(1, days-1)]

        soil_moisture_data = []

        # request the data one day at a time:
        for date in dates:
            # Get the coordinates for the current date
            current_c = podpac.Coordinates([lat_clinspace, lon_clinspace, date], ['lat', 'lon', 'time'])
            # Get soil moisture data for the given coordinates
            o = sm.eval(current_c)
            soil_moisture_data.append(o.values[1][1][0])
        
        # Sanitize soil_moisture_data for nans:
        soil_moisture_data = np.array(soil_moisture_data)
        soil_moisture_data = soil_moisture_data[~np.isnan(soil_moisture_data)]
        print(soil_moisture_data)
        # if soil moisture data is not empty:
        if len(soil_moisture_data) > 0:
            pdfs.append(calculate_pdf(soil_moisture_data, days))
            print(year)

    return np.array(pdfs)

def get_soil_moisture_levels(years, month, lat_clinspace, lon_clinspace, days):
    sm = soilmap.datalib.geowatch.SoilMoisture().solmst_0_10
    soil_moisture_data_all_years = []

    for year in years:
        # Create a list of dates for the specified month up to and including the 28th day
        dates = [f"{year}-{month:02d}-{day:02d}" for day in range(1, days-1)]

        soil_moisture_data = []

        # request the data one day at a time:
        for date in dates:
            # Get the coordinates for the current date
            current_c = podpac.Coordinates([lat_clinspace, lon_clinspace, date], ['lat', 'lon', 'time'])
            print(current_c)
            # Get soil moisture data for the given coordinates
            o = sm.eval(current_c)

            if o is np.nan:
                continue
            print(o)
            soil_moisture_data.append(o.values[1][1][0])

        soil_moisture_data_all_years.append(soil_moisture_data)

    return np.array(soil_moisture_data_all_years)

from scipy import stats



month = 6
years = range(2012, 2023)
lat_clinspace = podpac.clinspace(43.09, 42.91, 3, 'lat')
lon_clinspace = podpac.clinspace(-73.155, -72.91, 3, 'lon')
soil_moisture_data = get_soil_moisture_pdfs(years, month, lat_clinspace, lon_clinspace, 28)
print(soil_moisture_data)
podpac.utils.clear_cache(mode="all")
np.save("datalib/data/june_soil_moisture_pdfs.npy", soil_moisture_data)