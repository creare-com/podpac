"""
Daymet 

Daymet provides long-term, continuous, gridded estimates of daily weather and climatology variables by 
interpolating and extrapolating ground-based observations through statistical modeling techniques.
https://daymet.ornl.gov/

    - Web Services Downloads: (https://daymet.ornl.gov/web_services)
    - Batch Downlaods examples: (https://github.com/ornldaac/gridded_subset_example_script)

Daymet data is defined in Lamert Conformal Conic (LCC) Projection system.
"""


from podpac.core.data.datasource import DataSource
import OrderedDict
import traitlets as tl
from podpac.core.data.rasterio_source import Rasterio

# create logger for module1
_logger = logging.getLogger(__name__)


class Daymet(Datasource):
    ''' Datasource to handle Daymet

    Daymet is accessed over HTTP, and the response is a NetCDF file.

    A standard DAYMET request is done over http, and the URL follows this pattern:
    https://thredds.daac.ornl.gov/thredds/ncss/grid/ornldaac/2129/`daymet_v4_daily_[region]_[DAYMETVAR]_[YEAR].nc
    
    Where [DAYMETVAR] can be:
        - tmax: Daily maximum 2-meter air temperature (Celsius)
        - tmin: Daily minimum 2-meter air temperature (Celsius)
        - prcp: Dailt total precipitation (mm/day)
        - srad: Incident shortwave radiation flux density (W/m2)
        - vp: Water vapor pressure (Pa)
        - swe: Snow water equivalent (kg/m2)
        - dayl: Duration of the daylight period (seconds/day)
    Where [YEAR] is
        4 digit year, most recent full calendar year
    Where [region] is 
        `na` for North America

    A *full* standard request from the NCSS service (getting subsets, etc) will take the following pattern:
    https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/2129/daymet_v4_daily_[region]_[DAYMETVAR]_[YEAR].nc?var=lat&var=lon&var=[DAYMETVAR]&north=&west=&east=&south=&disableProjSubset=on&horizStride=1&time_start=Z&time_end=&timeStride=&accept=netcdf

    Attributes:
        - north: The northern extent of the bounding box (latitude in decimal degrees) of the subset
        - west: The western extent of the bounding box (longitude in decimal degrees) of the subset
        - east: The eastern extent of the bounding box (longitude in decimal degrees) of the subset
        - south: The southern extent of the bounding box (latitude in decimal degrees) of the subset
        - horizStride: Will take every nth point (in both x and Y) of the gridded dataset. The default, "1", will take every point
        - time_start: The beginning of the time range. Specify a time range subset in the form: yyyy '-' mm '-' dd 'T' hh ':' mm ':' ss Z
        - time_end: The end of the time range. Specify a time range subset in the form: yyyy '-' mm '-' dd 'T' hh ':' mm ':' ss Z
        - time_stride: Will take only every nth time in the available series on gridded datasets. The default, "1", will take every time step
        - accept: The format of the subset data returned by the NCSS: "netcdf" for netCDF v3 format is the only option currently available

    '''

    
