"""
MODIS on AWS OpenData

MODIS Coordinates Grids: https://modis-land.gsfc.nasa.gov/MODLAND_grid.html
"""

import logging
import datetime

import numpy as np
import traitlets as tl

import podpac
from podpac.utils import cached_property
from podpac.compositor import TileCompositorRaw
from podpac.core.data.rasterio_source import RasterioRaw
from podpac.authentication import S3Mixin
from podpac.interpolators import InterpolationMixin

_logger = logging.getLogger(__name__)

BUCKET = "modis-pds"
PRODUCTS = ["MCD43A4.006", "MOD09GA.006", "MYD09GA.006", "MOD09GQ.006", "MYD09GQ.006"]
CRS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs +type=crs"

SINUSOIDAL_HORIZONTAL = {
    "00": (-20014877.697641734, -18903390.490691263),
    "01": (-18902927.177974734, -17791439.971025266),
    "02": (-17790976.658308737, -16679489.451358264),
    "03": (-16679026.138641736, -15567538.931691263),
    "04": (-15567075.618974736, -14455588.412025262),
    "05": (-14455125.099308735, -13343637.892358262),
    "06": (-13343174.579641735, -12231687.372691263),
    "07": (-12231224.059974736, -11119736.853025263),
    "08": (-11119273.540308736, -10007786.333358264),
    "09": (-10007323.020641735, -8895835.813691262),
    "10": (-8895372.500974735, -7783885.294025263),
    "11": (-7783421.981308736, -6671934.774358263),
    "12": (-6671471.461641735, -5559984.254691264),
    "13": (-5559520.941974737, -4448033.735025264),
    "14": (-4447570.422308736, -3336083.215358263),
    "15": (-3335619.902641736, -2224132.695691264),
    "16": (-2223669.382974736, -1112182.176025264),
    "17": (-1111718.863308736, -231.656358264),
    "18": (231.656358264, 1111718.863308736),
    "19": (1112182.176025264, 2223669.382974736),
    "20": (2224132.695691264, 3335619.902641736),
    "21": (3336083.215358264, 4447570.422308736),
    "22": (4448033.735025263, 5559520.941974737),
    "23": (5559984.254691265, 6671471.461641736),
    "24": (6671934.774358264, 7783421.981308737),
    "25": (7783885.294025264, 8895372.500974735),
    "26": (8895835.813691264, 10007323.020641737),
    "27": (10007786.333358264, 11119273.540308736),
    "28": (11119736.853025265, 12231224.059974737),
    "29": (12231687.372691264, 13343174.579641737),
    "30": (13343637.892358264, 14455125.099308737),
    "31": (14455588.412025264, 15567075.618974738),
    "32": (15567538.931691265, 16679026.138641737),
    "33": (16679489.451358264, 17790976.658308737),
    "34": (17791439.971025266, 18902927.177974734),
    "35": (18903390.490691263, 20014877.697641734),
}

SINUSOIDAL_VERTICAL = {
    "00": (10007323.020641735, 8895835.813691262),
    "01": (8895372.500974735, 7783885.294025263),
    "02": (7783421.981308736, 6671934.774358263),
    "03": (6671471.461641735, 5559984.254691264),
    "04": (5559520.941974737, 4448033.735025264),
    "05": (4447570.422308736, 3336083.215358263),
    "06": (3335619.902641736, 2224132.695691264),
    "07": (2223669.382974736, 1112182.176025264),
    "08": (1111718.863308736, 231.656358264),
    "09": (-231.656358264, -1111718.863308736),
    "10": (-1112182.176025264, -2223669.382974736),
    "11": (-2224132.695691264, -3335619.902641736),
    "12": (-3336083.215358264, -4447570.422308736),
    "13": (-4448033.735025263, -5559520.941974737),
    "14": (-5559984.254691265, -6671471.461641736),
    "15": (-6671934.774358264, -7783421.981308737),
    "16": (-7783885.294025264, -8895372.500974735),
    "17": (-8895835.813691264, -10007323.020641737),
}


def _parse_modis_date(date):
    return datetime.datetime.strptime(date, "%Y%j").strftime("%Y-%m-%d")


def _available(s3, *l):
    prefix = "/".join([BUCKET] + list(l))
    return [obj.replace(prefix + "/", "") for obj in s3.ls(prefix) if "_scenes.txt" not in obj]


def get_tile_coordinates(h, v):
    """use pre-fetched lat and lon bounds to get coordinates for a single tile"""
    lat_start, lat_stop = SINUSOIDAL_VERTICAL[v]
    lon_start, lon_stop = SINUSOIDAL_HORIZONTAL[h]
    lat = podpac.clinspace(lat_start, lat_stop, 2400, name="lat")
    lon = podpac.clinspace(lon_start, lon_stop, 2400, name="lon")
    return podpac.Coordinates([lat, lon], crs=CRS)


class MODISSource(RasterioRaw):
    """
    Individual MODIS data tile using AWS OpenData, with caching.

    Attributes
    ----------
    product : str
        MODIS product ('MCD43A4.006', 'MOD09GA.006', 'MYD09GA.006', 'MOD09GQ.006', or 'MYD09GQ.006')
    horizontal : str
        column in the MODIS Sinusoidal Tiling System, e.g. '21'
    vertical : str
        row in the MODIS Sinusoidal Tiling System, e.g. '07'
    date : str
        year and three-digit day of year, e.g. '2011260'
    data_key : str
        individual object (varies by product)
    """

    product = tl.Enum(values=PRODUCTS, help="MODIS product ID").tag(attr=True)
    horizontal = tl.Unicode(help="column in the MODIS Sinusoidal Tiling System, e.g. '21'").tag(attr=True)
    vertical = tl.Unicode(help="row in the MODIS Sinusoidal Tiling System, e.g. '07'").tag(attr=True)
    date = tl.Unicode(help="year and three-digit day of year, e.g. '2011460'").tag(attr=True)
    data_key = tl.Unicode(help="data to retrieve (varies by product)").tag(attr=True)
    anon = tl.Bool(True)
    check_exists = tl.Bool(True)

    _repr_keys = ["prefix", "data_key"]

    def init(self):
        """validation"""
        for key in ["horizontal", "vertical", "date", "data_key"]:
            if not getattr(self, key):
                raise ValueError("MODISSource '%s' required" % key)
        if self.horizontal not in ["%02d" % h for h in range(36)]:
            raise ValueError("MODISSource horizontal invalid ('%s' should be between '00' and '35')" % self.horizontal)
        if self.vertical not in ["%02d" % v for v in range(36)]:
            raise ValueError("MODISSource vertical invalid ('%s' should be between '00' and '17'" % self.vertical)
        try:
            _parse_modis_date(self.date)
        except ValueError:
            raise ValueError("MODISSource date invalid ('%s' should be year and doy, e.g. '2009260'" % self.date)
        if self.check_exists and not self.exists:
            raise ValueError("No S3 object found at '%s'" % self.source)

    @cached_property(use_cache_ctrl=True)
    def filename(self):
        _logger.info(
            "Looking up source filename (product=%s, h=%s, v=%s, date=%s, data_key=%s)..."
            % (self.product, self.horizontal, self.vertical, self.date, self.data_key)
        )
        prefix = "/".join([BUCKET, self.product, self.horizontal, self.vertical, self.date])
        objs = [obj.replace(prefix + "/", "") for obj in self.s3.ls(prefix) if obj.endswith("%s.TIF" % self.data_key)]
        if len(objs) == 0:
            raise RuntimeError("No matches found for data_key='%s' at '%s'" % (self.data_key, prefix))
        if len(objs) > 1:
            raise RuntimeError("Too many matches for data_key='%s' at '%s' (%s)" % (self.data_key, prefix, objs))
        return objs[0]

    @property
    def prefix(self):
        return "%s/%s/%s/%s" % (self.product, self.horizontal, self.vertical, self.date)

    @cached_property
    def source(self):
        return "s3://%s/%s/%s" % (BUCKET, self.prefix, self.filename)

    @cached_property
    def exists(self):
        return self.s3.exists(self.source)

    def get_coordinates(self):
        # use pre-fetched coordinate bounds (instead of loading from the dataset)
        spatial_coords = get_tile_coordinates(self.horizontal, self.vertical)
        time_coords = podpac.Coordinates([_parse_modis_date(self.date)], ["time"], crs=spatial_coords.crs)
        return podpac.coordinates.merge_dims([spatial_coords, time_coords])


class MODISComposite(S3Mixin, TileCompositorRaw):
    """MODIS whole-world compositor.
    For documentation about the data, start here: https://ladsweb.modaps.eosdis.nasa.gov/search/order/1
    For information about the bands, see here: https://modis.gsfc.nasa.gov/about/specifications.php

    Attributes
    ----------
    product : str
        MODIS product ('MCD43A4.006', 'MOD09GA.006', 'MYD09GA.006', 'MOD09GQ.006', or 'MYD09GQ.006')
    data_key : str
        individual object (varies by product)
    """

    product = tl.Enum(values=PRODUCTS, help="MODIS product ID").tag(attr=True, required=True)
    data_key = tl.Unicode(help="data to retrieve (varies by product)").tag(attr=True, required=True)

    tile_width = (1, 2400, 2400)
    start_date = "2013-01-01"
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    anon = tl.Bool(True)

    dims = ["time", "lat", "lon"]

    _repr_keys = ["product", "data_key"]

    @cached_property(use_cache_ctrl=True)
    def tile_coordinates(self):
        return [get_tile_coordinates(*hv) for hv in self.available_tiles]

    @cached_property(use_cache_ctrl=True)
    def available_tiles(self):
        _logger.info("Looking up available tiles...")
        return [(h, v) for h in _available(self.s3, self.product) for v in _available(self.s3, self.product, h)]

    def select_sources(self, coordinates, _selector=None):
        """2d select sources filtering"""

        # filter tiles spatially
        ct = coordinates.transform(CRS)
        tiles = [at for at, atc in zip(self.available_tiles, self.tile_coordinates) if ct.select(atc.bounds).size > 0]
        sources = []
        for tile in tiles:
            h, v = tile
            available_dates = _available(self.s3, self.product, h, v)
            dates = [_parse_modis_date(date) for date in available_dates]
            date_coords = podpac.Coordinates([dates], dims=["time"])
            # Filter individual tiles temporally
            if _selector is not None:
                _, I = _selector(date_coords, ct, index_type="numpy")
            else:
                _, I = date_coords.intersect(ct, outer=True, return_index=True)
            valid_dates = np.array(available_dates)[I]
            valid_sources = [
                MODISSource(
                    product=self.product,
                    horizontal=h,
                    vertical=v,
                    date=date,
                    data_key=self.data_key,
                    check_exists=False,
                    cache_ctrl=self.cache_ctrl,
                    force_eval=self.force_eval,
                    cache_output=self.cache_output,
                    cache_dataset=True,
                    s3=self.s3,
                )
                for date in valid_dates
            ]
            sources.extend(valid_sources)
        self.set_trait("sources", sources)
        return sources


class MODIS(InterpolationMixin, MODISComposite):
    pass


if __name__ == "__main__":
    from matplotlib import pyplot

    # -------------------------------------------------------------------------
    # basic modis source
    # -------------------------------------------------------------------------

    source = MODISSource(
        product=PRODUCTS[0],
        data_key="B01",
        horizontal="01",
        vertical="11",
        date="2020009",
        cache_ctrl=["disk"],
        cache_dataset=True,
        cache_output=False,
    )

    print("source: %s" % repr(source))
    print("path: %s" % source.source)
    print("coordinates: %s", source.coordinates)

    # native coordinates
    o1 = source.eval(source.coordinates)

    # cropped and resampled using EPSG:4326 coordinates
    c = podpac.Coordinates([podpac.clinspace(-22, -20, 200), podpac.clinspace(-176, -174, 200)], dims=["lat", "lon"])
    o2 = source.eval(c)

    # -------------------------------------------------------------------------
    # modis tile with time
    # -------------------------------------------------------------------------

    tile = MODISTile(
        product=PRODUCTS[0], data_key="B01", horizontal="01", vertical="11", cache_ctrl=["disk"], cache_output=False
    )

    print("tile: %s" % repr(tile))
    print(
        "available dates: %s-%s (n=%d)" % (tile.available_dates[0], tile.available_dates[-1], len(tile.available_dates))
    )
    print("coordinates: %s" % tile.coordinates)

    # existing date
    assert "2020009" in tile.available_dates
    ct1 = podpac.Coordinates(["2020-01-09", c["lat"], c["lon"]], dims=["time", "lat", "lon"])
    o2 = tile.eval(ct1)

    # nearest date
    assert "2020087" not in tile.available_dates
    ct2 = podpac.Coordinates(["2020-03-27", c["lat"], c["lon"]], dims=["time", "lat", "lon"])
    o3 = tile.eval(ct2)

    # time-series
    ct3 = podpac.Coordinates([["2019-01-01", "2019-02-01", "2019-03-01"], -21.45, -174.92], dims=["time", "lat", "lon"])
    o4 = tile.eval(ct3)

    # -------------------------------------------------------------------------
    # modis compositor
    # -------------------------------------------------------------------------

    node = MODIS(product=PRODUCTS[0], data_key="B01", cache_ctrl=["disk"], cache_output=False)

    print("node: %s" % repr(node))
    print("sources: n=%d" % len(node.sources))
    print("   .e.g: %s" % repr(node.sources[0]))

    # single tile
    assert len(node.select_sources(ct2)) == 1
    o5 = node.eval(ct2)

    # time-series in a single tile
    assert len(node.select_sources(ct3)) == 1
    o6 = node.eval(ct3)

    # multiple tiles
    ct3 = podpac.Coordinates(
        ["2020-01-09", podpac.clinspace(45, 55, 200), podpac.clinspace(-80, -40, 200)], dims=["time", "lat", "lon"]
    )
    assert len(node.select_sources(ct3)) == 7
    o7 = node.eval(ct3)

    # o7.plot()
    # pyplot.show()
