from __future__ import division, unicode_literals, print_function, absolute_import

import xarray as xr
import pytest
import numpy as np

import podpac
from podpac.core.data.datasource import DataSource
from podpac.core.data.array_source import Array
from podpac.core.algorithms.utility import Arange
from podpac.core.algorithms.coord_select import ExpandCoordinates, SelectCoordinates, YearSubstituteCoordinates


def setup_module(module):
    global COORDS
    COORDS = podpac.Coordinates(
        ["2017-09-01", podpac.clinspace(45, 66, 4), podpac.clinspace(-80, -70, 5)], dims=["time", "lat", "lon"]
    )


class MyDataSource(DataSource):
    coordinates = podpac.Coordinates(
        [
            podpac.crange("2010-01-01", "2018-01-01", "4,h"),
            podpac.clinspace(-180, 180, 6),
            podpac.clinspace(-80, -70, 6),
        ],
        dims=["time", "lat", "lon"],
    )

    def get_data(self, coordinates, slc):
        node = Arange()
        return node.eval(coordinates)


# TODO add assertions to tests
class TestExpandCoordinates(object):
    def test_no_expansion(self):
        node = ExpandCoordinates(source=Arange())
        o = node.eval(COORDS)

    def test_time_expansion(self):
        node = ExpandCoordinates(source=Arange(), time=("-5,D", "0,D", "1,D"))
        o = node.eval(COORDS)

    def test_spatial_expansion(self):
        node = ExpandCoordinates(source=Arange(), lat=(-1, 1, 0.1))
        o = node.eval(COORDS)

    def test_time_expansion_implicit_coordinates(self):
        node = ExpandCoordinates(source=MyDataSource(), time=("-15,D", "0,D"))
        o = node.eval(COORDS)

        node = ExpandCoordinates(source=MyDataSource(), time=("-15,Y", "0,D", "1,Y"))
        o = node.eval(COORDS)

        node = ExpandCoordinates(source=MyDataSource(), time=("-5,M", "0,D", "1,M"))
        o = node.eval(COORDS)

        # Behaviour a little strange on these?
        node = ExpandCoordinates(source=MyDataSource(), time=("-15,Y", "0,D", "4,Y"))
        o = node.eval(COORDS)

        node = ExpandCoordinates(source=MyDataSource(), time=("-15,Y", "0,D", "13,M"))
        o = node.eval(COORDS)

        node = ExpandCoordinates(source=MyDataSource(), time=("-144,M", "0,D", "13,M"))
        o = node.eval(COORDS)

    def test_spatial_expansion_ultiple_outputs(self):
        multi = Array(source=np.random.random(COORDS.shape + (2,)), coordinates=COORDS, outputs=["a", "b"])
        node = ExpandCoordinates(source=multi, lat=(-1, 1, 0.1))
        o = node.eval(COORDS)


class TestSelectCoordinates(object):
    def test_no_expansion(self):
        node = SelectCoordinates(source=Arange())
        o = node.eval(COORDS)

    def test_time_selection(self):
        node = SelectCoordinates(source=Arange(), time=("2017-08-01", "2017-09-30", "1,D"))
        o = node.eval(COORDS)

    def test_spatial_selection(self):
        node = SelectCoordinates(source=Arange(), lat=(46, 56, 1))
        o = node.eval(COORDS)

    def test_time_selection_implicit_coordinates(self):
        node = SelectCoordinates(source=MyDataSource(), time=("2011-01-01", "2011-02-01"))
        o = node.eval(COORDS)

        node = SelectCoordinates(source=MyDataSource(), time=("2011-01-01", "2017-01-01", "1,Y"))
        o = node.eval(COORDS)

    def test_spatial_selection_multiple_outputs(self):
        multi = Array(source=np.random.random(COORDS.shape + (2,)), coordinates=COORDS, outputs=["a", "b"])
        node = SelectCoordinates(source=multi, lat=(46, 56, 1))
        o = node.eval(COORDS)


class TestYearSubstituteCoordinates(object):
    def test_year_substitution(self):
        node = YearSubstituteCoordinates(source=Arange(), year="2018")
        o = node.eval(COORDS)
        assert o.time.dt.year.data[0] == 2018
        assert not np.array_equal(o["time"], COORDS["time"].coordinates)

    def test_year_substitution_orig_coords(self):
        node = YearSubstituteCoordinates(source=Arange(), year="2018", substitute_eval_coords=True)
        o = node.eval(COORDS)
        assert o.time.dt.year.data[0] == xr.DataArray(COORDS["time"].coordinates).dt.year.data[0]
        np.testing.assert_array_equal(o["time"], COORDS["time"].coordinates)

    def test_year_substitution_missing_coords(self):
        source = Array(
            source=[[1, 2, 3], [4, 5, 6]],
            coordinates=podpac.Coordinates(
                [podpac.crange("2018-01-01", "2018-01-02", "1,D"), podpac.clinspace(45, 66, 3)], dims=["time", "lat"]
            ),
        )
        node = YearSubstituteCoordinates(source=source, year="2018")
        o = node.eval(COORDS)
        assert o.time.dt.year.data[0] == 2018
        assert o["time"].data != xr.DataArray(COORDS["time"].coordinates).data

    def test_year_substitution_missing_coords_orig_coords(self):
        source = Array(
            source=[[1, 2, 3], [4, 5, 6]],
            coordinates=podpac.Coordinates(
                [podpac.crange("2018-01-01", "2018-01-02", "1,D"), podpac.clinspace(45, 66, 3)], dims=["time", "lat"]
            ),
        )
        node = YearSubstituteCoordinates(source=source, year="2018", substitute_eval_coords=True)
        o = node.eval(COORDS)
        assert o.time.dt.year.data[0] == 2017
        np.testing.assert_array_equal(o["time"], COORDS["time"].coordinates)

    def test_year_substitution_multiple_outputs(self):
        multi = Array(source=np.random.random(COORDS.shape + (2,)), coordinates=COORDS, outputs=["a", "b"])
        node = YearSubstituteCoordinates(source=multi, year="2018")
        o = node.eval(COORDS)
        assert o.time.dt.year.data[0] == 2018
        assert not np.array_equal(o["time"], COORDS["time"].coordinates)
