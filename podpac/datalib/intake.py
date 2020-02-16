from __future__ import division, unicode_literals, print_function, absolute_import

import logging
import datetime

import traitlets as tl
import numpy as np

# Helper utility for optional imports
from lazy_import import lazy_module

# Internal imports
import podpac
from podpac import Coordinates

intake = lazy_module("intake")
# lazy_module('intake.catalog.local.LocalCatalogEntry')


class IntakeCatalog(podpac.data.DataSource):
    """
    Support for Intake Catalogs (https://intake.readthedocs.io/en/latest/index.html)
    This primarily supports CSV data sources while we expand for Intake Catalogs.

    Parameters
    ----------
    uri : str, required
        Intake Catalog uri (local path to catalog yml file, or remote uri)
        See https://intake.readthedocs.io/en/latest/catalog.html#local-catalogs
    source : str, required
        Intake Catalog source
    field : str, optional,
        If source is a dataframe with multiple fields, this specifies the field to use for analysis.for
        Can be defined in the metadata in the intake catalog source.
    dims : dict, optional
        Dictionary defining the native coordinates dimensions in the intake catalog source.
        Keys are the podpac dimensions (lat, lon, time, alt) in stacked or unstacked form.
        Values are the identifiers which locate the coordinates in the datasource.
        Can be defined in the metadata in the intake catalog source.
        Examples:
            {'lat': 'lat column', 'time': 'time column'}
            {'lat_lon': ['lat column', 'lon column']}
            {'time': 'time'}
    crs : str, optional
        Coordinate reference system of the native coordinates.
        Can be defined in the metadata in the intake catalog source.


    Attributes
    ----------
    catalog : :class:`intake.catalog.Catalog`
        Loaded intake catalog class
        See https://intake.readthedocs.io/en/latest/api_base.html#intake.catalog.Catalog
    datasource : :class:`intake.catalog.local.CatalogEntry`
        Loaded intake catalog data source
        See https://intake.readthedocs.io/en/latest/api_base.html#intake.catalog.entry.CatalogEntry
    """

    # input parameters
    source = tl.Unicode().tag(attr=True)
    uri = tl.Unicode()

    # optional input parameters
    field = tl.Unicode(default_value=None, allow_none=True)
    dims = tl.Dict(default_value=None, allow_none=True)
    crs = tl.Unicode(default_value=None, allow_none=True)

    # attributes
    catalog = tl.Any()  # This should be lazy-loaded, but haven't problems with that currently
    # tl.Instance(intake.catalog.Catalog)
    datasource = tl.Any()  # Same as above
    # datasource = tl.Instance(intake.catalog.local.LocalCatalogEntry)

    @tl.default("catalog")
    def _default_catalog(self):
        return intake.open_catalog(self.uri)

    @tl.default("datasource")
    def _default_datasource(self):
        return getattr(self.catalog, self.source)

    # TODO: validators may not be necessary

    # @tl.validate('uri')
    # def _validate_uri(self, proposed):
    #     p = proposed['value']
    #     self.catalog = intake.open_catalog(p)
    #     self.datasource = getattr(self.catalog, self.source)

    # @tl.validate('source')
    # def _validate_source(self, proposed):
    #     s = proposed['value']
    #     self.datasource = getattr(self.catalog, s)

    @tl.validate("field")
    def _validate_field(self, proposed):
        f = proposed["value"]

        if self.datasource.container == "dataframe" and f is None:
            raise ValueError("Field is required when source container is a dataframe")

        return f

        # # more strict checking
        # if 'fields' not in self.datasource.metadata:
        #     raise ValueError('No fields defined in catalog metadata')
        # if f not in self.datasource.metadata['fields'].keys():
        #     raise ValueError('Field {} not defined in catalog'.format(f))

    @tl.validate("dims")
    def _validate_dims(self, proposed):
        dims = proposed["value"]

        # TODO: this needs to be improved to expand validation
        for dim in dims:
            udims = dim.split("_")
            if isinstance(dims[dim], list) and len(dims[dim]) != len(udims):
                raise ValueError(
                    'Native Coordinate dimension "{}" does not have an identifier defined'.format(dims[dim])
                )

        return dims

    @property
    def native_coordinates(self):
        """Get native coordinates from catalog definition or input dims
        """

        # look for dims in catalog
        if self.dims is None:
            if "dims" in self.datasource.metadata:
                self.dims = self.datasource.metadata["dims"]
            else:
                raise ValueError("No native coordinates dims defined in catalog or input")

        # look for crs in catalog
        if self.crs is None:
            if "crs" in self.datasource.metadata:
                self.crs = self.datasource.metadata["crs"]

        src_data = self.datasource.read()  # TODO: support subselecting data
        c_data = []

        # indentifiers are columns when container is a dataframe
        if self.datasource.container == "dataframe":
            for dim in self.dims:
                c_data.append(src_data[self.dims[dim]].values)

            return Coordinates(c_data, dims=list(self.dims.keys()))

        ## TODO: this needs to be tested
        elif self.datasource.container == "ndarray":
            for dim in self.dims:
                c_data.append(src_data[self.dims[dim]])

            return Coordinates(c_data, dims=list(self.dims.keys()))

        else:
            raise ValueError(
                "podpac does not currently support datasource container {}".format(self.datasource.container)
            )

    def get_data(self, coordinates, coordinates_index):
        """Get Data from intake catalog source definition"""

        data = self.datasource.read()  # TODO: support subselecting data

        # dataframe container
        if self.datasource.container == "dataframe":

            # look for field in catalog
            if self.field is None:
                if "field" in self.datasource.metadata:
                    self.field = self.datasource.metadata["field"]
                else:
                    raise ValueError("No field defined in catalog or input")

            data = data[self.field]

        # create UnitDataArray with subselected data (idx)
        uda = self.create_output_array(coordinates, data=data[coordinates_index])
        return uda
