"""
Test podpac.core.data.data module
"""

import pytest

import numpy as np
from traitlets import TraitError

from podpac import DataSource


class TestDataSource(object):
    """Test podpac.core.data.data module"""

    def test_allow_missing_modules(self):
        """TODO: Allow user to be missing rasterio and scipy"""
        pass


    @pytest.mark.skip(reason="traitlets does not currently honor the `allow_none` field")
    def test_traitlets_allow_none(self):
        """TODO: it seems like allow_none = False doesn't work
        """
        with pytest.raises(TraitError):
            node = DataSource(source=None)

        with pytest.raises(TraitError):
            node = DataSource(no_data_vals=None)

    def test_traitlets_errors(self):
        """ make sure traitlet errors are reased with improper inputs """

        with pytest.raises(TraitError):
            node = DataSource(interpolation=None)

        with pytest.raises(TraitError):
            node = DataSource(interpolation='myowninterp')


    def test_methods_must_be_implemented(self):
        """These class methods must be implemented"""

        node = DataSource()

        with pytest.raises(NotImplementedError):
            node.get_native_coordinates()

        with pytest.raises(NotImplementedError):
            node.get_data(None, None)

    def test_definition(self):

        node = DataSource(source='test')
        d = node.definition

        assert d
        assert 'node' in d
        assert d['source'] == node.source
        assert d['attrs']['interpolation'] == node.interpolation



class MockDataSource(DataSource):
    """ Mock Data Source for testing
    """

    source = np.random.rand(100)

    def get_data(self, coordinates, coordinates_slice):
        """Summary
        
        Parameters
        ----------
        coordinates : TYPE
            Description
        coordinates_slice : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        pass


