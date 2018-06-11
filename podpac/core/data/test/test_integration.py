"""
Data Source Integration Tests
"""

import pytest

@pytest.mark.integration
class TestDataSourceIntegration():

    """Test Data Source Integrations"""
    
    def test_array(self):
        """Test array data source"""
        
        from podpac.core.data.type import NumpyArray

        arr = np.random.rand(16, 11)
        lat = np.random.rand(16)
        lon = np.random.rand(16)
        coord = Coordinate(lat_lon=(lat, lon), time=(0, 10, 11), 
                           order=['lat_lon', 'time'])
        node = NumpyArray(source=arr, native_coordinates=coord)
        #a1 = node.execute(coord)

        coordg = Coordinate(lat=(0, 1, 8), lon=(0, 1, 8), order=('lat', 'lon'))
        coordt = Coordinate(time=(3, 5, 2))

        at = node.execute(coordt)
        ag = node.execute(coordg)
