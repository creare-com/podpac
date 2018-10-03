"""
Data Source Integration Tests
"""

import os

import numpy as np
import pytest
import boto3

import podpac
from podpac.core.coordinates import Coordinates, clinspace
from podpac.core.data.types import Array, S3, ReprojectedSource, WCS
# from podpac.datalib.smap import SMAPSentinelSource

# @pytest.mark.integration
@pytest.mark.skip('TODO: implement integration tests')
class TestDataSourceIntegration():

    """Test Data Source Integrations"""
    
    def test_array(self):
        """Test array data source"""
        
        from podpac.core.data.types import Array

        arr = np.random.rand(16, 11)
        lat = np.random.rand(16)
        lon = np.random.rand(16)
        coord = Coordinate(lat_lon=(lat, lon), time=(0, 10, 11),
                           order=['lat_lon', 'time'])
        node = Array(source=arr, native_coordinates=coord)

        coordg = Coordinate(lat=(0, 1, 8), lon=(0, 1, 8), order=('lat', 'lon'))
        coordt = Coordinate(time=(3, 5, 2))

        node.execute(coordt)
        node.execute(coordg)


    def test_s3_source(self):
        """test s3 data source"""
        source = r'SMAPSentinel/SMAP_L2_SM_SP_1AIWDV_20170801T000000_20170731T114719_094E21N_T15110_002.h5'
        s3 = S3(source=source)

        assert s3.s3_data

        s3.node_class = SMAPSentinelSource
        assert s3.node_class


        ###
        # Copied from main of s3Source
        ###
        
        # Rename files in s3 bucket
        s3_bucket = 'podpac-s3' # TODO: what bucket was this referring to?
        s3 = boto3.resource('s3').Bucket(s3_bucket)
        s3.Bucket(name=s3_bucket)
        obs = list(s3.objects.all())
        obs2 = [o for o in obs if 'SMAP_L2_SM_SP' in o.key]
        
        rootpath = obs2[0].key.split('/')[0] + '/'
        for o in obs2:
            newkey = rootpath + os.path.split(o.key)[1]
            s3.Object(newkey).copy_from(CopySource=s3_bucket + '/' + o.key)
            
        obs3 = list(s3.objects.all())
        obsD = [o for o in obs3 if 'ASOwusu' in o.key]
        for o in obsD:
            o.delete()


    def test_wcs_source(self):
        """test wcs and reprojected source"""

        #coordinates = podpac.Coordinate(lat=(45, 0, 16), lon=(-70., -65., 16),
                                        #order=['lat', 'lon'])
        coordinates = podpac.Coordinate(lat=(39.3, 39., 64), lon=(-77.0, -76.7, 64), time='2017-09-03T12:00:00', 
                                        order=['lat', 'lon', 'time'])    
        reprojected_coordinates = podpac.Coordinate(lat=(45, 0, 3), lon=(-70., -65., 3),
                                                    order=['lat', 'lon']),
        #                                           'TopographicWetnessIndexComposited3090m'),
        #          )

        # TODO: this section needs to be edited, copied from old main() of type.py
        wcs = WCS()
        o = wcs.execute(coordinates)
        reprojected = ReprojectedSource(source=wcs,
                                        reprojected_coordinates=reprojected_coordinates,
                                        interpolation='bilinear')

        from podpac.datalib.smap import SMAP
        smap = SMAP(product='SPL4SMAU.003')
        reprojected = ReprojectedSource(source=wcs,
                                        coordinates_source=smap,
                                        interpolation='nearest')    
        o2 = reprojected.execute(coordinates)

        coordinates_zoom = podpac.Coordinate(lat=(24.8, 30.6, 64), lon=(-85.0, -77.5, 64), time='2017-08-08T12:00:00', 
                                             order=['lat', 'lon', 'time'])
        o3 = wcs.execute(coordinates_zoom)



    @pytest.mark.skip('TODO: implement integration tests')
    class TestBasicInterpolation(object):

        """ Test interpolation methods"""
        
        def setup_method(self, method):
            self.coord_src = Coordinates(
                [clinspace(45, 0, 16), clinspace(-70., -65., 16), clinspace(0, 1, 2)], 
                dims=['lat', 'lon', 'time'])
            
            LON, LAT, TIME = np.meshgrid(
                self.coord_src['lon'].coordinates,
                self.coord_src['lat'].coordinates,
                self.coord_src['time'].coordinates)
            
            self.latSource = LAT
            self.lonSource = LON
            self.timeSource = TIME
            
            self.nasLat = Array(
                source=LAT.astype(float),
                native_coordinates=self.coord_src,
                interpolation='bilinear')
            
            self.nasLon = Array(
                source=LON.astype(float),
                native_coordinates=self.coord_src,
                interpolation='bilinear')

            self.nasTime = Array(source=TIME.astype(float),
                native_coordinates=self.coord_src,
                interpolation='bilinear')

        def test_raster_to_raster(self):
            coord_dst = Coordinates(
                [clinspace(5., 40., 50), clinspace(-68., -66., 100)],
                dims=['lat', 'lon'])

            oLat = self.nasLat.execute(coord_dst)
            oLon = self.nasLon.execute(coord_dst)
            
            LON, LAT = np.meshgrid(
                coord_dst['lon'].coordinates,
                coord_dst['lat'].coordinates)
            
            np.testing.assert_array_almost_equal(oLat.data[..., 0], LAT)
            np.testing.assert_array_almost_equal(oLon.data[..., 0], LON)
            
        # def test_raster_to_points(self):
        #     coord_dst = Coordinates(lat_lon=((5., 40), (-68., -66), 60))
        #     oLat = self.nasLat.execute(coord_dst)
        #     oLon = self.nasLon.execute(coord_dst)
            
        #     LAT = coord_dst.coords['lat_lon']['lat']
        #     LON = coord_dst.coords['lat_lon']['lon']
            
        #     np.testing.assert_array_almost_equal(oLat.data[..., 0], LAT)
        #     np.testing.assert_array_almost_equal(oLon.data[..., 0], LON)
