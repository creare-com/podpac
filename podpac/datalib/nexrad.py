"""
NEXRAD Support
"""



from podpac.data import DataSource, S3


@S3
class NexradSource(DataSource):
    """DataSource to handle single Nexrad file

    Hosted on AWS S3 at https://noaa-nexrad-level2.s3.amazonaws.com/
    """
    bucket = 'noaa-nexrad-level2'


# class NexradS3(S3):
#     """Nexrad S3 handling
#     """

#     s3_bucket = 'noaa-nexrad-level2'
#     node_class = NexradSource

