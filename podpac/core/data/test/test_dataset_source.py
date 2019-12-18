import pytest
from podpac.core.data.file import DatasetSource


class TestDatasetSource(object):
    """ test csv data source
    """

    def test_close(self):
        with pytest.raises(NotImplementedError):
            DatasetSource()
