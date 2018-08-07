
from __future__ import division, unicode_literals, print_function, absolute_import

import os
import pytest

from podpac.core.coordinate import Coordinate
from podpac.core.algorithm.algorithm import Arange
from podpac.core.pipeline.output import FileOutput, FTPOutput, S3Output

coords = Coordinate(lat=(0, 1, 10), lon=(0, 1, 10), order=['lat', 'lon'])
node = Arange()
node.execute(coords)

class TestFileOutput(object):
    def _test(self, format):
        output = FileOutput(node=node, name='test', outdir='.', format=format)
        output.write()

        assert output.path != None
        assert os.path.isfile(output.path)
        os.remove(output.path)
        
    def test_pickle(self):
        self._test('pickle')

    def test_png(self):
        # self._test('png')

        output = FileOutput(node=node, name='test', outdir='.', format='png')
        with pytest.raises(NotImplementedError):
            output.write()

    def test_geotif(self):
        # self._test('geotif')

        output = FileOutput(node=node, name='test', outdir='.', format='geotif')
        with pytest.raises(NotImplementedError):
            output.write()

class TestFTPOutput(object):
    def test(self):
        output = FTPOutput(node=node, name='test', url='none', user='none')
        with pytest.raises(NotImplementedError):
            output.write()

class TestS3Output(object):
    def test(self):
        output = S3Output(node=node, name='test', user='none', bucket='none')
        with pytest.raises(NotImplementedError):
            output.write()