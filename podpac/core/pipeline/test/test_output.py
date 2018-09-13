
from __future__ import division, unicode_literals, print_function, absolute_import

import os
import pytest

import podpac
from podpac.core.algorithm.algorithm import Arange
from podpac.core.pipeline.output import FileOutput, FTPOutput, S3Output, NoOutput, ImageOutput

coords = podpac.Coordinates([[0, 1, 2], [10, 20, 30]], dims=['lat', 'lon'])
node = Arange()
node.execute(coords)

class TestNoOutput(object):
    def test(self):
        output = NoOutput(node=node, name='test')
        output.write()

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

class TestImageOutput(object):
    def test(self):
        output = ImageOutput(node=node, name='test')
        output.write()
        assert output.image is not None