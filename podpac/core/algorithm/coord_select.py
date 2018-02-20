
import traitlets as tl
import numpy as np
import scipy.signal

from podpac.core.coordinate import Coordinate, UniformCoord
from podpac.core.coordinate import make_coord_delta
from podpac.core.node import Node
from podpac.core.algorithm.algorithm import Algorithm
from podpac.core.time_utils import add_time_coords

def add_coords(base, off):
    try:
        return base + off
    except TypeError as e:
        if not (isinstance(base, np.datetime64) \
                and isinstance(off, np.timedelta64)):
            raise e

        return add_time_coords(base, off) 

class ExpandCoordinates(Algorithm):
    source = tl.Instance(Node)
    native_coordinates_source = tl.Instance(Node, allow_none=True)
    input_coordinates = tl.Instance(Coordinate, allow_none=True)
    implicit_pipeline_evaluation = tl.Bool(False)

    @property
    def native_coordinates(self):
        try:
            if self.native_coordinates_source:
                return self.native_coordinates_source.native_coordinates
            else:
                return self.source.native_coordinates
        except:
            raise Exception("no native coordinates found")

    def get_expanded_coord(self, dim):
        icoords = self.input_coordinates[dim]
        
        if dim not in self.params:
            # no expansion in this dimension
            return icoords

        if len(self.params[dim]) not in [2, 3]:
            raise ValueError("Invalid expansion params for '%s'" % dim)

        # get start and stop offsets
        dstart = make_coord_delta(self.params[dim][0])
        dstop = make_coord_delta(self.params[dim][1])

        if len(self.params[dim]) == 2:
            # expand and use native coordinates
            ncoord = self.native_coordinates[dim]
            
            # TODO GroupCoord
            xcoords = [
                ncoord.select((add_coords(c, dstart), add_coords(c, dstop)))
                for c in icoords.coordinates
            ]
            xcoord = sum(xcoords[1:], xcoords[0])

        elif len(self.params[dim]) == 3:
            # or expand explicitly
            delta = make_coord_delta(self.params[dim][2])
            
            # TODO GroupCoord
            xcoords = [
                UniformCoord(add_coords(c, dstart), add_coords(c, dstop), delta)
                for c in icoords.coordinates]
            xcoord = sum(xcoords[1:], xcoords[0])

        return xcoord

    @property
    def expanded_coordinates(self):
        kwargs = {}
        for dim in self.input_coordinates.dims:
            kwargs[dim] = self.get_expanded_coord(dim)
        kwargs['order'] = self.input_coordinates.dims
        return Coordinate(**kwargs)
    
    def execute(self, coordinates, params=None, output=None):
        self.input_coordinates = coordinates
        self.params = params or {}
        self.output = output

        self.evaluated_coordinates = self.expanded_coordinates

        if self.output is None:
            self.output = self.initialize_output_array()
            
        result = self.source.execute(self.evaluated_coordinates, params)
            
        if self.output.shape is ():
            self.output.data = result
        else:
            self.output[:] = result.transpose(*self.output.dims)
        
        self.evaluated = True

        return self.output

if __name__ == '__main__':
    from podpac.core.algorithm.algorithm import Arange
    from podpac.core.data.data import DataSource
    
    coords = Coordinate(
        time='2017-09-01',
        lat=(45., 66., 4),
        lon=(-80., -70., 5),
        order=('time', 'lat', 'lon'))

    node = ExpandCoordinates(source=Arange())

    # source
    o = Arange().execute(coords)
    print(o.coords)

    # no expansion
    o = node.execute(coords)
    print(o.coords)

    # basic time expansion
    o = node.execute(coords, params={'time': ('-15,D', '0,D', '1,D') })
    print(o.coords)

    # basic spatial expansion
    o = node.execute(coords, params={'lat': (-1, 1, 0.1) })
    print(o.coords)

    # time expansion using native coordinates
    class Test(DataSource):
        def get_native_coordinates(self):
            return Coordinate(
                time=('2010-01-01', '2018-01-01', '4,h'),
                lat=(-180., 180., 1800),
                lon=(-80., -70., 1800),
                order=('time', 'lat', 'lon'))

        def get_data(self, coordinates, slc):
            node = Arange()
            return node.execute(coordinates)
    
    node = Test()
    o = node.execute(coords)
    print (o.coords)

    node = ExpandCoordinates(source=Test())
    o = node.execute(coords, params={'time': ('-15,D', '0,D')})
    print (o.coords)

    node.params={'time': ('-15,Y', '0,D', '1,Y')}
    print (node.get_expanded_coord('time'))

    o = node.execute(coords, params={'time': ('-5,M', '0,D', '1,M')})
    print (o.coords)
