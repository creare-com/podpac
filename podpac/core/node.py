"""
Node Summary
"""

from __future__ import division, print_function, absolute_import

import os
import glob
import shutil
import inspect
from collections import OrderedDict
from io import BytesIO
import base64
import json
import numpy as np
import traitlets as tl
import matplotlib


try:
    import cPickle  # Python 2.7
except:
    import _pickle as cPickle

try:
    import boto3
except:
    boto3 = None

from podpac import settings
from podpac import Units, UnitsDataArray
from podpac import Coordinate


class NodeException(Exception):
    """Summary
    """

    pass

class Style(tl.HasTraits):
    """Summary

    Attributes
    ----------
    clim : TYPE
        Description
    cmap : TYPE
        Description
    enumeration_colors : TYPE
        Description
    enumeration_legend : TYPE
        Description
    is_enumerated : TYPE
        Description
    name : TYPE
        Description
    units : TYPE
        Description
    """

    def __init__(self, node=None, *args, **kwargs):
        if node:
            self.name = self.node.__class.__name__
            self.units = self.node.units
        super(Style, self).__init__(*args, **kwargs)

    name = tl.Unicode()
    units = Units(allow_none=True)

    is_enumerated = tl.Bool(default_value=False)
    enumeration_legend = tl.Tuple(trait=tl.Unicode)
    enumeration_colors = tl.Tuple(trait=tl.Tuple)

    clim = tl.List(default_value=[None, None])
    cmap = tl.Instance(matplotlib.colors.Colormap)
    tl.default('cmap')

    def _cmap_default(self):
        return matplotlib.cm.get_cmap('viridis')


class Node(tl.HasTraits):
    """Summary

    Attributes
    ----------
    cache_type : TYPE
        Description
    dtype : TYPE
        Description
    evaluated : TYPE
        Description
    evaluated_coordinates : TYPE
        Description
    implicit_pipeline_evaluation : TYPE
        Description
    native_coordinates : TYPE
        Description
    node_defaults : TYPE
        Description
    output : TYPE
        Description
    params : TYPE
        Description
    style : TYPE
        Description
    units : TYPE
        Description
    """

    output = tl.Instance(UnitsDataArray, allow_none=True, default_value=None)
    @tl.default('output')
    def _output_default(self):
        return self.initialize_output_array('nan')

    native_coordinates = tl.Instance('podpac.core.coordinate.Coordinate',
                                     allow_none=True)
    evaluated = tl.Bool(default_value=False)
    implicit_pipeline_evaluation = tl.Bool(default_value=True, help="Evaluate the pipeline implicitly (True, Default)")
    evaluated_coordinates = tl.Instance('podpac.core.coordinate.Coordinate',
                                        allow_none=True)
    params = tl.Dict(default_value=None, allow_none=True)
    units = Units(default_value=None, allow_none=True)
    dtype = tl.Any(default_value=float)
    cache_type = tl.Enum([None, 'disk', 'ram'], allow_none=True)

    node_defaults = tl.Dict(allow_none=True)

    style = tl.Instance(Style)
    @tl.default('style')
    def _style_default(self):
        return Style()

    @property
    def shape(self):
        """Summary

        Returns
        -------
        TYPE
            Description

        Raises
        ------
        NodeException
            Description
        """

        # Changes here likely will also require changes in initialize_output_array
        ev = self.evaluated_coordinates
        #nv = self._trait_values.get('native_coordinates',  None)
        # Switching from _trait_values to hasattr because "native_coordinates"
        # not showing up in _trait_values
        if hasattr(self, 'native_coordinates'):
            nv = self.native_coordinates
        else:
            nv = None
        if ev is not None and nv is not None:
            return nv.get_shape(ev)
        elif ev is not None and nv is None:
            return ev.shape
        elif nv is not None:
            return nv.shape
        else:
            raise NodeException("Cannot determine shape if "
                                "evaluated_coordinates and native_coordinates"
                                " are both None.")

    def __init__(self, **kwargs):
        """ Do not overwrite me """
        tkwargs = self._first_init(**kwargs)

        # Add default values listed in dictionary
        # self.node_defaults.update(tkwargs) <-- could almost do this...
        #                                        but don't want to overwrite
        #                                        node_defaults and want to
        #                                        ignore 'node_defaults'
        for key, val in self.node_defaults.items():
            if key == 'node_defaults':
                continue  # ignore this entry
            if key not in tkwargs:  # Only add value if not in input
                tkwargs[key] = val

        # Call traitlest constructor
        super(Node, self).__init__(**tkwargs)
        self.init()

    def _first_init(self, **kwargs):
        """Only overwrite me if absolutely necessary

        Parameters
        ----------
        **kwargs
            Description

        Returns
        -------
        TYPE
            Description
        """
        return kwargs

    def init(self):
        """Summary
        """
        pass

    def execute(self, coordinates, params=None, output=None):
        """This is the common interface used for ALL nodes. Pipelines only
        understand this and get_description.

        Parameters
        ----------
        coordinates : TYPE
            Description
        params : None, optional
            Description
        output : None, optional
            Description

        Raises
        ------
        NotImplementedError
            Description
        """
        raise NotImplementedError

    def get_output_coords(self, coords=None):
        """Summary

        Parameters
        ----------
        coords : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """

        # Changes here likely will also require changes in shape
        if coords is None:
            coords = self.evaluated_coordinates
        if not isinstance(coords, (Coordinate)):
            coords = Coordinate(coords)

        #if self._trait_values.get("native_coordinates", None) is not None:
        # Switching from _trait_values to hasattr because "native_coordinates"
        # not showing up in _trait_values
        if hasattr(self, "native_coordinates") and self.native_coordinates is not None:
            crds = self.native_coordinates.replace_coords(coords)
        else:
            crds = coords
        return crds

    def initialize_output_array(self, init_type='nan', fillval=0, style=None,
                                no_style=False, shape=None, coords=None,
                                dims=None, units=None, dtype=np.float, **kwargs):
        """Summary

        Parameters
        ----------
        init_type : str, optional
            Description
        fillval : int, optional
            Description
        style : None, optional
            Description
        no_style : bool, optional
            Description
        shape : None, optional
            Description
        coords : None, optional
            Description
        dims : None, optional
            Description
        units : None, optional
            Description
        dtype : TYPE, optional
            Description
        **kwargs
            Description

        Returns
        -------
        TYPE
            Description
        """
        crds = self.get_output_coords(coords).coords
        dims = list(crds.keys())
        return self.initialize_array(init_type, fillval, style, no_style, shape,
                                     crds, dims, units, dtype, **kwargs)

    def copy_output_array(self, init_type='nan'):
        """Summary

        Parameters
        ----------
        init_type : str, optional
            Description

        Returns
        -------
        TYPE
            Description

        Raises
        ------
        ValueError
            Description
        """
        x = self.output.copy(True)
        shape = x.data.shape

        if init_type == 'empty':
            x.data = np.empty(shape)
        elif init_type == 'nan':
            x.data = np.full(shape, np.nan)
        elif init_type == 'zeros':
            x.data = np.zeros(shape)
        elif init_type == 'ones':
            x.data = np.ones(shape)
        else:
            raise ValueError('Unknown init_type={}'.format(init_type))

        return x

    def initialize_coord_array(self, coords, init_type='nan', fillval=0,
                               style=None, no_style=False, units=None,
                               dtype=np.float, **kwargs):
        """Summary

        Parameters
        ----------
        coords : TYPE
            Description
        init_type : str, optional
            Description
        fillval : int, optional
            Description
        style : None, optional
            Description
        no_style : bool, optional
            Description
        units : None, optional
            Description
        dtype : TYPE, optional
            Description
        **kwargs
            Description

        Returns
        -------
        TYPE
            Description
        """
        return self.initialize_array(init_type, fillval, style, no_style,
                                     coords.shape, coords.coords, coords.dims,
                                     units, dtype, **kwargs)

    def initialize_array(self, init_type='nan', fillval=0, style=None,
                         no_style=False, shape=None, coords=None,
                         dims=None, units=None, dtype=np.float, **kwargs):
        """Initialize output data array

        Parameters
        ----------
        init_type : str, optional
            How to initialize the array. Options are:
                nan: uses np.full(..., np.nan) (Default option)
                empty: uses np.empty
                zeros: uses np.zeros()
                ones: uses np.ones
                full: uses np.full(..., fillval)
                data: uses the fillval as the input array
        fillval : number, optional
            used if init_type=='full' or 'data', default = 0
        style : Style, optional
            The style to use for plotting. Uses self.style by default
        no_style : bool, optional
            Default is False. If True, self.style will not be assigned to
            arr.attr['layer_style']
        shape : tuple
            Shape of array. Uses self.shape by default.
        coords : dict/list
            input to UnitsDataArray
        dims : list(str)
            input to UnitsDataArray
        units : pint.unit.Unit, optional
            Default is self.units The Units for the data contained in the
            DataArray
        dtype : type, optional
            Default is np.float. Datatype used by default
        **kwargs
            Description

        Returns
        -------
        arr : UnitsDataArray
            Unit-aware xarray DataArray of the desired size initialized using
            the method specified

        Deleted Parameters
        ------------------
        kwargs : kwargs
            other keyword arguments passed to UnitsDataArray

        Raises
        ------
        ValueError
            Description
        """

        if style is None: style = self.style
        if shape is None: shape = self.shape
        if units is None: units = self.units
        if not isinstance(coords, (dict, OrderedDict)): coords = dict(coords)

        if init_type == 'empty':
            data = np.empty(shape)
        elif init_type == 'nan':
            data = np.full(shape, np.nan)
        elif init_type == 'zeros':
            data = np.zeros(shape)
        elif init_type == 'ones':
            data = np.ones(shape)
        elif init_type == 'full':
            data = np.full(shape, fillval)
        elif init_type == 'data':
            data = fillval
        else:
            raise ValueError('Unknown init_type={}'.format(init_type))

        x = UnitsDataArray(data, coords=coords, dims=dims, **kwargs)

        if not no_style:
            x.attrs['layer_style'] = style
        if units is not None:
            x.attrs['units'] = units
        x.attrs['params'] = self.params
        return x

    def plot(self, show=True, interpolation='none', **kwargs):
        """
        Plot function to display the output

        TODO: Improve this substantially please

        Parameters
        ----------
        show : bool, optional
            Description
        interpolation : str, optional
            Description
        **kwargs
            Description
        """

        import matplotlib.pyplot as plt

        if kwargs:
            plt.imshow(self.output.data, cmap=self.style.cmap,
                       interpolation=interpolation, **kwargs)
        else:
            self.output.plot()
        if show:
            plt.show()

    @property
    def base_ref(self):
        """
        Default pipeline node reference/name in pipeline node definitions

        Returns
        -------
        TYPE
            Description
        """
        return self.__class__.__name__


    def _base_definition(self):
        """populates 'node' and 'plugin', if necessary

        Returns
        -------
        TYPE
            Description
        """
        d = OrderedDict()

        if self.__module__ == 'podpac':
            d['node'] = self.__class__.__name__
        elif self.__module__.startswith('podpac.'):
            _, module = self.__module__.split('.', 1)
            d['node'] = '%s.%s' % (module, self.__class__.__name__)
        else:
            d['plugin'] = self.__module__
            d['node'] = self.__class__.__name__

        return d

    @property
    def definition(self):
        """
        Pipeline node definition. Implemented in primary base nodes, with
        custom implementations or extensions necessary for specific nodes.

        Should be an OrderedDict with at least a 'node' attribute.

        Raises
        ------
        NotImplementedError
            Description
        """
        parents = inspect.getmro(self.__class__)
        podpac_parents = [
            '%s.%s' % (p.__module__.split('.', 1)[1:], p.__name__)
            for p in parents
            if p.__module__.startswith('podpac.')]
        raise NotImplementedError('See %s' % ', '.join(podpac_parents))

    @property
    def pipeline_definition(self):
        """
        Full pipeline definition for this node.

        Returns
        -------
        TYPE
            Description
        """

        from podpac.core.pipeline import make_pipeline_definition
        return make_pipeline_definition(self)

    @property
    def pipeline_json(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return json.dumps(self.pipeline_definition, indent=4)

    @property
    def pipeline(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        from pipeline import Pipeline
        return Pipeline(self.pipeline_definition)

    def get_hash(self, coordinates=None, params=None):
        """Summary

        Parameters
        ----------
        coordinates : None, optional
            Description
        params : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        if params is not None:
            # convert to OrderedDict with consistent keys
            params = OrderedDict(sorted(params.items()))

            # convert dict values to OrderedDict with consistent keys
            for key, value in params.items():
                if isinstance(value, dict):
                    params[key] = OrderedDict(sorted(value.items()))

        return hash((str(coordinates), str(params)))

    @property
    def evaluated_hash(self):
        """Summary

        Returns
        -------
        TYPE
            Description

        Raises
        ------
        Exception
            Description
        """
        if self.evaluated_coordinates is None:
            raise Exception("node not evaluated")

        return self.get_hash(self.evaluated_coordinates, self.params)

    @property
    def latlon_bounds_str(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        return self.evaluated_coordinates.latlon_bounds_str


    def get_output_path(self, filename, outdir=None):
        """Summary

        Parameters
        ----------
        filename : TYPE
            Description
        outdir : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        if outdir is None:
            outdir = settings.OUT_DIR

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        return os.path.join(outdir, filename)


    def write(self, name, outdir=None, format='pickle'):
        """Summary

        Parameters
        ----------
        name : TYPE
            Description
        outdir : None, optional
            Description
        format : str, optional
            Description

        Raises
        ------
        NotImplementedError
            Description
        """
        filename = '%s_%s_%s.pkl' % (
            name,
            self.evaluated_hash,
            self.latlon_bounds_str)
        path = self.get_output_path(filename, outdir=outdir)

        if format == 'pickle':
            with open(path, 'wb') as f:
                cPickle.dump(self.output, f)
        else:
            raise NotImplementedError

    def load(self, name, coordinates, params, outdir=None):
        """Summary

        Parameters
        ----------
        name : TYPE
            Description
        coordinates : TYPE
            Description
        params : TYPE
            Description
        outdir : None, optional
            Description
        """
        filename = '%s_%s_%s.pkl' % (
            name,
            self.get_hash(coordinates, params),
            coordinates.latlon_bounds_str)
        path = self.get_output_path(filename, outdir=outdir)

        with open(path, 'rb') as f:
            self.output = cPickle.load(f)

    def load_from_file(self, path):
        """Summary

        Parameters
        ----------
        path : TYPE
            Description
        """
        with open(path, 'rb') as f:
            output = cPickle.load(f)

        self.output = output
        self.evaluated_coordinates = self.output.coordinates
        self.params = self.output.attrs['params']

    def get_image(self, format='png', vmin=None, vmax=None):
        """Summary

        Parameters
        ----------
        format : str, optional
            Description
        vmin : None, optional
            Description
        vmax : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        matplotlib.use('agg')
        from matplotlib.image import imsave

        data = self.output.data.squeeze()

        if np.isnan(vmin):
            vmin = np.nanmin(data)
        if np.isnan(vmax):
            vmax = np.nanmax(data)
        if vmax == vmin:
            vmax += 1e-16

        c = (data - vmin) / (vmax - vmin)
        i = matplotlib.cm.viridis(c, bytes=True)
        im_data = BytesIO()
        imsave(im_data, i, format='png')
        im_data.seek(0)
        return base64.b64encode(im_data.getvalue())

    @property
    def cache_dir(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        basedir = settings.CACHE_DIR
        subdir = str(self.__class__)[8:-2].split('.')
        dirs = [basedir] + subdir
        return os.path.join(*dirs)

    def cache_path(self, filename):
        """Summary

        Parameters
        ----------
        filename : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        pre = str(self.source).replace('/', '_').replace('\\', '_').replace(':', '_')
        return os.path.join(self.cache_dir, pre  + '_' + filename)

    def cache_obj(self, obj, filename):
        """Summary

        Parameters
        ----------
        obj : TYPE
            Description
        filename : TYPE
            Description
        """
        path = self.cache_path(filename)
        if settings.S3_BUCKET_NAME is None or settings.CACHE_TO_S3 == False:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            with open(path, 'wb') as fid:
                cPickle.dump(obj, fid)#, protocol=cPickle.HIGHEST_PROTOCOL)
        else:
            s3 = boto3.resource('s3').Bucket(settings.S3_BUCKET_NAME)
            io = BytesIO(cPickle.dumps(obj))
            s3.upload_fileobj(io, path)

    def load_cached_obj(self, filename):
        """Summary

        Parameters
        ----------
        filename : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        path = self.cache_path(filename)
        if settings.S3_BUCKET_NAME is None or not settings.CACHE_TO_S3:
            with open(path, 'rb') as fid:
                obj = cPickle.load(fid)
        else:
            s3 = boto3.resource('s3').Bucket(settings.S3_BUCKET_NAME)
            io = BytesIO()
            s3.download_fileobj(path, io)
            io.seek(0)
            obj = cPickle.loads(io.read())
        return obj

    def clear_disk_cache(self, attr='*', node_cache=False, all_cache=False):
        """Helper function to clear disk cache.

        WARNING: This function will permanently delete cached values
        
        Parameters
        ----------
        attr : str, optional
            Default '*'. Specific attribute to be cleared for specific
            instance of this Node. By default all attributes are cleared.
        node_cache : bool, optional
            Default False. If True, will ignore `attr` and clear all attributes
            for all variants/instances of this Node.
        all_cache : bool, optional
            Default False. If True, will clear the entire podpac cache.
        """
        if all_cache:
            shutil.rmtree(settings.CACHE_DIR)
        elif node_cache:
            shutil.rmtree(self.cache_dir)
        else:
            for f in glob.glob(self.cache_path(attr)):
                os.remove(f)


if __name__ == "__main__":
    # checking creation of output node
    c1 = Coordinate(lat_lon=((0, 1, 10), (0, 1, 10)), time=(0, 1, 2))
    c2 = Coordinate(lat_lon=((0.5, 1.5, 15), (0.1, 1.1, 15)))

    n = Node(native_coordinates=c1)
    print(n.initialize_output_array().shape)
    n.evaluated_coordinates = c2
    print(n.initialize_output_array().shape)

    n = Node(native_coordinates=c1.unstack())
    print(n.initialize_output_array().shape)
    n.evaluated_coordinates = c2
    print(n.initialize_output_array().shape)

    n = Node(native_coordinates=c1)
    print(n.initialize_output_array().shape)
    n.evaluated_coordinates = c2.unstack()
    print(n.initialize_output_array().shape)
    print("Nothing to do")
