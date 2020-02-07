Examples
========

The documents provides examples for the core functionality of PODPAC as `Simple Tutorials <#simple>`_ and `interactive Jupyter Notebooks <#notebooks>`_.

See `Supported Datasets <datasets.html>`_ for links to specific dataset examples in the repository.


Simple
------

To follow along, open a Python interpreter or Jupyter notebook in the Python environment where PODPAC is installed.

.. code:: bash

    # activate the PODPAC environment, using anaconda
    $ conda activate podpac

    # open jupyter lab
    $ jupyter lab

    # or start a ipython interpreter
    $ ipython


Array of Data
~~~~~~~~~~~~~

This example builds a ``DataSource`` Node for data values located on a 11 x 21 *lat* x *lon* grid.

In the Python interpreter of Jupyter notebook, import ``podpac`` and ``numpy`` as ``np``:

.. code:: python

    In [1]: import podpac
    In [2]: import numpy as np


Define a matrix (``np.ndarray``) of data values on a 11 x 21 grid using ``numpy``:

.. code:: python

    # mock grid data
    In [3]: data = np.random.rand(11, 21)
    In [4]: data
    Out[4]:
    array([[0.23698237, 0.59569972, 0.76340507, 0.88851687, 0.01036634,
            0.87147232, 0.70486927, 0.48485402, 0.11070812, 0.01146267,
            0.89649864, 0.59089579, 0.11195743, 0.58360194, 0.15225759,
            0.99246553, 0.31122967, 0.80974094, 0.00474486, 0.00650152,
            0.08999056],
            ...
    ])


Define lat and lon coordinates that define the axis coordinates for this grid.
We'll assume the short side of the grid are lat coordinates (length 11) and the long side are lon coordinates (length 21).

.. code:: python

    # mock coordinates for data
    In [5]: lat = np.linspace(40, 50, 11)
    In [6]: lon = np.linspace(-10, 10, 21)

Assemble PODPAC ``Coordinates`` from these coordinate values.
Note the order of the ``dims`` keyword must match the shape of our data.

.. code:: python

    # create native coordinates for data
    In [7]: coords = podpac.Coordinates([lat, lon], dims=['lat', 'lon'])
    In [8]: coords
    Out[8]: coords
    Coordinates (EPSG:4326)
            lat: ArrayCoordinates1d(lat): Bounds[40.0, 50.0], N[11], ctype['midpoint']
            lon: ArrayCoordinates1d(lon): Bounds[-10.0, 10.0], N[21], ctype['midpoint']

Create a PODPAC ``Array`` Node from ``data`` and ``coords``.
An ``Array`` Node is a sub-class of ``DataSource`` Node.

.. code:: python

    # create node for data source
    In [9]: node = podpac.data.Array(source=data, native_coordinates=coords)
    In [10]: node
    Out[10]:
    Array DataSource
            source:
    [[0.23698237 0.59569972 0.76340507 0.88851687 0.01036634 0.87147232
      0.70486927 0.48485402 0.11070812 0.01146267 0.89649864 0.59089579
      0.11195743 0.58360194 0.15225759 0.99246553 0.31122967 0.80974094
      0.00474486 0.00650152 0.08999056]
        ...]]
            native_coordinates:
                    lat: ArrayCoordinates1d(lat): Bounds[40.0, 50.0], N[11], ctype['midpoint']
                    lon: ArrayCoordinates1d(lon): Bounds[-10.0, 10.0], N[21], ctype['midpoint']
            interpolation: nearest

We've successfully created a ``DataSource`` Node that describes an 11 x 21 grid of data values with lat and lon ``Coordinates``.

Evaluate the Node at arbitrary coordinates:

.. code:: python

    # Create coordinates to evaluate this node
    In [11]: other_coords = podpac.Coordinates([42.2, 5.7], dims=['lat', 'lon'])

    # Retrieve the datapoint from the array
    In [12]: output = node.eval(other_coords) 
    In [13]: output
    <xarray.UnitsDataArray (lat: 1, lon: 1)>
    array([[0.734135]])
    Coordinates:
      * lat      (lat) float64 42.2
      * lon      (lon) float64 5.7
    Attributes:
        layer_style:  <podpac.core.style.Style object at 0x000001FA86896F60>
        crs:          EPSG:4326


SMAP Data Source
~~~~~~~~~~~~~~~~

**Note**: This example uses the SMAP node, which requires a `NASA
Earth Data Account <earthdata.html>`_ with OpenDAP access
configured.

.. code:: python

    # import the library
    import podpac  

    # Create a SMAP Node
    node = podpac.datalib.smap.SMAP(username=<your_username>, password=<your_password>)  

    # Create coordinates to evaluate this node
    coords = podpac.Coordinates(['2018-01-01 12:00:00', 0, 0], dims=['time', 'lat', 'lon'])

    # Retrieve the datapoint from NSIDC's OpenDAP server
    output = node.eval(coords) 


`Notebooks <https://github.com/creare-com/podpac_examples/tree/master/notebooks>`__
---------

Interactive PODPAC examples are distributed as `example Jupyter
notebooks <https://github.com/creare-com/podpac_examples/tree/master/notebooks>`_
hosted in the `creare-com/podpac-examples <https://github.com/creare-com/podpac_examples/>`_ repository.

.. include:: example-links.inc

Download
~~~~~~~~

You can download the notebooks two ways:

-  `Download zip of podpac-examples
   repository <https://github.com/creare-com/podpac-examples/archive/master.zip>`_
   and unzip the repository to a folder
-  Clone the `podpac-examples <https://github.com/creare-com/podpac_examples>`_ repository

.. code:: bash

    $ git clone https://github.com/creare-com/podpac-examples.git

    
Run Jupyterlab
~~~~~~~~~~~~~~

If using the `standalone Window 10
Installation <install.html#windows-10>`_, run the
``run_podpac_jupyterlab.bat`` script by double-clicking its icon.

If using a different installation:

-  Make sure the optional ``notebook`` or ``all`` optional dependencies are installed

.. code:: bash

    # via pip
    $ pip install podpac[notebook]
    $ pip install podpac[all]

    # from source
    $ pip install -e .[notebook]
    $ pip install -e .[all]

-  Start a new ``JupyterLab`` session

.. code:: bash

    $ cd <podpac-examples>
    $ jupyter lab

-  Browse the example notebooks directory
   ``<podpac-examples>/notebooks/``
-  Open a notebook that you want to run
-  From the ``JupyterLab`` menu, select ``Run --> Run All``

    **Note**: Not all the examples will work due to authentication or
    private resources (with unpublished URLs) for development.
