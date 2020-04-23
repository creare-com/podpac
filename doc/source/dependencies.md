# Dependencies

This document provides an overview of the dependencies leveraged by PODPAC.

## Requirements

- [Python](https://www.python.org/) (3.6 or later) &mdash; [Anaconda Python Distribution](https://www.anaconda.com/distribution/#download-section) recommended

## OS Specific Requirements

If using `pip` to install, the following OS specific dependencies are required to successfully install and run PODPAC.

### Windows

> No external dependencies necessary, though using Anaconda is recommended.

### Mac

> No external dependencies necessary, though using Anaconda is recommended.

### Linux

- `build-essential`
- `python-dev`

For debian installations:

```bash
$ sudo apt-get update
$ sudo apt-get install build-essential python-dev
```

## Core Dependencies

> See [setup.py](https://github.com/creare-com/podpac/blob/master/setup.py) for the latest dependencies listing.

    "matplotlib>=2.1",
    "numpy>=1.14",
    "pint>=0.8",
    "scipy>=1.0",
    "traitlets>=4.3",
    "xarray>=0.10",
    "requests>=2.18",
    "pyproj>=2.4",
    "lazy-import>=0.2.2",
    "psutil",

- [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), [xarray](http://xarray.pydata.org/en/stable/): array handling
- [traitlets](https://github.com/ipython/traitlets): input and type handling
- [pint](https://pint.readthedocs.io/en/latest/): unit handling
- [requests](http://docs.python-requests.org/en/master/): HTTP requests
- [matplotlib](https://matplotlib.org/): plotting
- [pyproj](http://pyproj4.github.io/pyproj/stable/): coordinate reference system handling
- [psutil](https://psutil.readthedocs.io/en/latest/): cache management


## Optional Dependencies

> Optional dependencies can be [installed using `pip`](install.html#install-with-pip)

- `datatype`: Data Handling
    - [h5py](https://www.h5py.org/): interface to the HDF5 data format
    - [pydap](http://www.pydap.org/en/latest/): python support for Data Access Protocol (OPeNDAP)
    - [rasterio](https://github.com/mapbox/rasterio): read GeoTiff and other raster datasets
    - [lxml](https://github.com/lxml/lxml): read xml and html files
    - [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/): text parser and screen scraper
    - [zarr](https://zarr.readthedocs.io/en/stable/): cloud optimized storage format
- `aws`: AWS integration
    - [awscli](https://github.com/aws/aws-cli): unified command line interface to Amazon Web Services
    - [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html): Amazon Web Services (AWS) SDK for Python
    - [s3fs](https://pypi.org/project/s3fs/): Convenient Filesystem interface over S3.
- `algorithm`: Algorithm development
    - [numexpr](https://github.com/pydata/numexpr): fast numerical expression evaluator for NumPy
- `notebook`: Jupyter Notebooks
    - [jupyterlab](https://github.com/jupyterlab/jupyterlab): extensible environment for interactive and reproducible computing
    - [ipyleaflet](https://github.com/jupyter-widgets/ipyleaflet): [LeafletJS](https://leafletjs.com/) interface for jupyter notebooks
    - [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/): interactive widgets for Jupyter notebooks
    - [ipympl](https://pypi.org/project/ipympl/): matplotlib jupyter notebook extension
    - [nodejs](https://github.com/markfinger/python-nodejs): Python bindings and utils for [NodeJS](https://nodejs.org/en/)
