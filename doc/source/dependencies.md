# Dependencies

This document provides an overview of the dependencies leveraged by PODPAC.

## Requirements

- Python (3.6 or later)
    - We suggest you use the the [Anaconda Python Distribution](https://www.anaconda.com/)

## OS Specific Requirements

If using `pip` to install, the following OS specific dependencies are required to successfully install and run PODPAC.

### Windows

> No external dependencies necessary

### Mac

> No external dependencies necessary

### Linux

- `build-essential`
- `python-dev`

For debian installations:

```bash
$ sudo apt-get update
$ sudo apt-get install build-essential python-dev
```

## Core Dependencies

> See [requirements.txt](https://github.com/creare-com/podpac/blob/develop/requirements.txt) and [setup.py](https://github.com/creare-com/podpac/blob/develop/setup.py) for the latest dependencies listing.

- Python 2.7<sup id="1">\[[1](#f1)\]</sup>, 3.5, 3.6, or 3.7
    - We suggest you use the the [Anaconda Python Distribution](https://www.anaconda.com/)
- [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), [xarray](http://xarray.pydata.org/en/stable/): array handling
- [traitlets](https://github.com/ipython/traitlets): input and type handling
- [pint](https://pint.readthedocs.io/en/latest/): unit handling
- [requests](http://docs.python-requests.org/en/master/): HTTP requests

## Optional Dependencies

> Optional dependencies can be [installed using `pip`](insatllation.html#installing-via-pip)

- Data Handling
    - [h5py](https://www.h5py.org/): interface to the HDF5 data format
    - [pydap](http://www.pydap.org/en/latest/): python support for Data Access Protocol (OPeNDAP)
    - [rasterio](https://github.com/mapbox/rasterio): read GeoTiff and other raster datasets
    - [lxml](https://github.com/lxml/lxml): read xml and html files
    - [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/): text parser and screen scraper
- AWS
    - [awscli](https://github.com/aws/aws-cli): unified command line interface to Amazon Web Services
    - [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html): Amazon Web Services (AWS) SDK for Python
- Algorithms
    - [numexpr](https://github.com/pydata/numexpr): fast numerical expression evaluator for NumPy
- Notebook
    - [jupyterlab](https://github.com/jupyterlab/jupyterlab): extensible environment for interactive and reproducible computing
    - [ipyleaflet](https://github.com/jupyter-widgets/ipyleaflet): [LeafletJS](https://leafletjs.com/) interface for jupyter notebooks
    - [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/): interactive widgets for Jupyter notebooks
    - [ipympl](https://pypi.org/project/ipympl/): matplotlib jupyter notebook extension
    - [nodejs](https://github.com/markfinger/python-nodejs): Python bindings and utils for [NodeJS](https://nodejs.org/en/)
