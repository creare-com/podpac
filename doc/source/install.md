# Installation

PODPAC is a python library available for Windows, Mac, and Linux.
You can install the `podpac` module by downloading one of the [standalone distributions](#standalone-distributions) (which include python and all dependencies), [using pip](#installing-via-pip), or [installing from source](#installing-from-source).

## Standalone Distibutions

### Windows 10

A full Windows 10 Installation of PODPAC can be downloaded from:

- [https://s3.amazonaws.com/podpac-s3/releases/PODPAC_latest_install_windows10.zip](https://s3.amazonaws.com/podpac-s3/releases/PODPAC_latest_install_windows10.zip).

For older versions, substitute `latest` in the url with the version number:

- `0.3.0`: [https://s3.amazonaws.com/podpac-s3/releases/PODPAC_0.3.0_install_windows10.zip](https://s3.amazonaws.com/podpac-s3/releases/PODPAC_0.3.0_install_windows10.zip)

Once downloaded, extract the zip file in a folder on your machine.
We recommend expanding it near the root of your drive (e.g. `C:\PODPAC`) due to long file paths that are part of the installation.
Once the folder is unzipped:

* To run an IPython session:
    1. Open a Windows command prompt in this directory
    2. Run the `run_ipython.bat` script
* To open the example notebooks, run the `run_podpac_jupyterlab.bat` script, by double-clicking the icon
    * This will open up a Windows command prompt, and launch a JupyterLab notebook in your default web browser
    * Older browsers may not support JupyterLab, as such the url with the token can be copied and pasted from the Windows command prompt that was launched
    * To close the notebook, close the browser tab, and close the Windows console


## Dependencies

All dependencies come pre-installed in standalone distributions, or are handled by `pip` during the installation process.
Dependencies are shown below for reference. 
See [requirements.txt](https://github.com/creare-com/podpac/blob/develop/requirements.txt) and [setup.py](https://github.com/creare-com/podpac/blob/develop/setup.py) for the latest dependencies listing.

- Python 2.7<sup id="1">\[[1](#f1)\]</sup>, 3.5, 3.6, or 3.7
    - We suggest you use the the [Anaconda Python Distribution](https://www.anaconda.com/)
- [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), [xarray](http://xarray.pydata.org/en/stable/): array handling
- [traitlets](https://github.com/ipython/traitlets): input and type handling
- [pint](https://pint.readthedocs.io/en/latest/): unit handling
- [requests](http://docs.python-requests.org/en/master/): HTTP requests

### Optional Dependencies

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

## Installing via pip

If you are using the [Anaconda Python Distribution](https://www.anaconda.com/),
we recommend that you create a new python 3 environment to install `podpac`:

```bash
$ conda create -n podpac python=3 anaconda      # installs all `anaconda` packages
$ conda activate podpac                         # Windows
$ source activate podpac                        # Linux / Mac
```

Once in the desired environment, install via `pip` using one of the commands:


```bash
$ pip install podpac                # base installation
$ pip install podpac[datatype]      # install podpac and optional data handling dependencies
$ pip install podpac[notebook]      # install podpac and optional notebook dependencies
$ pip install podpac[aws]           # install podpac and optional aws dependencies
$ pip install podpac[algorithm]     # install podpac and optional algorithm dependencies
$ pip install podpac[all]           # install podpac and all optional dependencies
```

PODPAC's dependencies are automatically installed through `pip` when `podpac` is installed.
Some dependencies are more difficult to install on certain systems. 

### Rasterio Installation

Some users may experience issues installing [rasterio](https://rasterio.readthedocs.io/en/latest/installation.html) (included in the `datatype` and `all` installations).
If you encounter issues, we recommend trying to install [rasterio](https://rasterio.readthedocs.io/en/latest/installation.html) into your activate environment, then re-installing `podpac`.
This may be simpler than letting `podpac` install `rasterio` using `pip`:

### Notebook Installation

> This step may not be necessary with the latest version of Anaconda and JupyterLab

If you are installing the optional `notebook` dependencies, you also need to run the following commands to install the **JupyterLab** plugins we use:

```bash
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
$ jupyter labextension install jupyter-leaflet
$ jupyter labextension install jupyter-matplotlib
$ jupyter nbextension enable --py widgetsnbextension
$ jupyter lab build
$ python -m ipykernel install --user
```

## Installing from Source

Clone the [podpac repository](https://github.com/creare-com/podpac) onto your machine:

```bash
$ cd <install-path>
$ git clone https://github.com/creare-com/podpac.git
$ cd podpac
```

Checkout the latest or desired version:

```bash
$ git checkout tags/<version> release/<version>  
```
For example to checkout version 0.3.0 use:

```bash
$ git checkout tags/0.3.0 release/0.3.0
```

To install podpac with only the core dependencies, execute the following from the `podpac` folder:

```bash
$ pip install .    # install podpac with only core dependencies
```

To install podpac with optional dependencies, use:

```bash
$ pip install .[datatype]      # install podpac and optional data handling dependencies
$ pip install .[notebook]      # install podpac and optional notebook dependencies
$ pip install .[aws]           # install podpac and optional aws dependencies
$ pip install .[algorithm]     # install podpac and optional algorithm dependencies
$ pip install .[all]           # install podpac and all optional dependencies
```

See sections on [Rasterio Installation](#rasterio-installation) and [Notebook installation](#notebook-installation) for more information.

### For Developers

The `master` branch is intented to be somewhat stable with working code.
For bleeding edge, checkout the `develop` branch instead:

```bash
$ git checkout -b develop origin/develop
```

To install popac and keep installation up to date with local changes, execute the following from the `podpac` folder:

```bash
$ pip install -e .          # install podpac with only core dependencies
$ pip install -e .[devall]  # install podpac and all optional dependencies
```

<small>
\[[1](#a1)\] <span id="f1"></span>
We strongly suggest using Python 3 for all future development.
Many of the libraries PODPAC utilizes will be dropping support for Python 2 starting in 2019.
For more information see the following references:

- [Python 3 Statement](http://www.python3statement.org/)
- [Porting to Python 3](https://docs.python.org/3/howto/pyporting.html)

</small>