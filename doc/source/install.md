# Installation

PODPAC is available for Windows, Mac, and Linux.

Select the installation method the best suits your development environment:

- [pip](#install-with-pip): Recommended for most users
- [Docker](#docker): For use in containers
- [Install from source](#install-from-source): For development
- [Windows Standalone distribution](#standalone-windows-distribution): Includes Python and all dependencies

## Install with pip

### Requirements

Confirm you have the required dependencies installed on your computer:

- [Python](https://www.python.org/) (3.6 or later) &mdash; [Anaconda Python Distribution](https://www.anaconda.com/distribution/#download-section) recommended
- See [operating system requirements](dependencies.html#os-specific-requirements)

### Environment

If using Anaconda Python, create a PODPAC dedicated Anconda environment:

```
# create environment with all `anaconda` packages
$ conda create -n podpac python=3 anaconda

# activate environment
$ conda activate podpac
```

If using a non-Anaconda Python distribution, create a PODPAC dedicated virtual environment:

```
# create environment in <DIR>
$ python3 -m venv <DIR>

# activate environment
$ source <DIR>/bin/activate
```

### Install

After activating the virtual environment, install using `pip` with one of the following commands:

```
$ pip install podpac                # base installation
$ pip install podpac[datatype]      # install podpac and optional data handling dependencies
$ pip install podpac[notebook]      # install podpac and optional notebook dependencies
$ pip install podpac[aws]           # install podpac and optional aws dependencies
$ pip install podpac[algorithm]     # install podpac and optional algorithm dependencies
$ pip install podpac[all]           # install podpac and all optional dependencies
```

See [Optional Dependencies](dependencies.html#optional-dependencies) more information on optional PODPAC dependencies.


## Docker

This guide assumes you have a working installation of Docker on your computer or server.

- [Install for Linux](https://docs.docker.com/install/linux/docker-ce/debian/)
- [Install for Mac](https://docs.docker.com/docker-for-mac/install/)
- [Install for Windows](https://docs.docker.com/docker-for-windows/install/)

Once you have Docker installed, the following steps will allow you to run PODPAC within an interactive Docker shell:

- Download the [PODPAC Dockerfile](https://github.com/creare-com/podpac/blob/master/Dockerfile) from the repository
- From the directory where you downloaded the `Dockerfile`, run:

```
# build the docker image with the tag `podpac`
$ docker build -t podpac .
```

- Run the built image

```
# run the docker image in an interactive shell
$ docker run -i -t podpac
```


## Standalone Windows Distribution

### Windows 10

The Window 10 standalone distribution requires no pre-installed operating system or external dependencies.

- Download the latest Window 10 standalone distribution
    - [https://s3.amazonaws.com/podpac-s3/releases/PODPAC_latest_install_windows10.zip](https://s3.amazonaws.com/podpac-s3/releases/PODPAC_latest_install_windows10.zip)
        - For older versions, substitute `latest` in the url with the version number, i.e. `PODPAC_1.2.0_install_windows10.zip`
- Once downloaded, extract the zip file into a folder on your machine.
    - We recommend expanding it near the root of your drive (e.g. `C:\PODPAC`) due to long file paths that are part of the installation.

Once the folder is unzipped:

- To run an IPython session:
    - Open a Windows command prompt in this directory
    - Run the `run_ipython.bat` script
- To open the example [Jupyter notebooks](https://jupyter.org/):
    - Double click the file `run_podpac_jupyterlab.bat`
    - This will open up a Windows command prompt, and launch a JupyterLab notebook in your default web browser
    - To close the notebook, close the browser tab, and close the Windows console

To make this standalone distribution, see the [deploy notes](deploy-notes.md).

## Install from Source

### Requirements

Confirm you have the required dependencies installed on your computer:

- [git](https://git-scm.com/)
- [Python](https://www.python.org/) (3.6 or later)
    - We recommend the [Anaconda Python Distribution](https://www.anaconda.com/distribution/#download-section)
- See [operating system requirements](dependencies.html#os-specific-requirements#os-specific-requirements)

### Environment

If using Anaconda Python, create a PODPAC dedicated Anconda environment:

```
# create environment with all `anaconda` packages
$ conda create -n podpac python=3 anaconda

# activate environment
$ conda activate podpac
```

If using a non-Anaconda Python distribution, create a PODPAC dedicated virtual environment:

```
# create environment in <DIR>
$ python3 -m venv <DIR>

# activate environment
$ source <DIR>/bin/activate
```

### Install

After activating the virtual environment, clone the [podpac repository](https://github.com/creare-com/podpac) onto your machine:

```
$ cd <install-path>
$ git clone https://github.com/creare-com/podpac.git
$ cd podpac
```

By default, PODPAC clones to the `master` branch, which is the latest stable release.

To use a previous release, checkout the `tags/<version>` reference.
For bleeding edge, checkout the `develop` branch.

```
$ git fetch origin                                  # fetch all remote branches
$ git checkout -b release/<version> tags/<version>  # checkout specific release
$ git checkout -b develop origin/develop            # latest stable version
```

From the root of the git repository, install using `pip` with one of the following commands:

```
$ pip install .                # base installation
$ pip install .[datatype]      # install podpac and optional data handling dependencies
$ pip install .[notebook]      # install podpac and optional notebook dependencies
$ pip install .[aws]           # install podpac and optional aws dependencies
$ pip install .[algorithm]     # install podpac and optional algorithm dependencies
$ pip install .[all]           # install podpac and all optional dependencies
```

See [Optional Dependencies](dependencies.html#optional-dependencies) more information on optional PODPAC dependencies.

To install PODPAC and keep installation up to date with local changes, use the option `-e` when installing:

```
$ pip install -e .          # install podpac with only core dependencies
$ pip install -e .[devall]  # install podpac and all optional dependencies
```


## Common Issues

### Rasterio

Some users may experience issues installing [rasterio](https://rasterio.readthedocs.io/en/latest/installation.html) (included in the `datatype` and `all` installations).
If you encounter issues, we recommend trying to install [rasterio](https://rasterio.readthedocs.io/en/latest/installation.html) into your active Python environment, then re-installing `podpac`.

### UnicodeDecodeError

```
UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 13: ordinal not in range(128)*
```

Some linux users may encounter this when using Ubuntu and Python 3.6.
See this [stack overflow answer](https://stackoverflow.com/a/49127686) for a solution.

### Python.h

```
psutil/_psutil_common.c:9:10: fatal error: Python.h: No such file or directory
 #include <Python.h>
          ^~~~~~~~~~
compilation terminated.
error: command 'x86_64-linux-gnu-gcc' failed with exit status 1
```

You are missing the `python-dev` header and static files.
See this [stack overflow answer](https://stackoverflow.com/a/21530768) for the appropriate solution.


