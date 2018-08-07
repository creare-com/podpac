# Install

At the moment, only a developer installation procedure exists.

These instructions assume you are using the [Anaconda Python distribution](https://www.anaconda.com/).

## Conda Environment

We recommend that you create a new python 3 environment to install `podpac`:

```bash
$ conda create -n podpac python=3 anaconda   # installs all `anaconda` packages
$ conda activate podpac  # Windows
$ source activate podpac # Linux / Mac
```

## Users

... tbd ...

## Developers

### Clone the Repository

Clone the [podpac repository](https://github.com/creare-com/podpac) onto your machine:

```bash
$ cd <install-path>
$ git clone https://github.com/creare-com/podpac.git
$ cd podpac
```

The `master` branch is intented to be somewhat stable with working code. For bleeding edge, checkout the `develop` branch instead:

```bash
$ git checkout -b develop origin/develop
```

### Installing podpac

After cloning the repository to your computer, install `podpac` in development mode using pip:

```bash
# Install podpac with only the core dependencies
$ pip install -e .

# development dependencies and all other dependencies
$ pip install -e .[devall]
```

Some users may experience issues installing [rasterio]https://rasterio.readthedocs.io/en/latest/installation.html#installing-with-anaconda) (included in the `datatype`, `all`, and `devall` installations). If you encounter issues, we recommend trying to install from the **conda-forge** channel. Depending on your platform, this may be simpler than letting `podpac` install `rasterio` using pip:

```bash
$ conda install rasterio --channel conda-forge
```
