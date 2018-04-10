# Install

## For developers
At the moment, only a developer installation procedure exists. 

These install notes assume you are using the anaconda python distribution.

We recommend that you create a new python 3 environment to install `podpac`:

```bash
$ conda create -n podpac python=3 anaconda   # installs all `anaconda` packages
```

Installation procedes as follows: 

-  *Optional* Activate your podpac conda environment:

```bash
$ conda activate podpac  # Windows
$ source activate podpac # Linux / Mac
```
- Install dependencies

```bash
# Core dependencies
$ conda install numpy scipy xarray traitlets    
$ pip install pint

# Optional dependencies
$ conda install numexpr requests beautifulsoup4 lxml urllib3 certifi
$ pip install pydap

# development dependencies
$ conda install pylint pytest coverage
```

- Install [rasterio](https://mapbox.github.io/rasterio/installation.html#installing-with-anaconda) from the **conda-forge** channel

```bash
$ conda install rasterio --channel conda-forge
```

- Install `podpac` in development mode:

```bash
$ cd <install-path>
$ git clone https://github.com/creare-com/podpac.git
$ cd podpac
$ python setup.py develop
```
*Note*: This procedure checks out the `master` branch, which is intented to be somewhat stable with working code. For bleeding edge, checkout the `develop` branch instead using `git checkout -b develop origin/develop`

