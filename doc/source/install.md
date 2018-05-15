# Install

## For developers

At the moment, only a developer installation procedure exists. 

These install notes assume you are using the anaconda python distribution.

### Setting up Conda Environment
We recommend that you create a new python 3 environment to install `podpac`:

```bash
$ conda create -n podpac python=3 anaconda   # installs all `anaconda` packages
$ conda activate podpac  # Windows
$ source activate podpac # Linux / Mac
```

### Manually install dependencies

To manually install dependencies using conda and pip:

```bash
# Core dependencies
$ conda install numpy scipy xarray traitlets matplotlib   
$ pip install pint

# Optional dependencies to support different datasource nodes
$ conda install beautifulsoup4  h5py lxml requests urllib3 certifi
$ pip install pydap

# Optional dependencies for different algorithm nodes
$ conda install numexpr 
```

We also recommend you install [rasterio](https://mapbox.github.io/rasterio/installation.html#installing-with-anaconda) from the **conda-forge** channel. Depending on your platform, this may be simpler than letting `podpac` install `rasterio` using pip.

```bash
$ conda install rasterio --channel conda-forge
```

### Installing podpac

Install `podpac` in development mode:

```bash
$ cd <install-path>
$ git clone https://github.com/creare-com/podpac.git
$ cd podpac
# Install podpac with only the core dependencies
$ pip install -e .
# development dependencies and all other dependencies
$ pip install -e .[devall]
```

*Note*: This procedure checks out the `master` branch, which is intented to be somewhat stable with working code. For bleeding edge, checkout the `develop` branch instead using `git checkout -b develop origin/develop`


