# Install

These install notes assume you are using the anaconda python distribution.

We recommend that you create a new python 3 environment to install `podpac`:

```bash
$ conda create -n podpac python=3 anaconda   # installs all `anaconda` packages
```

## Windows

-  *Optional* Activate your podpac conda environment:

```bash
$ activate podpac
```

- Install [rasterio](https://mapbox.github.io/rasterio/installation.html#installing-with-anaconda) from the **conda-forge** channel

```bash
$ conda install rasterio --channel conda-forge
```

- Install `podpac` in development mode:

```bash
$ python setup.py develop
```


