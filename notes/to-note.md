## Install

Podpac should be installed in a conda environment because rasterio has the potential to affect your global matplotlib installation.

Create your environment (with python 2 or 3):

```bash
$ conda create -n podpac
```

Activate environment:

```bash
$ source activate podpac
```

Install the dependencies:

```bash
$ conda config --add channels conda-forge
$ conda install rasterio
$ conda install numpy scipy xarray traitlets pint, pydap, requests, beautifulsoup4
```

To install podpac, open a terminal in the root of this repository and run:

```bash
python setup.py develop
```

## Formats

- Datetime: `np.datetime64`

## Data Classes

- Parent abstract Data class:
    - `from core.data.data import Datasource`
- Required implementation methods:
    - `native_coordinates(self)`
    - `get_data(self, coordinates, coordinates_slice)`
        - coordinates = `podpac.Coordinate`
- Implemented Data classes (`from core.data.type import ...`):
    - `PyDAP`
    - `Numpy`