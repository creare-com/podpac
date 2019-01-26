# Examples

PODPAC examples are distributed as [example Jupyter notebooks](https://github.com/creare-com/podpac_examples/tree/master/notebooks) hosted in the <a href="https://github.com/creare-com/podpac" class="fa fa-github"> creare-com/podpac-examples</a> repository.
Github will render most of these notebooks for you, showing the expected outputs. 

See [Supported Datasets](datasets.html) for links to specific dataset examples in the repository.

## Notebooks
*  [Introduction.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/Introduction.ipynb)
* `.ipynb_checkpoints`
   * [Introduction-checkpoint.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/.ipynb_checkpoints/Introduction-checkpoint.ipynb)
* `basic_examples`
   * [001-open-raster-file.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/basic_examples/001-open-raster-file.ipynb)
   * [002-open-point-file.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/basic_examples/002-open-point-file.ipynb)
   * [003-combining-data-in-algorithm.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/basic_examples/003-combining-data-in-algorithm.ipynb)
   * [analyzing-SMAP-data.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/basic_examples/analyzing-SMAP-data.ipynb)
   * [composite-array-datasources.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/basic_examples/composite-array-datasources.ipynb)
   * [retrieving-SMAP-data.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/basic_examples/retrieving-SMAP-data.ipynb)
   * [running-on-aws-lambda.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/basic_examples/running-on-aws-lambda.ipynb)
   * [using-coordinates.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/basic_examples/using-coordinates.ipynb)
   * [working-with-SMAP-Sentinel-data.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/basic_examples/working-with-SMAP-Sentinel-data.ipynb)
* `demos`
   * [gfs.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/demos/gfs.ipynb)
   * [pipeline-from-JSON.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/demos/pipeline-from-JSON.ipynb)
   * [SMAP-downscaling-example-application.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/demos/SMAP-downscaling-example-application.ipynb)
   * [SMAP-level4-data-access.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/demos/SMAP-level4-data-access.ipynb)
   * [SMAP-widget-data-access.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/demos/SMAP-widget-data-access.ipynb)
   * [Terrain-Tiles-Pattern-Match.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/demos/Terrain-Tiles-Pattern-Match.ipynb)
   * [Terrain-Tiles.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/demos/Terrain-Tiles.ipynb)
* `developer`
   * [Array-DataSource.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/developer/Array-DataSource.ipynb)
   * [aws-lambda-tutorial.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/developer/aws-lambda-tutorial.ipynb)
   * [Coordinates.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/developer/Coordinates.ipynb)
   * [Interpolation.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/developer/Interpolation.ipynb)
   * [traitlets-tutorial.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/developer/traitlets-tutorial.ipynb)
   * [UnitsDataArray.ipynb](https://github.com/creare-com/podpac-examples/blob/develop/notebooks/developer/UnitsDataArray.ipynb)
### Download Notebooks

* [Download zip of podpac-examples repository](https://github.com/creare-com/podpac-examples/archive/master.zip) and unzip the repository to a folder
* Clone the [podpac-examples repository](https://github.com/creare-com/podpac_examples)

```bash
$ git clone https://github.com/creare-com/podpac-examples.git
```

### Run Jupyterlab

If using the [provided standalone Window 10 Installation](/install.html#window-10), run the `run_podpac_jupyterlab.bat` script by double-clicking its icon.

If using a different installation:

* Make sure the optional `notebook` or `all` dependencies are installed

```bash
# via pip
$ pip install podpac[notebook]
$ pip install podpac[all]

# from source
$ pip install -e .[notebook]
$ pip install -e .[all]
```

* Start a new `JupyterLab` session

```bash
$ cd <podpac-examples>
$ jupyter lab
```

* Browse the example notebooks directory `<podpac-examples>/notebooks/`
* Open a notebook that you want to run
* From the `JupyterLab` menu, select `Run --> Run All`

> **Note**: Not all the examples will work due to authentication or private resources (with unpublished URLs) for development. 

## Simple Examples

### Array of Data

```python
import podpac

# mock data
data = np.random.rand(21, 21)
lat = podpac.clinspace(-10, 10, 21)
lon = podpac.clinspace(-10, 10, 21)

# create native coordinates for data
native_coords = podpac.Coordinates([lat, lon], ['lat', 'lon'])

# create node for data source
node = podpac.data.Array(source=data, native_coordinates=native_coords)

# Create coordinates to evaluate this node
c = podpac.Coordinates([5, 5], dims=['lat', 'lon'])

# Retrieve the datapoint from the array
o = node.eval(c) 
```

### SMAP Data Source

> **Note**: This example uses our SMAP node, which requires a [NASA Earth Data Account](user/earthdata) with OpenDAP access configured. 

```python
# import the library
import podpac  

# Create a SMAP Node
n = podpac.datalib.smap.SMAP(username=<your_username>, password=<your_password>)  

# Create coordinates to evaluate this node
c = podpac.Coordinates(['2018-01-01 12:00:00', 0, 0], dims=['time', 'lat', 'lon'])

# Retrieve the datapoint from NSIDC's OpenDAP server
o = n.eval(c) 
```