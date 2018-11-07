# Examples

Most of the examples for using PODPAC are in the [the example notebooks](https://github.com/creare-com/podpac/tree/master/doc/notebooks) on Github. Github will render most of these notebooks for you, and we save the expected outputs. To run these notebooks yourself:

* If using the provided `Window 10 Installation`, just run the `run_podpac_jupyterlab.bat` script by double-clicking its icon
* If using a different installation: 
    * Make sure the optional `notebook` or `all` dependencies are installed
```bash
pip install -e .[notebook]  # If installing locally
pip install podpac[notebook]  # otherwise
```
    * Start a new `JupyterLab` session
```bash
jupyter lab
``` 
    * Browse the example notebooks directory `<podpac_root>/doc/notebook`
    * Open a notebook that you want to run
    * From the `JupyterLab` menu, select `Run-->Run All`

> **Note**: Not all the examples will work because we use internal WCS sources (with unpublished URLs) for development. 

A quick example to get started is shown below.

> **Note**: This example uses our SMAP node, which requires a [NASA Earth Data Account](user/earthdata)

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

