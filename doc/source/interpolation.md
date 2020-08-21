# Interpolation

## Description

PODPAC allows users to specify various different interpolation schemes for nodes with
increased granularity, and even lets users write their own interpolators. 

Relevant example notebooks include:
* [Advanced Interpolation](https://github.com/creare-com/podpac-examples/blob/master/notebooks/4-advanced/interpolation.ipynb)
* [Basic Interpolation](https://github.com/creare-com/podpac-examples/blob/master/notebooks/2-combining-data/automatic-interpolation-and-regridding.ipynb)
* [Drought Monitor Data Access Harmonization Processing](https://github.com/creare-com/podpac-examples/blob/master/notebooks/examples/drought-monitor/03-data-access-harmonization-processing.ipynb)

## Examples
Consider a `DataSource` with `lat`, `lon`, `time` coordinates that we will instantiate as: 
`node = DataSource(..., interpolation=interpolation)` 

`interpolation` can be specified ...

### ...as a string

`interpolation='nearest'`
* **Descripition**: All dimensions are interpolated using nearest neighbor interpolation. This is the default, but available options can be found here: `podpac.core.interpolation.interpolation.INTERPOLATION_METHODS` .
* **Details**: PODPAC will automatically select appropriate interpolators based on the source coordinates and eval coordinates. Default interpolator orders can be found in `podpac.core.interpolation.interpolation.INTERPOLATION_METHODS_DICT` 

### ...as a dictionary

```python
interpolation = {
    'method': 'nearest',
    'params': {    # Optional. Available parameters depend on the particular interpolator
        'spatial_tolerance': 1.1,
        'time_tolerance': np.timedelta64(1, 'D')
    },
    'interpolators': [ScipyGrid, NearestNeighbor]  # Optional. Available options are in podpac.core.interpolation.interpolation.INTERPOLATORS
}
```
* **Descripition**: All dimensions are interpolated using nearest neighbor interpolation, and the type of interpolators are tried in the order specified. For applicable interpolators, the specified parameters will be used.
* **Details**: PODPAC loops through the `interpolators` list, checking if the interpolator is able to interpolate between the evaluated and source coordinates. The first capable interpolator available will be used. 

### ...as a list

```python
interpolation = [
    {
        'method': 'bilinear',
        'dims': ['lat', 'lon']
    },
    {
        'method': 'nearest',
        'dims': ['time']
    }
]
```

* **Descripition**: The dimensions listed in the `'dims'` list will used the specified method. These dictionaries can also specify the same field shown in the previous section. 
* **Details**: PODPAC loops through the `interpolation` list, using the settings specified for each dimension independently. 

## Notes and Caveats
While the API is well developed, all conceivable functionality is not. For example, while we can interpolate gridded data to point data, point data to grid data interpolation is not as well supported, and there may be errors or unexpected results. Advanced users can develop their own interpolators, but this is not currently well-documented. 

**Gotcha**: Parameters for a specific interpolator may silently be ignored if a different interpolator is automatically selected. 

