# Units

This document describes how units are handled by PODPAC's `units` module.
Units in PODPAC are built off of [pint](https://pint.readthedocs.io/en/stable/index.html).

First, an overview of `pint` is given. Then, we dive into (with examples!) using units in PODPAC. Finally, we go into depth on PODPAC's implementation of units with `UnitsDatArray`s. 

## pint Overview

`pint` uses a `UnitRegistry` object to load all units and handle unit conversions. The `UnitRegistry` is used to create `Quantities`, which consist of `magnitude`, `units`, and `dimensionality`. Units can be used either by using an attribute of the `UnitRegistry` object or through a string parser, which PODPAC makes use of to abstract the `UnitRegistry`. Unit conversion is done in the back-end via the `UnitRegistry`.


## Using Units in PODPAC

When using PODPAC, you may access unit information by importing `podpac.units`, which abstracts `pint`'s functionality.  

### PODPAC Quantities
Creating a quantity is simple with PODPAC's units. Multiply desired magnitude value(s) with `podpac.units`'s parser. 

```
q = 1 * podpac.units("feet")
```

Pass the desired units to the parser. Available units: [default registry](https://github.com/hgrecco/pint/blob/master/pint/default_en.txt).

You can also apply units to arrays (including numpy).
```
arr = np.array([1,2,3,4]) * podpac.units("meter")
```

Converting units can be done using `pint`'s `.to()` implementation.

```
In [.]: arr.to("feet")
Out[.]: array([3.2808399 , 6.56167979, 3.2808399 , 3.2808399 ]) <Unit('foot')>
```

## UnitsDataArray
