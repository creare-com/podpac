# Why PODPAC?

## Problems

Observational and modeled data products from agencies such as NASA, ESA, and academic institutions encompass terabytes 
to petabytes of scientific data available for analysis, analytics, and exploitation. Unfortunately, these data sets are 
highly underutilized by the scientific community due to:
* Vast computational resource requirements 
* Disparate formats, projections, and resolutions that hinder data fusion and integrated analyses across different data sets
* Complex and disjoint data access and retrieval protocols
* Task specific and non-reusable code development processes that hinder algorithm sharing and collaboration

Moreover, NASA programs such as Earth Observing System Data and Information System (EOSDIS) are actively investigating
migration of their vast data archives to storage on commercial cloud services such as Amazon Web Services (AWS). In 
order to maximize the benefit of cloud-based data storage, it is necessary to also enable capabilities for cloud-based
data analysis and analytics so that data processing occurs "close" to where it is stored and also exploits the powerful
resources of highly distributed cloud services. However, the ability to deploy scalable, distributed cloud-based data
analyses and analytics requires a high degree of cloud computing expertise, and thus greatly exceeds the current
capabilities of typical NASA and non-NASA earth scientists.

## PODPAC
PODPAC is an open-source, standards-based Python software framework that removes major barriers to 
widespread exploitation of earth science data and cloud-based distributed data processing. PODPAC uses a pipeline-based software architecture that

1. Enables multiscale and multi-windowed access, exploration, and integration of available earth science data sets to 
   support both analysis and analytics
2. Automatically accounts for differences in underlying geospatial data formats, projections, and resolutions
3. Greatly simplifies the implementation and parallelization of geospatial data processing routines
4. Directly integrates with advanced machine learning frameworks and other open source Python libraries
5. Unifies the access, processing, and sharing of data and algorithms through easy-to-use interfaces to existing 
   NASA data repositories
6. Enables scientists with minimal expertise in cloud computing to seamlessly transition data product pipelines 
   developed on a local workstation to execution via scalable, highly distributed cloud processing 


## Other related projects
There are other excellent open source Python projects that address aspects of these problems. In fact, PODPAC uses many of these, and aims to incorporate a multitude of useful tools within our approach for dealing with geospatial data. 

* [xarray](http://xarray.pydata.org/en/stable/index.html)
* [Geopandas](http://geopandas.org/)
* [Dask](https://dask.pydata.org/en/latest/)
* [metpy](https://unidata.github.io/MetPy/latest/index.html)
* [rasterio](https://rasterio.readthedocs.io/en/latest/)
* [Pangeo](http://pangeo-data.org/)
* [OpenDataCube](https://www.opendatacube.org/)