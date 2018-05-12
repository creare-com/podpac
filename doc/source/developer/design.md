# Design

The essential class structure is captured in the following image:

<img src="Images/ClassStructure.png" width="640">

The directory structure is as follows:

* `dist`: Contains installation instructions and environments for various deployments, including cloud deployment on AWS
* `doc`: Any documentation
* `html`: HTML pages used for demonstrations
* `podpac`: The PODPAC Python library
    * `core`: The core PODPAC functionality -- contains general implementation so of classes
    * `datalib`: Library of Nodes used to access specific data sources -- this is where the SMAP node is implemented (for example)
    * `alglib`: Library of specific algorithms that may be limited to particular scientific domains
    * 
