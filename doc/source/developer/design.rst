
Design
======

The essential class structure is captured in the following image:

.. image:: /_static/img/class-structure.png
    :width: 60%

The directory structure is as follows:

- ``dist``: Contains installation instructions and environments for various deployments, including cloud deployment on AWS
- ``doc``: Sphinx based documentation
    - ``source``: sphinx docs source
    - ``notebooks``: example jupyter notebooks
- ``html``: HTML pages used for demonstrations
- ``podpac``: The PODPAC Python library
    - ``core``: The core PODPAC functionality -- contains general implementation so of classes
    - ``datalib``: Library of Nodes used to access specific data sources -- this is where the SMAP node is implemented (for example)
    - ``alglib``: Library of specific algorithms that may be limited to particular scientific domains
