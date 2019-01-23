# Notes

> Note this it not included in built documentation

## Creating the Windows Installation PODPAC Conda environment

This section describes the process used to create the [PODPAC Window 10 Installation](https://s3.amazonaws.com/podpac-s3/releases/PODPAC_latest_install_windows10.zip).

These instructions only assume that you already have [git](https://git-scm.com/) installed on your Windows 10 machine. 

* Install miniconda to a `<root_folder>`\miniconda on a Windows machine
    * Install for "Just the current user"
    * Do not register Python to the operating system
* Open a Windows command prompt in the `<root_folder>` 
* Clone podpac and set up the conda environment

```bash
$ git clone https://github.com/creare-com/podpac.git
$ git clone https://github.com/creare-com/podpac_examples.git
$ cd podpac
$ git checkout tags/<version> release/<version>  # as of writing, the <version> is 0.2.0
$ cd ..
$ copy podpac\dist\local_Windows_install\* .
$ set_local_conda_path.bat

# Verify path is set correctly
$ where conda
<root_folder>\miniconda\Library\bin\conda.bat  # Should be the first entry
<root_folder>\miniconda\Scripts\conda.exe      # Should be the second entry
...                                            # May have additional entries
```

* Set up the podpac environment and install dependencies

```bash
$ conda create -n podpac python=3

# Install core dependencies
$ conda install matplotlib>=2.1 numpy>=1.14 pint>=0.8 scipy>=1.0 traitlets>=4.3 xarray>=0.10 ipython

# Install dependencies for handling various file datatype
$ conda install rasterio>=0.36 -c conda-forge
$ conda install beautifulsoup4>=4.6 h5py>=2.7 lxml>=4.2 pydap>=3.2 requests>=2.18 

# Install dependencies for AWS
$ conda install awscli>=1.11 boto3>=1.4

# Install dependencies for algorithm nodes
$ conda install numexpr>=2.6

# Install dependencies for JupyterLab and Jupyter Notebooks
$ conda install jupyterlab ipyleaflet ipywidgets ipympl nodejs -c conda-forge

# Set up jupyter lab extensions
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
$ jupyter labextension install jupyter-leaflet
$ jupyter labextension install jupyter-matplotlib
$ jupyter nbextension enable --py widgetsnbextension
$ jupyter lab build
$ python -m ipykernel install --user

# clean conda environment
$ conda clean -a -y
```

* To run a `JupyterLab` sessions in the `<root_folder>\podpac_examples\notebooks` directory, double-click on the `run_podpac_jupyterlab.bat`. This will launch a browser window in the folder where PODPAC keeps its example notebooks.
* To run an IPython console: Open up a Windows command prompt in `<root_folder>`

```bash
$ bin\run_ipython.bat
```

