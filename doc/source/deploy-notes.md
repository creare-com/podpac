# Deploy Notes for Developers

> Note this it not included in built documentation

## Checklist
* [ ] Update version number
* [ ] Update changelog
* [ ] Update windows installation (see below)
* [ ] Check all of the notebooks using the updated windows installation 
* [ ] Update the conda environment .yml file (do this by-hand with any new packages in setup.py)
* [ ] Update the explicit conda environemnt file `conda list --explicit > filename.json`
* [ ] Update the `podpac_deps.zip` and `podpac_dist.zip` for the lambda function installs

## Uploading to pypi
Run this command to create the wheel and source code tarball
```bash
$ python setup.py sdist bdist_wheel
```

In case Twine is not installed, install it

```bash
$ pip install --upgrade twine
```

Now upload the package

```bash
$ python -m twine upload dist\*
```

This will prompt you for your username and password. At the moment, only mpu and mls have privileges to upload to the PODPAC PYPI index. 

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
$ git checkout -b release/<version> tags/<version>  # as of writing, the <version> is 0.3.0
$ cd ..
$ xcopy podpac\dist\local_Windows_install\* . /E 
$ bin\set_local_conda_path.bat

# Verify path is set correctly
$ where conda
<root_folder>\miniconda\Library\bin\conda.bat  # Should be the first entry
<root_folder>\miniconda\Scripts\conda.exe      # Should be the second entry
...                                            # May have additional entries
```

* Set up the podpac environment and install dependencies

```bash
$ conda create -n podpac python=3
$ bin\activate_podpac_conda_env.bat

# Install core dependencies
$ conda install matplotlib>=2.1 numpy>=1.14 scipy>=1.0 traitlets>=4.3 xarray>=0.10 ipython psutil requests>=2.18
$ conda install pyproj>=2.2 rasterio>=1.0 -c conda-forge
$ pip install pint>=0.8 lazy-import>=0.2.2

# Install dependencies for handling various file datatype
$ # conda install rasterio>=1.0  # Installed above alongside pyproj
$ conda install beautifulsoup4>=4.6 h5py>=2.9 lxml>=4.2 zarr>=2.3 intake>=0.5
$ pip install pydap>=3.2

# Install dependencies for AWS
$ conda install boto3>=1.4 s3fs>=0.2
$ pip install awscli>=1.11

# Install dependencies for algorithm nodes
$ conda install numexpr>=2.6

# Install dependencies for JupyterLab and Jupyter Notebooks
$ conda install jupyterlab ipywidgets nodejs
$ conda install ipympl ipyleaflet -c conda-forge

# Set up jupyter lab extensions
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
$ jupyter labextension install jupyter-leaflet
$ jupyter labextension install jupyter-matplotlib
$ jupyter nbextension enable --py widgetsnbextension
$ jupyter lab build
$ ~~python -m ipykernel install --user~~

# clean conda environment
$ conda clean -a -y
# Also delete miniconda/pkgs/.trash for a smaller installation
```


* To run a `JupyterLab` sessions in the `<root_folder>\podpac_examples\notebooks` directory, double-click on the `run_podpac_jupyterlab.bat`. This will launch a browser window in the folder where PODPAC keeps its example notebooks.
* To run an IPython console: Open up a Windows command prompt in `<root_folder>`

```bash
$ run_ipython.bat
```

